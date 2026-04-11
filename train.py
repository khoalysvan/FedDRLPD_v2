import torch
import numpy as np
import time
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from server import FederatedServer
from client import Client, MaliciousClient, ClientManager
from dqn_agent import DQNAgent
from dataset import get_datasets
from model import get_model
from malicious_score import flatten_weights


# =========================
# PCA UTILS  (module-level — safe to import)
# =========================

def reduce_updates_with_pca(delta_updates, output_dim):
    """
    delta_updates: list[np.ndarray], mỗi phần tử là vector delta full-dim.
    Trả về list[np.ndarray] với cùng độ dài list đầu vào và chiều cố định output_dim.
    """
    if len(delta_updates) == 0:
        return []

    x = np.stack(delta_updates).astype(np.float32)
    n_samples, n_features = x.shape

    # PCA cần >=2 samples để fit ổn định.
    max_components = min(output_dim, n_samples, n_features)

    if max_components >= 1 and n_samples >= 2:
        pca = PCA(
            n_components=max_components,
            svd_solver="randomized",
            random_state=42,
        )
        x_reduced = pca.fit_transform(x).astype(np.float32)
    else:
        x_reduced = x[:, :max_components].astype(np.float32)

    if max_components < output_dim:
        pad = np.zeros((x_reduced.shape[0], output_dim - max_components), dtype=np.float32)
        x_reduced = np.concatenate([x_reduced, pad], axis=1)

    return [x_reduced[i] for i in range(x_reduced.shape[0])]


# =========================
# ENTRY POINT
# — tất cả runtime code nằm trong guard này để
#   Windows multiprocessing (spawn) có thể import
#   module mà không re-execute training (Issue #9)
# =========================

if __name__ == "__main__":

    # ── CONFIG ──────────────────────────────────────────────────────
    NUM_USERS    = 20
    BATCH_SIZE   = 16
    LOCAL_EPOCH  = 5
    PCA_COMPONENTS        = 20
    REPLAY_BUFFER_RANGE   = (5000, 10000)
    REPLAY_BUFFER_CAPACITY = 8000

    NUM_CLIENTS   = NUM_USERS
    ROUNDS        = 20
    DATASET       = "cifar10"
    IID           = False
    ALPHA         = 0.5
    DQN_BATCH_SIZE = BATCH_SIZE

    # ── ATTACKER CONFIG (Issue #3) ──────────────────────────────────
    MALICIOUS_RATIO = 0.3         # 30% clients là attacker (paper: 5%-40%)
    ATTACK_TYPE     = "label_flipping"  # "label_flipping" | "backdoor" | "noise"

    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    DQN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── INIT DATASET ─────────────────────────────────────────────────
    client_datasets = get_datasets(
        name=DATASET,
        num_clients=NUM_CLIENTS,
        iid=IID,
        alpha=ALPHA,
    )

    # ── INIT SERVER + MODEL ──────────────────────────────────────────
    server = FederatedServer(
        num_clients=NUM_CLIENTS,
        device=DEVICE,
        dataset_name=DATASET,
        model=get_model(DATASET),
    )

    # ── INIT CLIENTS (benign + malicious) ─────────────────────────────
    num_malicious = max(0, int(NUM_CLIENTS * MALICIOUS_RATIO))
    malicious_ids = set(
        np.random.choice(NUM_CLIENTS, size=num_malicious, replace=False).tolist()
    )

    clients = []
    for i in range(NUM_CLIENTS):
        if i in malicious_ids:
            clients.append(
                MaliciousClient(
                    client_id=i,
                    dataset=client_datasets[i],
                    model=server.global_model,
                    attack_type=ATTACK_TYPE,
                    device=DEVICE,
                    batch_size=BATCH_SIZE,
                )
            )
        else:
            clients.append(
                Client(
                    client_id=i,
                    dataset=client_datasets[i],
                    model=server.global_model,
                    device=DEVICE,
                    batch_size=BATCH_SIZE,
                )
            )

    print(f"\n=== Attacker Setup ===")
    print(f"MALICIOUS_RATIO: {MALICIOUS_RATIO} ({num_malicious}/{NUM_CLIENTS})")
    print(f"ATTACK_TYPE:     {ATTACK_TYPE}")
    print(f"Attacker IDs:    {sorted(malicious_ids)}")
    print(f"Benign IDs:      {sorted(set(range(NUM_CLIENTS)) - malicious_ids)}")

    # ── INIT DQN ─────────────────────────────────────────────────────
    # mỗi client đóng góp: PCA_COMPONENTS + data_size + malicious_score
    state_dim = NUM_CLIENTS * (PCA_COMPONENTS + 2) + 1

    dqn = DQNAgent(
        state_dim=state_dim,
        num_clients=NUM_CLIENTS,
        select_ratio=0.5,
        device=DQN_DEVICE,
        replay_capacity=REPLAY_BUFFER_CAPACITY,
    )

    # ── INIT CLIENT MANAGER ──────────────────────────────────────────
    client_manager = ClientManager(clients, dqn_agent=None)

    # ── TRAIN LOOP INIT ──────────────────────────────────────────────
    prev_acc    = 0.0
    prev_reward = 0.0
    current_state = np.zeros(state_dim, dtype=np.float32)

    all_weights         = [np.zeros(PCA_COMPONENTS, dtype=np.float32) for _ in range(NUM_CLIENTS)]
    all_data_sizes      = [0.0] * NUM_CLIENTS
    all_malicious_scores = [0.0] * NUM_CLIENTS

    print("=== Runtime Config ===")
    print(f"DEVICE:       {DEVICE}")
    print(f"NUM_USERS:    {NUM_USERS}")
    print(f"BATCH_SIZE:   {BATCH_SIZE}")
    print(f"LOCAL_EPOCH:  {LOCAL_EPOCH}")
    print(f"PCA_COMPONENTS: {PCA_COMPONENTS}")
    print(
        f"REPLAY_BUFFER: {REPLAY_BUFFER_RANGE[0]}~{REPLAY_BUFFER_RANGE[1]} "
        f"(using {REPLAY_BUFFER_CAPACITY})"
    )

    round_bar    = tqdm(range(1, ROUNDS + 1), desc="Training Rounds", unit="round")
    avg_round_sec = None

    # ── MAIN TRAINING LOOP ───────────────────────────────────────────
    for round_idx in round_bar:
        round_start = time.perf_counter()
        tqdm.write(f"\n========== ROUND {round_idx} ==========")

        # Step 1: Broadcast global model + capture w_g^t (Eq.17)
        global_weights = server.broadcast_model()
        global_flat    = flatten_weights(global_weights)   # w_g^t cho reward Eq.(17)

        # Step 2: DQN select top-P clients
        selected_ids = dqn.select_action(current_state)

        # Step 3-4: Local train + malicious scoring
        updates_pack = client_manager.train_clients(
            global_weights,
            round_idx,
            local_epochs=LOCAL_EPOCH,
            selected_ids=selected_ids,
        )
        updates          = updates_pack["updates"]
        selected_ids     = updates_pack["selected_ids"]
        trained_clients  = len(selected_ids)
        selected_samples = int(sum(u["data_size"] for u in updates))

        # Step 5: Aggregate + evaluate + feedback
        result      = server.training_round(updates_pack)
        global_acc  = result["accuracy"]
        global_loss = result["loss"]
        feedback    = result["dqn_feedback"]

        # ── BUILD NEXT STATE S_{t+1} ─────────────────────────────────
        full_delta_list  = []
        data_sizes       = []
        malicious_scores = []
        client_ids       = []

        for u in updates:
            local_delta = flatten_weights(u["weights"]).astype(np.float32)
            full_delta_list.append(local_delta)
            data_sizes.append(float(u["data_size"]))
            malicious_scores.append(float(u["malicious_score"]))
            client_ids.append(int(u["client_id"]))

        tqdm.write(f"Selected client sizes: {data_sizes}")

        weights_list = reduce_updates_with_pca(full_delta_list, PCA_COMPONENTS)

        for i, cid in enumerate(client_ids):
            all_weights[cid]          = weights_list[i]
            all_data_sizes[cid]       = data_sizes[i]
            all_malicious_scores[cid] = malicious_scores[i]

        next_state = dqn.build_state(
            all_weights,
            all_data_sizes,
            all_malicious_scores,
            global_acc,
            client_ids=list(range(NUM_CLIENTS)),
        )

        # ── REWARD (Eq. 15-18) ───────────────────────────────────────
        # global_flat = w_g^t broadcast đầu round (trước aggregate)
        # full_delta_list = delta_i = w_i - w_g cho mỗi client được chọn
        reward = dqn.compute_reward(
            prev_reward,
            full_delta_list,
            global_flat,
            feedback["round_accuracy"],
            feedback["prev_accuracy"],
            feedback["malicious_scores"],
        )

        # ── DQN UPDATE ───────────────────────────────────────────────
        dqn.update_transition(
            curr_state=current_state,
            action=selected_ids,
            reward=reward,
            next_state=next_state,
            done=(round_idx == ROUNDS),
        )
        loss = dqn.train(batch_size=DQN_BATCH_SIZE)

        # ── LOGGING ──────────────────────────────────────────────────
        replay_size   = len(dqn.memory)
        round_sec     = time.perf_counter() - round_start

        if avg_round_sec is None:
            avg_round_sec = round_sec
        else:
            avg_round_sec = 0.9 * avg_round_sec + 0.1 * round_sec

        eta_sec = max(0.0, (ROUNDS - round_idx) * avg_round_sec)

        dqn_info = (
            f"DQN Loss: {loss:.6f} | Epsilon: {dqn.epsilon:.4f} | Replay: {replay_size}"
            if loss is not None
            else f"DQN Loss: warming up ({replay_size}/{DQN_BATCH_SIZE}) | Epsilon: {dqn.epsilon:.4f}"
        )
        tqdm.write(dqn_info)
        tqdm.write(
            f"Round {round_idx:02d} | time={round_sec:.2f}s | "
            f"clients={trained_clients} | samples={selected_samples} | "
            f"acc={global_acc:.4f} | loss={global_loss:.4f} | reward={reward:.4f}"
        )
        round_bar.set_postfix(
            cli  = f"{trained_clients}",
            smp  = f"{selected_samples}",
            acc  = f"{global_acc:.4f}",
            gl   = f"{global_loss:.4f}",
            rw   = f"{reward:.4f}",
            dqn  = f"{loss:.4f}" if loss is not None else "warmup",
            eps  = f"{dqn.epsilon:.4f}",
            rsec = f"{round_sec:.2f}s",
            eta  = f"{eta_sec/60:.1f}m",
        )

        prev_acc      = global_acc
        prev_reward   = reward
        current_state = next_state