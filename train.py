import os
# Tắt các log verbose của TensorFlow/oneDNN trước khi import tensorboard
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"   # 0=all, 1=info, 2=warning, 3=error only
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # tắt oneDNN custom ops messages

import torch
import numpy as np
import time
import datetime
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

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


def transform_updates_with_pca(delta_updates, pca_model, output_dim):
    """
    Transform updates bằng PCA đã fit trước đó (KHÔNG refit mỗi round).
    Luôn trả về vector có chiều cố định output_dim.
    """
    if len(delta_updates) == 0:
        return []

    x = np.stack(delta_updates).astype(np.float32)
    x_reduced = pca_model.transform(x).astype(np.float32)

    curr_dim = x_reduced.shape[1]
    if curr_dim < output_dim:
        pad = np.zeros((x_reduced.shape[0], output_dim - curr_dim), dtype=np.float32)
        x_reduced = np.concatenate([x_reduced, pad], axis=1)
    elif curr_dim > output_dim:
        x_reduced = x_reduced[:, :output_dim]

    return [x_reduced[i] for i in range(x_reduced.shape[0])]


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":

    # ── CONFIG ──────────────────────────────────────────────────────
    NUM_USERS    = 100
    BATCH_SIZE   = 16
    LOCAL_EPOCH  = 2
    PCA_COMPONENTS        = 50
    REPLAY_BUFFER_RANGE   = (5000, 10000)
    REPLAY_BUFFER_CAPACITY = 8000

    NUM_CLIENTS   = NUM_USERS
    ROUNDS        = 200
    DATASET       = "cifar10"
    IID           = False
    ALPHA         = 0.5
    DQN_BATCH_SIZE      = 32
    DQN_GAMMA           = 0.95
    DQN_EPSILON_START   = 1.0
    DQN_EPSILON_DECAY   = 0.97
    DQN_EPSILON_MIN     = 0.02
    DQN_WARMUP_STEPS    = 32
    DQN_TARGET_UPD_STEP = 10
    PCA_WARMUP_ROUNDS   = 5

    # ── ATTACKER CONFIG ──────────────────────────────────────────────
    MALICIOUS_RATIO = 0.3         # 30% clients là attacker (paper: 5%-40%)
    ATTACK_TYPE     = "label_flipping"  # "label_flipping" | "backdoor" | "noise"

    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    DQN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── TENSORBOARD SETUP ────────────────────────────────────────────
    run_name = (
        f"{DATASET}_"
        f"N{NUM_CLIENTS}_"
        f"mal{int(MALICIOUS_RATIO*100)}pct_{ATTACK_TYPE}_"
        f"ep{LOCAL_EPOCH}_"
        f"r{ROUNDS}_"
        f"{datetime.datetime.now().strftime('%m%d_%H%M')}"
    )
    log_dir = os.path.join("runs", run_name)
    writer  = SummaryWriter(log_dir=log_dir)
    print(f"\nTensorBoard logs → {log_dir}")
    print(f"  Xem bằng lệnh: tensorboard --logdir runs")

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

    # ── INIT CLIENTS (benign + malicious) ────────────────────────────
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

    print(f"\n=== Runtime Config ===")
    print(f"DEVICE:          {DEVICE}")
    print(f"NUM_USERS:       {NUM_USERS}")
    print(f"BATCH_SIZE:      {BATCH_SIZE}")
    print(f"LOCAL_EPOCH:     {LOCAL_EPOCH}")
    print(f"ROUNDS:          {ROUNDS}")
    print(f"PCA_COMPONENTS:  {PCA_COMPONENTS}")
    print(f"REPLAY_BUFFER:   {REPLAY_BUFFER_RANGE[0]}~{REPLAY_BUFFER_RANGE[1]} (using {REPLAY_BUFFER_CAPACITY})")
    print(f"PCA warmup:      fit-once after round {PCA_WARMUP_ROUNDS}")
    print(
        f"DQN cfg:         batch={DQN_BATCH_SIZE}, gamma={DQN_GAMMA}, "
        f"eps=({DQN_EPSILON_START}->{DQN_EPSILON_MIN}, decay={DQN_EPSILON_DECAY}), "
        f"warmup={DQN_WARMUP_STEPS}, target_upd={DQN_TARGET_UPD_STEP}"
    )

    print(f"\n=== Attacker Setup ===")
    print(f"MALICIOUS_RATIO: {MALICIOUS_RATIO} ({num_malicious}/{NUM_CLIENTS})")
    print(f"ATTACK_TYPE:     {ATTACK_TYPE}")
    print(f"Attacker IDs:    {sorted(malicious_ids)}")
    print(f"Benign IDs:      {sorted(set(range(NUM_CLIENTS)) - malicious_ids)}")

    # Ghi hparams vào TensorBoard
    writer.add_text("config/dataset",        DATASET,                      0)
    writer.add_text("config/attack_type",    ATTACK_TYPE,                  0)
    writer.add_text("config/malicious_ids",  str(sorted(malicious_ids)),   0)
    writer.add_hparams(
        hparam_dict={
            "num_clients":      NUM_CLIENTS,
            "malicious_ratio":  MALICIOUS_RATIO,
            "attack_type":      ATTACK_TYPE,
            "local_epoch":      LOCAL_EPOCH,
            "rounds":           ROUNDS,
            "batch_size":       BATCH_SIZE,
            "pca_components":   PCA_COMPONENTS,
            "replay_capacity":  REPLAY_BUFFER_CAPACITY,
            "iid":              IID,
        },
        metric_dict={"hparam/final_acc": 0.0},  # placeholder, cập nhật cuối
    )

    # ── INIT DQN ─────────────────────────────────────────────────────
    # Per-client state feature dim: PCA update + data ratio + malicious score + global acc.
    client_state_dim = PCA_COMPONENTS + 3

    dqn = DQNAgent(
        state_dim=client_state_dim,
        num_clients=NUM_CLIENTS,
        select_ratio=0.5,
        device=DQN_DEVICE,
        gamma=DQN_GAMMA,
        epsilon=DQN_EPSILON_START,
        replay_capacity=REPLAY_BUFFER_CAPACITY,
        update_target_steps=DQN_TARGET_UPD_STEP,
        epsilon_decay=DQN_EPSILON_DECAY,
        epsilon_min=DQN_EPSILON_MIN,
        warmup_steps=DQN_WARMUP_STEPS,
    )

    # ── INIT CLIENT MANAGER ──────────────────────────────────────────
    client_manager = ClientManager(clients, dqn_agent=None)

    # ── TRAIN LOOP INIT ──────────────────────────────────────────────
    prev_acc      = 0.0
    prev_rewards  = np.zeros(NUM_CLIENTS, dtype=np.float32)
    current_state = np.zeros((NUM_CLIENTS, client_state_dim), dtype=np.float32)
    reward_history_list = []

    all_weights          = [np.zeros(PCA_COMPONENTS, dtype=np.float32) for _ in range(NUM_CLIENTS)]
    all_data_sizes       = [0.0] * NUM_CLIENTS
    all_malicious_scores = [0.0] * NUM_CLIENTS
    all_full_deltas      = [None] * NUM_CLIENTS

    pca_fitted = None

    round_bar     = tqdm(range(1, ROUNDS + 1), desc="Training Rounds", unit="round")
    avg_round_sec = None

    # ── MAIN TRAINING LOOP ───────────────────────────────────────────
    for round_idx in round_bar:
        round_start = time.perf_counter()
        tqdm.write(f"\n========== ROUND {round_idx}/{ROUNDS} ==========")

        # Step 1: Broadcast global model + capture w_g^t (Eq.17)
        global_weights = server.broadcast_model()
        global_flat    = flatten_weights(global_weights)

        # Step 2: Round-1 bootstrap train all clients; from round-2 use DQN
        if round_idx == 1:
            selected_ids = list(range(NUM_CLIENTS))
            q_stats = dqn.get_q_stats(current_state)
            selection_info = {
                "epsilon": float(dqn.epsilon),
                "mode": "bootstrap_all_clients",
                "random_count": 0,
                "greedy_count": NUM_CLIENTS,
                "selected_q_mean": q_stats["q_mean"],
                **q_stats,
            }
        else:
            selected_ids_np, selection_info = dqn.select_action(current_state, return_info=True)
            selected_ids = [int(i) for i in selected_ids_np.tolist()]

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

        # ── DETECTION METRICS (ground truth) ─────────────────────────
        selected_set = set(selected_ids)
        mal_selected = [i for i in selected_ids if i in malicious_ids]
        ben_selected = [i for i in selected_ids if i not in malicious_ids]
        attacker_sel_ratio = len(mal_selected) / max(1, trained_clients)
        random_pick_ratio = selection_info["random_count"] / max(1, trained_clients)

        tp  = len(mal_selected)                            # malicious bị chọn (xấu)
        fp  = len(ben_selected)                            # benign bị loại nhầm (0 ở đây)
        fn  = len([i for i in malicious_ids if i not in selected_set])  # mal bị miss
        tn  = len([i for i in range(NUM_CLIENTS) if i not in malicious_ids and i not in selected_set])

        # Từ góc độ DEFENSE: "detected" = attacker KHÔNG được chọn (bị loại)
        detected_mal = fn   # malicious bị DQN loại ra (đúng)
        missed_mal   = tp   # malicious vẫn lọt vào (sai)
        false_excl   = tn   # benign bị loại nhầm

        tpr = detected_mal / max(1, num_malicious)  # True Positive Rate (detection rate)
        fpr = false_excl   / max(1, NUM_CLIENTS - num_malicious)  # False Positive Rate

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

        # Cập nhật memory bank full-dim deltas cho tất cả clients.
        for i, cid in enumerate(client_ids):
            all_full_deltas[cid] = full_delta_list[i]

        # Fit PCA MỘT LẦN sau warmup rounds bằng full memory bank (all clients).
        if pca_fitted is None and round_idx >= PCA_WARMUP_ROUNDS:
            bank = [v for v in all_full_deltas if v is not None]
            if len(bank) >= PCA_COMPONENTS:
                x_bank = np.stack(bank).astype(np.float32)
                n_features = x_bank.shape[1]
                n_comp = min(PCA_COMPONENTS, x_bank.shape[0], n_features)
                if n_comp >= 1:
                    pca_fitted = PCA(
                        n_components=n_comp,
                        svd_solver="randomized",
                        random_state=42,
                    )
                    pca_fitted.fit(x_bank)
                    tqdm.write(
                        f"[PCA] Fitted once at round {round_idx} "
                        f"with {x_bank.shape[0]} samples, n_components={n_comp}"
                    )

        if pca_fitted is not None:
            weights_list = transform_updates_with_pca(full_delta_list, pca_fitted, PCA_COMPONENTS)
        else:
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
        # Khôi phục local absolute: theta_l = theta_g + delta_l trước khi tính distance.
        local_abs_list = [global_flat + delta for delta in full_delta_list]
        # Dùng cùng một global reference vector cho mỗi local absolute vector.
        global_refs = [global_flat for _ in range(len(full_delta_list))]
        reward = dqn.compute_reward(
            prev_rewards,
            selected_ids,
            local_abs_list,
            global_refs,
            feedback["round_accuracy"],
            feedback["prev_accuracy"],
            feedback["malicious_scores"],
        )

        # Episode total reward: chỉ cộng rewards của clients được chọn trong round.
        episode_total_reward = float(np.sum(np.asarray(reward, dtype=np.float32)[selected_ids]))
        reward_history_list.append(episode_total_reward)

        # ── REWARD LOG (per-client + group means) ─────────────────────
        _rw_arr = np.asarray(reward, dtype=np.float32)
        _per_client_lines = []
        _ben_rewards = []
        _mal_rewards = []
        for cid in range(NUM_CLIENTS):
            rw_i = float(_rw_arr[cid])
            tag  = "[M]" if clients[cid].is_malicious else "[B]"
            sel  = "*" if cid in selected_set else " "
            _per_client_lines.append(f"  C{cid:02d} {tag}{sel} rw={rw_i:+.4f}")
            if clients[cid].is_malicious:
                _mal_rewards.append(rw_i)
            else:
                _ben_rewards.append(rw_i)
        _mean_ben = float(np.mean(_ben_rewards)) if _ben_rewards else 0.0
        _mean_mal = float(np.mean(_mal_rewards)) if _mal_rewards else 0.0
        _gap      = _mean_ben - _mean_mal
        tqdm.write("--- Per-Client Reward ---")
        for line in _per_client_lines:
            tqdm.write(line)
        tqdm.write(
            f"--- AVG  rw_benign={_mean_ben:+.4f}  rw_malicious={_mean_mal:+.4f}  "
            f"gap(B-M)={_gap:+.4f} {'OK' if _gap > 0 else 'BAD'} ---"
        )

        # ── DQN UPDATE ───────────────────────────────────────────────
        dqn.update_transition(
            curr_state=current_state,
            action=selected_ids,
            reward=reward,
            next_state=next_state,
        )
        dqn_loss = dqn.train(batch_size=DQN_BATCH_SIZE)

        # ── TENSORBOARD LOGGING ──────────────────────────────────────
        # 1) FL Performance
        writer.add_scalar("FL/global_accuracy",  global_acc,  round_idx)
        writer.add_scalar("FL/global_loss",       global_loss, round_idx)
        writer.add_scalar("FL/accuracy_delta",    global_acc - prev_acc, round_idx)

        # 2) DQN / RL
        writer.add_scalar("DQN/episode_total_reward", episode_total_reward, round_idx)
        writer.add_scalar("DQN/epsilon",          dqn.epsilon,       round_idx)
        writer.add_scalar("DQN/replay_size",      len(dqn.memory),   round_idx)
        writer.add_scalar("DQN/q_min",            selection_info["q_min"], round_idx)
        writer.add_scalar("DQN/q_max",            selection_info["q_max"], round_idx)
        writer.add_scalar("DQN/q_mean",           selection_info["q_mean"], round_idx)
        writer.add_scalar("DQN/q_std",            selection_info["q_std"], round_idx)
        writer.add_scalar("DQN/q_topk_mean",      selection_info["q_topk_mean"], round_idx)
        writer.add_scalar("DQN/selected_q_mean",  selection_info["selected_q_mean"], round_idx)
        if dqn_loss is not None:
            writer.add_scalar("DQN/loss",         dqn_loss,          round_idx)

        # 3) Detection metrics
        writer.add_scalar("Defense/TPR",          tpr,               round_idx)
        writer.add_scalar("Defense/FPR",          fpr,               round_idx)
        writer.add_scalar("Defense/detected_mal", detected_mal,      round_idx)
        writer.add_scalar("Defense/missed_mal",   missed_mal,        round_idx)

        # 4) Malicious scores của các clients được chọn
        avg_mal_score_selected = float(np.mean(malicious_scores)) if malicious_scores else 0.0
        writer.add_scalar("Defense/avg_malicious_score", avg_mal_score_selected, round_idx)

        # 5) Client selection breakdown
        writer.add_scalar("Selection/total_selected",    trained_clients,     round_idx)
        writer.add_scalar("Selection/malicious_in_sel",  len(mal_selected),   round_idx)
        writer.add_scalar("Selection/attacker_ratio",    attacker_sel_ratio,  round_idx)
        writer.add_scalar("Selection/random_pick_ratio", random_pick_ratio,   round_idx)
        writer.add_scalar("Selection/benign_in_sel",     len(ben_selected),   round_idx)
        writer.add_scalar("Selection/total_samples",     selected_samples,    round_idx)

        # ── CONSOLE LOGGING ──────────────────────────────────────────
        replay_size = len(dqn.memory)
        round_sec   = time.perf_counter() - round_start

        if avg_round_sec is None:
            avg_round_sec = round_sec
        else:
            avg_round_sec = 0.9 * avg_round_sec + 0.1 * round_sec

        eta_sec = max(0.0, (ROUNDS - round_idx) * avg_round_sec)

        dqn_train_min_buffer = max(DQN_BATCH_SIZE, DQN_WARMUP_STEPS)
        dqn_info = (
            f"DQN Loss: {dqn_loss:.6f} | Epsilon: {dqn.epsilon:.4f} | Replay: {replay_size}"
            if dqn_loss is not None
            else f"DQN Loss: warming up ({replay_size}/{dqn_train_min_buffer}) | Epsilon: {dqn.epsilon:.4f}"
        )
        tqdm.write(dqn_info)
        tqdm.write(
            f"Round {round_idx:02d} | time={round_sec:.1f}s | "
            f"acc={global_acc:.4f} | loss={global_loss:.4f} | rw_sum={episode_total_reward:.4f} | "
            f"TPR={tpr:.2f} | FPR={fpr:.2f} | "
            f"mal_in_sel={len(mal_selected)}/{num_malicious} | "
            f"atk_ratio={attacker_sel_ratio:.2f} | "
            f"rand_pick={selection_info['random_count']}/{trained_clients} | "
            f"q_mean={selection_info['q_mean']:.4f}"
        )
        round_bar.set_postfix(
            acc  = f"{global_acc:.4f}",
            gl   = f"{global_loss:.4f}",
            rw   = f"{episode_total_reward:.4f}",
            dqn  = f"{dqn_loss:.4f}" if dqn_loss is not None else "warmup",
            eps  = f"{dqn.epsilon:.4f}",
            TPR  = f"{tpr:.2f}",
            eta  = f"{eta_sec/60:.1f}m",
        )

        prev_acc      = global_acc
        prev_rewards  = np.asarray(reward, dtype=np.float32)
        current_state = next_state

    # ── FINAL SUMMARY ────────────────────────────────────────────────
    writer.add_scalar("hparam/final_acc", global_acc, 0)
    writer.close()

    print(f"\n=== Training Done ===")
    print(f"Final Accuracy:  {global_acc:.4f}")
    print(f"TensorBoard log: {log_dir}")
    print(f"Xem bằng lệnh:   tensorboard --logdir runs")