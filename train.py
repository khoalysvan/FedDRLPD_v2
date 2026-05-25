import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import numpy as np
import time
import sys
import datetime
import pickle
from argparse import ArgumentParser
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
# PCA UTILS
# =========================

def reduce_updates_with_pca(delta_updates, output_dim):
    if len(delta_updates) == 0:
        return []
    x = np.stack(delta_updates).astype(np.float32)
    n_samples, n_features = x.shape
    max_components = min(output_dim, n_samples, n_features)
    if max_components >= 1 and n_samples >= 2:
        pca = PCA(n_components=max_components, svd_solver="randomized", random_state=42)
        x_reduced = pca.fit_transform(x).astype(np.float32)
    else:
        x_reduced = x[:, :max_components].astype(np.float32)
    if max_components < output_dim:
        pad = np.zeros((x_reduced.shape[0], output_dim - max_components), dtype=np.float32)
        x_reduced = np.concatenate([x_reduced, pad], axis=1)
    return [x_reduced[i] for i in range(x_reduced.shape[0])]


def transform_updates_with_pca(delta_updates, pca_model, output_dim):
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
# CHECKPOINT SAVE / LOAD
# =========================

def save_checkpoint(path, round_idx, server, dqn, client_manager,
                    prev_acc, prev_rewards, current_state, reward_history_list,
                    all_weights, all_data_sizes, all_malicious_scores, all_full_deltas,
                    pca_fitted, malicious_ids, best_acc, run_name, log_dir):
    """Luu toan bo trang thai de resume."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "round_idx":            round_idx,
        "global_model":         server.global_model.state_dict(),
        "dqn_primary":          dqn.primary_q_net.state_dict(),
        "dqn_target":           dqn.target_q_net.state_dict(),
        "dqn_optimizer":        dqn.optimizer.state_dict(),
        "dqn_epsilon":          dqn.epsilon,
        "dqn_step_count":       dqn.step_count,
        "replay_buffer":        list(dqn.memory.buffer)[-200:],  # chi luu 200 entries gan nhat
        "client_history":       client_manager.client_history,
        "all_client_updates":   client_manager.all_client_updates,
        "prev_acc":             prev_acc,
        "prev_rewards":         prev_rewards,
        "current_state":        current_state,
        "reward_history_list":  reward_history_list,
        "all_weights":          all_weights,
        "all_data_sizes":       all_data_sizes,
        "all_malicious_scores": all_malicious_scores,
        "all_full_deltas":      [],               # khong luu (qua lon ~800MB)
        "pca_fitted":           pca_fitted,        # PCA fitted model (quan trong)
        # all_full_deltas KHONG luu -- qua lon (~800MB), fill lai sau vai rounds
        "malicious_ids":        sorted(malicious_ids),
        "best_acc":             best_acc,
        "run_name":             run_name,
        "log_dir":              log_dir,
        "save_dir":             os.path.dirname(path),  # folder chua checkpoint nay
    }
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)

    # Xoay vòng backup: last_checkpoint.pt → last_checkpoint_prev.pt (round trước)
    # Nếu file hien tai bi corrupt, van con file round truoc de fallback.
    if "last_checkpoint" in os.path.basename(path) and os.path.isfile(path):
        prev_path = path.replace("last_checkpoint", "last_checkpoint_prev")
        try:
            os.replace(path, prev_path)
        except Exception:
            pass

    os.replace(tmp_path, path)   # atomic rename — khong bi corrupt neu Ctrl+C ngat giua chung
    print(f"[CHECKPOINT] Saved to {path} (round {round_idx})")


def load_checkpoint(path, server, dqn, client_manager):
    """Load checkpoint va khoi phuc toan bo trang thai."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    server.global_model.load_state_dict(ckpt["global_model"])
    dqn.primary_q_net.load_state_dict(ckpt["dqn_primary"])
    dqn.target_q_net.load_state_dict(ckpt["dqn_target"])
    dqn.optimizer.load_state_dict(ckpt["dqn_optimizer"])
    dqn.epsilon    = ckpt["dqn_epsilon"]
    dqn.step_count = ckpt["dqn_step_count"]

    # Restore replay buffer (warm-start: chi 200 entries)
    dqn.memory.buffer.clear()
    for item in ckpt.get("replay_buffer", []):
        dqn.memory.buffer.append(item)

    # Restore client manager state (validate sizes)
    n = len(client_manager.clients)
    _hist = ckpt.get("client_history", [])
    _upd  = ckpt.get("all_client_updates", [])
    client_manager.client_history     = _hist if len(_hist) == n else [0] * n
    client_manager.all_client_updates = _upd  if len(_upd) == n  else [None] * n

    print(f"[CHECKPOINT] Loaded from {path} (round {ckpt['round_idx']})")
    return ckpt


# =========================
# ARGUMENT PARSER
# =========================

def get_args():
    parser = ArgumentParser(description="FedDRLPD Training")
    parser.add_argument("--rounds", "-r", type=int, default=0,
                        help="Number of rounds (0 = unlimited, run until Ctrl+C)")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="Path to checkpoint file to resume from")
    parser.add_argument("--save-dir", "-s", type=str, default="trained_models",
                        help="Directory to save checkpoints")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N rounds")
    parser.add_argument("--attack-type", type=str, default="label_flipping",
                        choices=["label_flipping", "backdoor", "noise",
                                 "noise_label_flipping", "noise_backdoor"])
    parser.add_argument("--num-clients", "-n", type=int, default=100)
    parser.add_argument("--malicious-ratio", type=float, default=0.3)
    parser.add_argument("--local-epoch", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mode", type=str, default="feddrlpd",
                        choices=["feddrlpd", "fedavg"],
                        help="feddrlpd: DQN-based client selection (default); "
                             "fedavg: all clients every round, no DQN")
    return parser.parse_args()


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":

    args = get_args()

    # -- CONFIG --------------------------------------------------------------
    NUM_USERS    = args.num_clients
    BATCH_SIZE   = args.batch_size
    LOCAL_EPOCH  = args.local_epoch
    MODE         = args.mode          # "feddrlpd" | "fedavg"
    PCA_COMPONENTS        = 50
    REPLAY_BUFFER_CAPACITY = 300

    NUM_CLIENTS   = NUM_USERS
    ROUNDS        = args.rounds  # 0 = unlimited
    DATASET       = "cifar10"
    IID           = False
    ALPHA         = 0.5
    DQN_BATCH_SIZE      = 32
    DQN_GAMMA           = 0.95
    DQN_EPSILON_START   = 1.0
    DQN_EPSILON_DECAY   = 0.98
    DQN_EPSILON_MIN     = 0.02
    DQN_WARMUP_STEPS    = 50
    DQN_TARGET_UPD_STEP = 10
    PCA_WARMUP_ROUNDS   = 10

    # -- ATTACKER CONFIG -----------------------------------------------------
    MALICIOUS_RATIO = args.malicious_ratio
    ATTACK_TYPE     = args.attack_type

    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    DQN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    SAVE_DIR   = args.save_dir
    SAVE_EVERY = args.save_every
    # Khi chay moi (khong resume), tu dong tao subfolder theo timestamp
    # de tranh ghi de checkpoint cu.
    if not args.checkpoint:
        _rounds_str = "inf" if ROUNDS == 0 else str(ROUNDS)
        _run_tag = (
            f"{MODE}_{DATASET}_N{NUM_CLIENTS}_"
            f"mal{int(MALICIOUS_RATIO*100)}pct_{ATTACK_TYPE}_"
            f"ep{LOCAL_EPOCH}_r{_rounds_str}_"
            f"{datetime.datetime.now().strftime('%m%d_%H%M')}"
        )
        SAVE_DIR = os.path.join(SAVE_DIR, _run_tag)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # -- DATASET -------------------------------------------------------------
    client_datasets = get_datasets(name=DATASET, num_clients=NUM_CLIENTS, iid=IID, alpha=ALPHA)

    # -- SERVER --------------------------------------------------------------
    server = FederatedServer(
        num_clients=NUM_CLIENTS, device=DEVICE,
        dataset_name=DATASET, model=get_model(DATASET),
    )

    # -- CLIENTS -------------------------------------------------------------
    num_malicious = max(0, int(NUM_CLIENTS * MALICIOUS_RATIO))
    malicious_ids = set(
        np.random.choice(NUM_CLIENTS, size=num_malicious, replace=False).tolist()
    )

    clients = []
    for i in range(NUM_CLIENTS):
        if i in malicious_ids:
            clients.append(MaliciousClient(
                client_id=i, dataset=client_datasets[i], model=server.global_model,
                attack_type=ATTACK_TYPE, device=DEVICE, batch_size=BATCH_SIZE,
            ))
        else:
            clients.append(Client(
                client_id=i, dataset=client_datasets[i], model=server.global_model,
                device=DEVICE, batch_size=BATCH_SIZE,
            ))

    # -- DQN -----------------------------------------------------------------
    client_state_dim = PCA_COMPONENTS + 3
    dqn = DQNAgent(
        state_dim=client_state_dim, num_clients=NUM_CLIENTS, select_ratio=0.3,
        device=DQN_DEVICE, gamma=DQN_GAMMA, epsilon=DQN_EPSILON_START,
        replay_capacity=REPLAY_BUFFER_CAPACITY, update_target_steps=DQN_TARGET_UPD_STEP,
        epsilon_decay=DQN_EPSILON_DECAY, epsilon_min=DQN_EPSILON_MIN,
        warmup_steps=DQN_WARMUP_STEPS,
    )

    client_manager = ClientManager(clients, dqn_agent=None)

    # -- TRAIN LOOP INIT -----------------------------------------------------
    start_round   = 1
    prev_acc      = 0.0
    prev_rewards  = np.zeros(NUM_CLIENTS, dtype=np.float32)
    current_state = np.zeros((NUM_CLIENTS, client_state_dim), dtype=np.float32)
    reward_history_list = []
    best_acc      = 0.0

    all_weights          = [np.zeros(PCA_COMPONENTS, dtype=np.float32) for _ in range(NUM_CLIENTS)]
    all_data_sizes       = [0.0] * NUM_CLIENTS
    all_malicious_scores = [0.0] * NUM_CLIENTS
    all_full_deltas      = [None] * NUM_CLIENTS
    pca_fitted           = None

    # -- TENSORBOARD ---------------------------------------------------------
    run_name = (
        f"{MODE}_{DATASET}_N{NUM_CLIENTS}_"
        f"mal{int(MALICIOUS_RATIO*100)}pct_{ATTACK_TYPE}_"
        f"ep{LOCAL_EPOCH}_r{'inf' if ROUNDS == 0 else ROUNDS}_"
        f"{datetime.datetime.now().strftime('%m%d_%H%M')}"
    )
    log_dir = os.path.join("runs", run_name)

    # -- LOAD CHECKPOINT IF PROVIDED -----------------------------------------
    if args.checkpoint and os.path.isfile(args.checkpoint):
        _ckpt_path = args.checkpoint
        _loaded = False
        # Auto-fallback: try specified → _prev → best
        _candidates = [
            _ckpt_path,
            _ckpt_path.replace("last_checkpoint", "last_checkpoint_prev"),
            os.path.join(os.path.dirname(_ckpt_path), "best_checkpoint.pt"),
        ]
        for _attempt in _candidates:
            if not os.path.isfile(_attempt):
                continue
            try:
                ckpt = load_checkpoint(_attempt, server, dqn, client_manager)
                _loaded = True
                break
            except Exception as _e:
                print(f"[WARN] Cannot load {_attempt}: {_e}")
                print(f"[WARN] Trying next fallback...")

        if _loaded:
            start_round         = ckpt.get("round_idx", 0) + 1
            prev_acc            = ckpt.get("prev_acc", 0.0)
            prev_rewards        = np.asarray(ckpt.get("prev_rewards", prev_rewards), dtype=np.float32)
            current_state       = np.asarray(ckpt.get("current_state", current_state), dtype=np.float32)
            reward_history_list = ckpt.get("reward_history_list", [])
            pca_fitted          = ckpt.get("pca_fitted", None)
            best_acc            = ckpt.get("best_acc", 0.0)
            if "malicious_ids" in ckpt:
                malicious_ids   = set(ckpt["malicious_ids"])
            run_name            = ckpt.get("run_name", run_name)
            log_dir             = ckpt.get("log_dir", log_dir)
            # Restore save_dir → checkpoint tiep tuc luu vao dung folder cu
            _ckpt_save_dir = ckpt.get("save_dir") or os.path.dirname(os.path.abspath(_attempt))
            if _ckpt_save_dir and os.path.isdir(_ckpt_save_dir):
                SAVE_DIR = _ckpt_save_dir
                print(f"[RESUME] save_dir restored: {SAVE_DIR}")

            # Restore list fields — validate length matches NUM_CLIENTS
            _ckpt_aw = ckpt.get("all_weights", [])
            _ckpt_ds = ckpt.get("all_data_sizes", [])
            _ckpt_ms = ckpt.get("all_malicious_scores", [])
            _ckpt_fd = ckpt.get("all_full_deltas", [])
            if len(_ckpt_aw) == NUM_CLIENTS:
                all_weights = _ckpt_aw
            else:
                print(f"[WARN] all_weights size mismatch ({len(_ckpt_aw)} vs {NUM_CLIENTS}), using defaults")
            if len(_ckpt_ds) == NUM_CLIENTS:
                all_data_sizes = _ckpt_ds
            else:
                print(f"[WARN] all_data_sizes size mismatch, using defaults")
            if len(_ckpt_ms) == NUM_CLIENTS:
                all_malicious_scores = _ckpt_ms
            else:
                print(f"[WARN] all_malicious_scores size mismatch, using defaults")
            if len(_ckpt_fd) == NUM_CLIENTS:
                all_full_deltas = _ckpt_fd
            else:
                all_full_deltas = [None] * NUM_CLIENTS

            print(f"[RESUME] Starting from round {start_round}, best_acc={best_acc:.4f}")
        else:
            print("[ERROR] All checkpoint fallbacks failed. Starting from scratch.")


    writer = SummaryWriter(log_dir=log_dir)

    # -- PRINT CONFIG --------------------------------------------------------
    rounds_str = "unlimited (Ctrl+C to stop)" if ROUNDS == 0 else str(ROUNDS)
    print(f"\nTensorBoard logs -> {log_dir}")
    print(f"  Xem bang lenh: tensorboard --logdir runs")
    print(f"\n=== Runtime Config ===")
    print(f"MODE: {MODE.upper()} | DEVICE: {DEVICE} | CLIENTS: {NUM_CLIENTS} | ROUNDS: {rounds_str} | EPOCH: {LOCAL_EPOCH}")
    if MODE == "feddrlpd":
        print(f"DQN: eps_decay={DQN_EPSILON_DECAY}, warmup={DQN_WARMUP_STEPS}, batch={DQN_BATCH_SIZE}")
    else:
        print("FedAvg mode: all clients selected every round, DQN/PCA/reward disabled")
    print(f"Checkpoint: save to {SAVE_DIR}/ every {SAVE_EVERY} rounds")
    print(f"\n=== Attacker Setup ===")
    print(f"MALICIOUS_RATIO: {MALICIOUS_RATIO} ({num_malicious}/{NUM_CLIENTS})")
    print(f"ATTACK_TYPE:     {ATTACK_TYPE}")
    print(f"Attacker IDs:    {sorted(malicious_ids)}")

    writer.add_text("config/dataset",       DATASET,                    0)
    writer.add_text("config/mode",          MODE,                       0)
    writer.add_text("config/attack_type",   ATTACK_TYPE,                0)
    writer.add_text("config/malicious_ids", str(sorted(malicious_ids)), 0)
    writer.add_hparams(
        hparam_dict={
            "num_clients": NUM_CLIENTS, "malicious_ratio": MALICIOUS_RATIO,
            "attack_type": ATTACK_TYPE, "local_epoch": LOCAL_EPOCH,
            "rounds": ROUNDS, "batch_size": BATCH_SIZE,
            "pca_components": PCA_COMPONENTS, "iid": IID, "mode": MODE,
        },
        metric_dict={"hparam/final_acc": 0.0},
    )

    # -- MAIN TRAINING LOOP --------------------------------------------------
    global_acc    = prev_acc
    global_loss   = 0.0
    avg_round_sec = None

    if ROUNDS == 0:
        # Unlimited: count up from start_round
        round_iter = iter(range(start_round, 999999))
        round_bar  = tqdm(round_iter, desc="Training Rounds", unit="round",
                          initial=start_round - 1, disable=True,
                          dynamic_ncols=False, position=0, leave=True,
                          file=sys.stderr)
    else:
        round_bar = tqdm(range(start_round, ROUNDS + 1), desc="Training Rounds", unit="round",
                         initial=start_round - 1, total=ROUNDS, disable=True,
                         dynamic_ncols=False, position=0, leave=True,
                         file=sys.stderr)

    try:
        for round_idx in round_bar:
            round_start = time.perf_counter()
            print(f"\n========== ROUND {round_idx} ==========")

            global_weights = server.broadcast_model()
            global_flat    = flatten_weights(global_weights)

            # ================================================================
            # MODE BRANCH — FedAvg: tất cả client, không DQN
            # ================================================================
            if MODE == "fedavg":
                selected_ids = list(range(NUM_CLIENTS))

                updates_pack = client_manager.train_clients(
                    global_weights, round_idx,
                    local_epochs=LOCAL_EPOCH, selected_ids=selected_ids,
                )
                updates      = updates_pack["updates"]
                selected_ids = updates_pack["selected_ids"]

                result      = server.training_round(updates_pack)
                global_acc  = result["accuracy"]
                global_loss = result["loss"]

                # Detection metrics (chỉ để log — fedavg không defense)
                selected_set    = set(selected_ids)
                mal_selected    = [i for i in selected_ids if i in malicious_ids]
                trained_clients = len(selected_ids)
                tpr = 0.0   # không có defense → không phát hiện ai
                fpr = 0.0

                round_sec = time.perf_counter() - round_start
                avg_round_sec = round_sec if avg_round_sec is None else 0.9 * avg_round_sec + 0.1 * round_sec

                # TensorBoard
                writer.add_scalar("FL/global_accuracy", global_acc,  round_idx)
                writer.add_scalar("FL/global_loss",     global_loss, round_idx)
                writer.add_scalar("FL/accuracy_delta",  global_acc - prev_acc, round_idx)
                writer.add_scalar("Selection/malicious_in_sel", len(mal_selected), round_idx)

                print(
                    f"Round {round_idx:03d} | time={round_sec:.1f}s | "
                    f"acc={global_acc:.4f} | loss={global_loss:.4f} | "
                    f"clients={trained_clients} | mal_in_sel={len(mal_selected)}/{num_malicious}"
                )
                round_bar.set_postfix(
                    acc=f"{global_acc:.4f}",
                    loss=f"{global_loss:.4f}",
                )

                prev_acc = global_acc

            # ================================================================
            # MODE BRANCH — FedDRLPD: DQN client selection (logic cũ)
            # ================================================================
            else:  # MODE == "feddrlpd"
                if round_idx == 1:
                    selected_ids = list(range(NUM_CLIENTS))
                    q_stats = dqn.get_q_stats(current_state)
                    selection_info = {
                        "epsilon": float(dqn.epsilon), "mode": "bootstrap_all_clients",
                        "random_count": 0, "greedy_count": NUM_CLIENTS,
                        "selected_q_mean": q_stats["q_mean"], **q_stats,
                    }
                else:
                    selected_ids_np, selection_info = dqn.select_action(current_state, return_info=True)
                    selected_ids = [int(i) for i in selected_ids_np.tolist()]

                updates_pack = client_manager.train_clients(
                    global_weights, round_idx,
                    local_epochs=LOCAL_EPOCH, selected_ids=selected_ids,
                )
                updates          = updates_pack["updates"]
                selected_ids     = updates_pack["selected_ids"]
                trained_clients  = len(selected_ids)
                selected_samples = int(sum(u["data_size"] for u in updates))

                result      = server.training_round(updates_pack)
                global_acc  = result["accuracy"]
                global_loss = result["loss"]
                feedback    = result["dqn_feedback"]

                # Detection metrics
                selected_set       = set(selected_ids)
                mal_selected       = [i for i in selected_ids if i in malicious_ids]
                ben_selected       = [i for i in selected_ids if i not in malicious_ids]
                attacker_sel_ratio = len(mal_selected) / max(1, trained_clients)
                random_pick_ratio  = selection_info["random_count"] / max(1, trained_clients)
                detected_mal       = len([i for i in malicious_ids if i not in selected_set])
                missed_mal         = len(mal_selected)
                false_excl         = len([i for i in range(NUM_CLIENTS)
                                          if i not in malicious_ids and i not in selected_set])
                tpr = detected_mal / max(1, num_malicious)
                fpr = false_excl   / max(1, NUM_CLIENTS - num_malicious)

                # Build next state
                full_delta_list  = []
                data_sizes       = []
                malicious_scores = []
                client_ids       = []

                for u in updates:
                    full_delta_list.append(flatten_weights(u["weights"]).astype(np.float32))
                    data_sizes.append(float(u["data_size"]))
                    malicious_scores.append(float(u["malicious_score"]))
                    client_ids.append(int(u["client_id"]))

                # Debug MD scores moi 10 rounds
                if round_idx % 10 == 0:
                    _md_ben, _md_mal = [], []
                    for i, cid in enumerate(client_ids):
                        md_raw = float(malicious_scores[i])  # = Att_p * MD, chia de lay MD
                        att_p  = 1.0 + client_manager.client_history[cid] / max(1, round_idx)
                        md_val = md_raw / max(att_p, 1e-8)
                        if cid in malicious_ids:
                            _md_mal.append(md_val)
                        else:
                            _md_ben.append(md_val)
                    _str  = f"[MD Debug] round={round_idx}"
                    _str += f" | benign  MD: mean={np.mean(_md_ben):.2f} max={np.max(_md_ben):.2f}" if _md_ben else " | benign  MD: N/A"
                    _str += f" | malicious MD: mean={np.mean(_md_mal):.2f} max={np.max(_md_mal):.2f}" if _md_mal else " | malicious MD: N/A"
                    print(_str)

                for i, cid in enumerate(client_ids):
                    all_full_deltas[cid] = full_delta_list[i]

                if pca_fitted is None and round_idx >= PCA_WARMUP_ROUNDS:
                    bank = [v for v in all_full_deltas if v is not None]
                    if len(bank) >= PCA_COMPONENTS:
                        x_bank = np.stack(bank).astype(np.float32)
                        n_comp = min(PCA_COMPONENTS, x_bank.shape[0], x_bank.shape[1])
                        if n_comp >= 1:
                            pca_fitted = PCA(n_components=n_comp, svd_solver="randomized", random_state=42)
                            pca_fitted.fit(x_bank)
                            print(f"[PCA] Fitted at round {round_idx} ({x_bank.shape[0]} samples, {n_comp} components)")

                if pca_fitted is not None:
                    weights_list = transform_updates_with_pca(full_delta_list, pca_fitted, PCA_COMPONENTS)
                else:
                    weights_list = reduce_updates_with_pca(full_delta_list, PCA_COMPONENTS)

                for i, cid in enumerate(client_ids):
                    all_weights[cid]          = weights_list[i]
                    all_data_sizes[cid]       = data_sizes[i]
                    all_malicious_scores[cid] = malicious_scores[i]

                next_state = dqn.build_state(
                    all_weights, all_data_sizes, all_malicious_scores, global_acc,
                    client_ids=list(range(NUM_CLIENTS)),
                )

                # Reward
                local_abs_list = [global_flat + delta for delta in full_delta_list]
                global_refs    = [global_flat for _ in range(len(full_delta_list))]
                reward = dqn.compute_reward(
                    prev_rewards, selected_ids, local_abs_list, global_refs,
                    feedback["round_accuracy"], feedback["prev_accuracy"], feedback["malicious_scores"],
                )

                episode_total_reward = float(np.sum(np.asarray(reward, dtype=np.float32)[selected_ids]))
                reward_history_list.append(episode_total_reward)

                # Reward log
                _rw_arr  = np.asarray(reward, dtype=np.float32)
                # Q-values: read-only call — không ảnh hưởng logic chọn client
                _q_vals  = dqn._compute_q_values(current_state)   # shape [NUM_CLIENTS]
                _per_client_lines = []
                _ben_rewards = []
                _mal_rewards = []
                for cid in range(NUM_CLIENTS):
                    rw_i = float(_rw_arr[cid])
                    q_i  = float(_q_vals[cid])
                    tag  = "[M]" if clients[cid].is_malicious else "[B]"
                    sel  = "*" if cid in selected_set else " "
                    m_i  = float(all_malicious_scores[cid])
                    _per_client_lines.append(
                        f"  C{cid:02d} {tag}{sel} rw={rw_i:+.4f} m={m_i:.3f} q={q_i:+.4f}"
                    )
                    if clients[cid].is_malicious:
                        _mal_rewards.append(rw_i)
                    else:
                        _ben_rewards.append(rw_i)
                _mean_ben = float(np.mean(_ben_rewards)) if _ben_rewards else 0.0
                _mean_mal = float(np.mean(_mal_rewards)) if _mal_rewards else 0.0
                _gap      = _mean_ben - _mean_mal
                print("--- Per-Client Reward ---")
                for line in _per_client_lines:
                    print(line)
                print(
                    f"--- AVG  rw_benign={_mean_ben:+.4f}  rw_malicious={_mean_mal:+.4f}  "
                    f"gap(B-M)={_gap:+.4f} {'OK' if _gap > 0 else 'BAD'} ---"
                )

                # DQN update
                dqn.update_transition(curr_state=current_state, action=selected_ids,
                                      reward=reward, next_state=next_state)
                dqn_loss = dqn.train(batch_size=DQN_BATCH_SIZE)

                # TensorBoard
                writer.add_scalar("FL/global_accuracy",        global_acc,           round_idx)
                writer.add_scalar("FL/global_loss",            global_loss,          round_idx)
                writer.add_scalar("FL/accuracy_delta",         global_acc - prev_acc, round_idx)
                writer.add_scalar("DQN/episode_total_reward",  episode_total_reward, round_idx)
                writer.add_scalar("DQN/epsilon",               dqn.epsilon,          round_idx)
                writer.add_scalar("DQN/replay_size",           len(dqn.memory),      round_idx)
                writer.add_scalar("DQN/q_mean",                selection_info["q_mean"], round_idx)
                if dqn_loss is not None:
                    writer.add_scalar("DQN/loss", dqn_loss, round_idx)
                writer.add_scalar("Defense/TPR",               tpr,                  round_idx)
                writer.add_scalar("Defense/FPR",               fpr,                  round_idx)
                writer.add_scalar("Selection/malicious_in_sel", len(mal_selected),   round_idx)
                writer.add_scalar("Selection/attacker_ratio",  attacker_sel_ratio,   round_idx)

                # Console
                replay_size = len(dqn.memory)
                round_sec   = time.perf_counter() - round_start
                avg_round_sec = round_sec if avg_round_sec is None else 0.9 * avg_round_sec + 0.1 * round_sec
                eta_rounds = (ROUNDS - round_idx) if ROUNDS > 0 else 0
                eta_sec = max(0.0, eta_rounds * avg_round_sec)

                if dqn_loss is not None:
                    dqn_info = f"DQN Loss: {dqn_loss:.6f} | Epsilon: {dqn.epsilon:.4f} | Replay: {replay_size}"
                else:
                    dqn_info = f"DQN Loss: warming up ({replay_size}/{max(DQN_BATCH_SIZE, DQN_WARMUP_STEPS)}) | Epsilon: {dqn.epsilon:.4f}"

                print(dqn_info)
                print(
                    f"Round {round_idx:03d} | time={round_sec:.1f}s | "
                    f"acc={global_acc:.4f} | loss={global_loss:.4f} | rw_sum={episode_total_reward:.4f} | "
                    f"rw_ben={_mean_ben:+.4f} | rw_mal={_mean_mal:+.4f} | "
                    f"TPR={tpr:.2f} | FPR={fpr:.2f} | "
                    f"mal_in_sel={len(mal_selected)}/{num_malicious}"
                )
                round_bar.set_postfix(
                    acc=f"{global_acc:.4f}", rw=f"{episode_total_reward:.4f}",
                    dqn=f"{dqn_loss:.4f}" if dqn_loss is not None else "warmup",
                    eps=f"{dqn.epsilon:.4f}", TPR=f"{tpr:.2f}",
                )

                prev_acc      = global_acc
                prev_rewards  = np.asarray(reward, dtype=np.float32)
                current_state = next_state

            # -- CHECKPOINT SAVE --
            # Save best
            if global_acc > best_acc:
                best_acc = global_acc
                save_checkpoint(
                    os.path.join(SAVE_DIR, "best_checkpoint.pt"),
                    round_idx, server, dqn, client_manager,
                    prev_acc, prev_rewards, current_state, reward_history_list,
                    all_weights, all_data_sizes, all_malicious_scores, all_full_deltas,
                    pca_fitted, malicious_ids, best_acc, run_name, log_dir,
                )

            # Save periodic
            if round_idx % SAVE_EVERY == 0:
                save_checkpoint(
                    os.path.join(SAVE_DIR, "last_checkpoint.pt"),
                    round_idx, server, dqn, client_manager,
                    prev_acc, prev_rewards, current_state, reward_history_list,
                    all_weights, all_data_sizes, all_malicious_scores, all_full_deltas,
                    pca_fitted, malicious_ids, best_acc, run_name, log_dir,
                )

    except KeyboardInterrupt:
        print(f"\n[INFO] Ctrl+C detected at round {round_idx} -- saving checkpoint before exit...")

    finally:
        # Luon save khi ket thuc (du hoan thanh hay Ctrl+C)
        save_checkpoint(
            os.path.join(SAVE_DIR, "last_checkpoint.pt"),
            round_idx, server, dqn, client_manager,
            prev_acc, prev_rewards, current_state, reward_history_list,
            all_weights, all_data_sizes, all_malicious_scores, all_full_deltas,
            pca_fitted, malicious_ids, best_acc, run_name, log_dir,
        )

        try:
            writer.add_scalar("hparam/final_acc", global_acc, 0)
        except Exception:
            pass
        writer.close()

        print(f"\n=== Training Finished ===")
        print(f"Last round:     {round_idx}")
        print(f"Final Accuracy: {global_acc:.4f}")
        print(f"Best Accuracy:  {best_acc:.4f}")
        print(f"Checkpoints:    {SAVE_DIR}/")
        print(f"  last_checkpoint.pt  -- resume voi: python train.py -c {SAVE_DIR}/last_checkpoint.pt")
        print(f"  best_checkpoint.pt  -- model co accuracy cao nhat")
        print(f"TensorBoard:    tensorboard --logdir runs")