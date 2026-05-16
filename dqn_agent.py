import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# =========================
# Q-Network
# =========================

class QNetwork(nn.Module):
    def __init__(self, state_dim):
        super(QNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        bsz, num_clients, feat_dim = x.shape
        y = self.net(x.reshape(bsz * num_clients, feat_dim))
        return y.reshape(bsz, num_clients)


# =========================
# Replay Buffer
# =========================

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_mask, reward, next_state):
        self.buffer.append((state, action_mask, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, action_masks, rewards, next_states = zip(*batch)

        states = np.asarray(states, dtype=np.float32)
        action_masks = np.asarray(action_masks, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        return (
            torch.from_numpy(states),
            torch.from_numpy(action_masks),
            torch.from_numpy(rewards),
            torch.from_numpy(next_states)
        )

    def __len__(self):
        return len(self.buffer)


# =========================
# DQN Agent
# =========================

class DQNAgent:

    def __init__(self,
                 state_dim,
                 num_clients,
                 select_ratio=0.1,
                 device="cuda",
                 gamma=0.9,
                 lr=1e-3,
                 epsilon=1.0,
                 epsilon_decay=0.995,
                 epsilon_min=0.05,
                 update_target_steps=10,
                 replay_capacity=10000,
                 warmup_steps=0):

        self.device = device

        self.num_clients = num_clients
        self.action_dim = num_clients

        self.select_ratio = select_ratio
        self.select_num = max(1, int(num_clients * select_ratio))
        self.state_dim = state_dim

        self.primary_q_net = QNetwork(state_dim).to(device)
        self.target_q_net = QNetwork(state_dim).to(device)
        self.target_q_net.load_state_dict(self.primary_q_net.state_dict())

        # Backward-compatible aliases
        self.q_net = self.primary_q_net
        self.target_net = self.target_q_net

        self.optimizer = optim.Adam(self.primary_q_net.parameters(), lr=lr)

        self.memory = ReplayBuffer(capacity=replay_capacity)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.update_target_steps = update_target_steps
        self.step_count = 0
        self.warmup_steps = int(max(0, warmup_steps))

        self.prev_state = None
        self.prev_action_mask = None

    def _ensure_state_dim(self, state_arr):
        arr = np.asarray(state_arr, dtype=np.float32)

        if arr.ndim == 1:
            if arr.size == self.state_dim:
                arr = np.tile(arr.reshape(1, -1), (self.num_clients, 1))
            elif arr.size == self.num_clients * self.state_dim:
                arr = arr.reshape(self.num_clients, self.state_dim)
            else:
                padded = np.zeros((self.num_clients, self.state_dim), dtype=np.float32)
                flat = arr.reshape(-1)
                take = min(flat.size, self.num_clients * self.state_dim)
                padded.reshape(-1)[:take] = flat[:take]
                arr = padded

        if arr.ndim != 2:
            raise ValueError(f"state must be 2D [num_clients, state_dim], got shape {arr.shape}")

        if arr.shape[0] != self.num_clients:
            fixed_rows = np.zeros((self.num_clients, arr.shape[1]), dtype=np.float32)
            take_rows = min(self.num_clients, arr.shape[0])
            fixed_rows[:take_rows] = arr[:take_rows]
            arr = fixed_rows

        if arr.shape[1] != self.state_dim:
            fixed_cols = np.zeros((self.num_clients, self.state_dim), dtype=np.float32)
            take_cols = min(self.state_dim, arr.shape[1])
            fixed_cols[:, :take_cols] = arr[:, :take_cols]
            arr = fixed_cols

        return arr

    def _action_to_mask(self, action):
        mask = np.zeros(self.num_clients, dtype=np.float32)

        if np.isscalar(action):
            idx = int(action)
            if 0 <= idx < self.num_clients:
                mask[idx] = 1.0
            return mask

        for idx in action:
            idx = int(idx)
            if 0 <= idx < self.num_clients:
                mask[idx] = 1.0

        return mask


# =========================
# Build State S_t
# =========================

    def build_state(self,
                    weights,
                    data_sizes,
                    malicious_scores,
                    global_acc,
                    client_ids=None):

        """
        weights:          list of PCA-reduced weight vectors  w_i  (all num_clients entries)
        data_sizes:       n_i  (raw sample counts)
        malicious_scores: m_i  (raw detector outputs)
        global_acc:       acc_g ∈ [0, 1]

        Normalization applied before concatenation:
          - PCA weights  : z-score  (mean=0, std=1) across all clients that participated
          - data_sizes   : proportion  n_i / Σ n_j  → [0, 1]
          - mal scores   : min-max → [0, 1]
          - global_acc   : already ∈ [0, 1]
        """

        eps = 1e-8

        if client_ids is None:
            client_ids = list(range(len(weights)))

        client_ids = [int(cid) for cid in client_ids]

        if len(weights) > 0:
            weight_dim = len(weights[0])
        else:
            weight_dim = 1

        # ── normalize PCA weights (z-score across participating clients) ──
        if len(weights) > 0:
            w_matrix = np.stack([np.asarray(w, dtype=np.float32) for w in weights])  # (C, D)
            w_mean = w_matrix.mean(axis=0)
            w_std  = w_matrix.std(axis=0)
            w_norm = (w_matrix - w_mean) / (w_std + eps)           # (C, D)
        else:
            w_norm = np.zeros((0, weight_dim), dtype=np.float32)

        # ── normalize data_sizes → proportions ──
        size_arr = np.asarray(data_sizes, dtype=np.float32)
        total_data = float(size_arr.sum())
        size_norm = size_arr / (total_data + eps)                   # (C,)

        # ── normalize malicious scores → [0, 1] min-max ──
        mal_arr = np.abs(np.asarray(malicious_scores, dtype=np.float32))
        mal_min, mal_max = float(mal_arr.min()), float(mal_arr.max())
        if mal_max - mal_min > eps:
            mal_norm = (mal_arr - mal_min) / (mal_max - mal_min + eps)
        else:
            mal_norm = np.zeros_like(mal_arr)                       # (C,)

        # ── build lookup maps (indexed by client_id) ──
        w_norm_map    = {cid: w_norm[i]    for i, cid in enumerate(client_ids)}
        size_norm_map = {cid: size_norm[i] for i, cid in enumerate(client_ids)}
        mal_norm_map  = {cid: mal_norm[i]  for i, cid in enumerate(client_ids)}

        # ── assemble state matrix [N, F] ──
        state_rows = []
        for cid in range(self.num_clients):
            row = []
            if cid not in w_norm_map:
                row.extend([0.0] * weight_dim)
                row.append(0.0)   # size  → 0 proportion
                row.append(0.0)   # score → 0 (benign-neutral)
            else:
                row.extend(w_norm_map[cid].tolist())
                row.append(float(size_norm_map[cid]))
                row.append(float(mal_norm_map[cid]))

            row.append(float(global_acc))
            state_rows.append(row)

        state_arr = np.array(state_rows, dtype=np.float32)
        return self._ensure_state_dim(state_arr)


# =========================
# Action selection (Top-P)
# =========================

    def _compute_q_values(self, state):
        state_arr = self._ensure_state_dim(np.asarray(state, dtype=np.float32))
        state_tensor = torch.from_numpy(state_arr).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.primary_q_net(state_tensor).squeeze(0).cpu().numpy().astype(np.float32)

        return q_values

    def get_q_stats(self, state):
        q_values = self._compute_q_values(state)
        topk_vals = np.sort(q_values)[-self.select_num:]

        return {
            "q_min": float(np.min(q_values)),
            "q_max": float(np.max(q_values)),
            "q_mean": float(np.mean(q_values)),
            "q_std": float(np.std(q_values)),
            "q_topk_mean": float(np.mean(topk_vals)),
        }

    def select_action(self, state, return_info=False):

        q_values = self._compute_q_values(state)
        ranked_clients = [int(i) for i in np.argsort(q_values)[::-1].tolist()]
        epsilon_used = float(self.epsilon)

        available = set(range(self.num_clients))
        selected = []
        random_count = 0
        greedy_count = 0

        # Mixed epsilon-greedy theo từng slot chọn (paper-style),
        # không phải all-or-nothing cho cả tập A_t.
        for _ in range(self.select_num):
            do_random = random.random() < epsilon_used

            if do_random:
                pick = random.choice(tuple(available))
                random_count += 1
            else:
                pick = None
                while ranked_clients:
                    cand = ranked_clients.pop(0)
                    if cand in available:
                        pick = cand
                        greedy_count += 1
                        break

                if pick is None:
                    pick = random.choice(tuple(available))
                    random_count += 1

            selected.append(int(pick))
            available.remove(int(pick))

        selected = np.asarray(selected, dtype=np.int64)

        # Decay epsilon theo mỗi bước ra quyết định (mỗi round FL có 1 action).
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        if not return_info:
            return selected

        q_stats = self.get_q_stats(state)
        selected_q = q_values[selected] if selected.size > 0 else np.array([0.0], dtype=np.float32)
        info = {
            "epsilon": epsilon_used,
            "mode": "mixed_epsilon_greedy",
            "random_count": int(random_count),
            "greedy_count": int(greedy_count),
            "selected_q_mean": float(np.mean(selected_q)),
            **q_stats,
        }

        return selected, info


# =========================
# Reward function
# =========================

    def compute_reward(self,
                       prev_rewards,
                       selected_ids,
                       local_weights,
                       global_weights,
                       global_acc,
                       prev_acc,
                       malicious_scores,
                       alpha=0.2,
                       beta=0.3,
                       lam=0.5):

        # Reward vector R_t với shape [N]
        eps = 1e-8
        prev_arr = np.asarray(prev_rewards, dtype=np.float32).reshape(-1)
        if prev_arr.size == 1:
            prev_arr = np.full((self.num_clients,), float(prev_arr[0]), dtype=np.float32)

        reward_vec = np.zeros((self.num_clients,), dtype=np.float32)
        take = min(self.num_clients, prev_arr.size)
        reward_vec[:take] = prev_arr[:take]

        if len(selected_ids) == 0:
            return reward_vec

        selected_ids = [int(cid) for cid in selected_ids]
        if len(local_weights) != len(selected_ids):
            raise ValueError("local_weights length must match selected_ids length.")

        if isinstance(global_weights, (list, tuple)):
            if len(global_weights) != len(local_weights):
                raise ValueError("global_weights length must match local_weights length.")
            ref_vecs = [np.asarray(gw, dtype=np.float32).reshape(-1) for gw in global_weights]
        else:
            ref = np.asarray(global_weights, dtype=np.float32).reshape(-1)
            ref_vecs = [ref for _ in range(len(local_weights))]

        m = np.abs(np.asarray(malicious_scores, dtype=np.float32)).reshape(-1)
        if m.size == 0:
            m = np.zeros((len(selected_ids),), dtype=np.float32)
        if m.size != len(selected_ids):
            fixed = np.zeros((len(selected_ids),), dtype=np.float32)
            fixed[:min(m.size, fixed.size)] = m[:min(m.size, fixed.size)]
            m = fixed

        for j, cid in enumerate(selected_ids):
            lw = np.asarray(local_weights[j], dtype=np.float32).reshape(-1)
            gw = ref_vecs[j]

            if lw.shape != gw.shape:
                raise ValueError(f"Shape mismatch at sample {j}: local {lw.shape} vs global {gw.shape}.")

            pn = max(1, lw.size)
            diff = (lw - gw) / (np.abs(gw) + eps)
            dist_i = float(np.sum(np.abs(diff)) / pn)

            acc_delta = float(global_acc - prev_acc)
            # exp(-dist): benign (dist nhỏ) → ~1.0, malicious (dist lớn) → ~0
            # acc_delta bonus: chỉ thưởng thêm khi accuracy cải thiện
            utility_i = float(np.exp(-dist_i) * (1.0 + max(0.0, acc_delta)))

            # Use RAW malicious score (no min-max normalization) to avoid
            # collapsing penalty signal when selected scores are close.
            m_raw = float(max(0.0, m[j]))
            penalty_i = float(1.0 - np.exp(-m_raw))
            reward_vec[cid] = float(alpha * reward_vec[cid] + beta * utility_i - lam * penalty_i)

        return reward_vec


# =========================
# Store experience
# =========================

    def remember(self, state, action, reward, next_state):
        state_arr = self._ensure_state_dim(np.asarray(state, dtype=np.float32))
        next_state_arr = self._ensure_state_dim(np.asarray(next_state, dtype=np.float32))
        action_mask = self._action_to_mask(action)
        reward_arr = np.asarray(reward, dtype=np.float32).reshape(-1)

        if reward_arr.size == 1:
            reward_arr = np.full((self.num_clients,), float(reward_arr[0]), dtype=np.float32)
        elif reward_arr.size != self.num_clients:
            fixed = np.zeros((self.num_clients,), dtype=np.float32)
            fixed[:min(self.num_clients, reward_arr.size)] = reward_arr[:min(self.num_clients, reward_arr.size)]
            reward_arr = fixed

        self.memory.push(state_arr, action_mask, reward_arr, next_state_arr)


# =========================
# Stateful transition update
# =========================

    def update_transition(self, curr_state, action, reward, next_state):
        self.remember(curr_state, action, reward, next_state)
        self.prev_state = np.asarray(next_state, dtype=np.float32)
        self.prev_action_mask = self._action_to_mask(action)


# =========================
# Train DQN
# =========================

    def train(self, batch_size=32):

        min_buffer = max(int(batch_size), self.warmup_steps)
        if len(self.memory) < min_buffer:
            return None

        states, action_masks, rewards, next_states = self.memory.sample(batch_size)

        states = states.to(self.device)
        action_masks = action_masks.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)

        # Q_main(s_t) có shape (B, N)
        q_values = self.primary_q_net(states)

        # Bellman target đúng cho top-P selection:
        # target_i = r_i + γ * mean(top-P Q(s'))  — shared bootstrap signal
        # Chỉ tính loss trên clients được chọn (action_mask == 1)
        with torch.no_grad():
            next_q_all = self.target_q_net(next_states)           # (B, N)
            top_p_vals, _ = torch.topk(next_q_all, self.select_num, dim=1)  # (B, P)
            next_q_bootstrap = top_p_vals.mean(dim=1, keepdim=True)         # (B, 1)
            # Broadcast: tất cả selected clients nhận cùng bootstrap value
            target_q = rewards + self.gamma * next_q_bootstrap.expand_as(rewards)  # (B, N)

        # Chỉ tính loss trên clients được chọn (action_mask == 1)
        loss_matrix = nn.MSELoss(reduction="none")(q_values, target_q)   # (B, N)
        masked_loss = (loss_matrix * action_masks).sum() / (action_masks.sum() + 1e-8)
        loss = masked_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.step_count += 1

        if self.step_count % self.update_target_steps == 0:
            self.target_q_net.load_state_dict(self.primary_q_net.state_dict())

        return float(loss.item())