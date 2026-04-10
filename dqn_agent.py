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
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Replay Buffer
# =========================

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_mask, reward, next_state, done=False):
        self.buffer.append((state, action_mask, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, action_masks, rewards, next_states, dones = zip(*batch)

        states = np.asarray(states, dtype=np.float32)
        action_masks = np.asarray(action_masks, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        return (
            torch.from_numpy(states),
            torch.from_numpy(action_masks),
            torch.from_numpy(rewards),
            torch.from_numpy(next_states),
            torch.from_numpy(dones)
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
                 replay_capacity=10000):

        self.device = device

        self.num_clients = num_clients
        self.action_dim = num_clients

        self.select_ratio = select_ratio
        self.select_num = max(1, int(num_clients * select_ratio))
        self.state_dim = state_dim

        self.primary_q_net = QNetwork(state_dim, self.action_dim).to(device)
        self.target_q_net = QNetwork(state_dim, self.action_dim).to(device)
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

        self.prev_state = None
        self.prev_action_mask = None

    def _ensure_state_dim(self, state_arr):
        if state_arr.shape[0] == self.state_dim:
            return state_arr

        if state_arr.shape[0] > self.state_dim:
            return state_arr[:self.state_dim]

        padded = np.zeros(self.state_dim, dtype=np.float32)
        padded[:state_arr.shape[0]] = state_arr
        return padded

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
        state = []

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

        # ── assemble state vector ──
        for cid in range(self.num_clients):
            if cid not in w_norm_map:
                state.extend([0.0] * weight_dim)
                state.append(0.0)   # size  → 0 proportion
                state.append(0.0)   # score → 0 (benign-neutral)
            else:
                state.extend(w_norm_map[cid].tolist())
                state.append(float(size_norm_map[cid]))
                state.append(float(mal_norm_map[cid]))

        state.append(float(global_acc))   # already ∈ [0, 1]

        state_arr = np.array(state, dtype=np.float32)
        return self._ensure_state_dim(state_arr)


# =========================
# Action selection (Top-P)
# =========================

    def select_action(self, state):

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if random.random() < self.epsilon:
            selected = np.random.choice(
                self.num_clients,
                self.select_num,
                replace=False
            )
            return selected

        with torch.no_grad():
            q_values = self.primary_q_net(state).cpu().numpy().flatten()

        # chọn top-P Q values
        selected = np.argsort(q_values)[-self.select_num:]

        return selected


# =========================
# Reward function
# =========================

    def compute_reward(self,
                       prev_reward,
                       local_weights,
                       global_weights,
                       global_acc,
                       prev_acc,
                       malicious_scores,
                       alpha=0.2,
                       beta=0.5,
                       lam=0.3):

        # ===== Utility =====
        eps = 1e-8
        if len(local_weights) == 0 or len(global_weights) == 0:
            distance = 0.0
        else:
            distance = 0.0
            pn = max(1, len(local_weights[0]))
            for lw, gw in zip(local_weights, global_weights):
                lw = np.asarray(lw, dtype=np.float32)
                gw = np.asarray(gw, dtype=np.float32)
                diff = (lw - gw) / (gw + eps)
                distance += float(np.sum(np.abs(diff)) / pn)

            distance /= max(1, len(local_weights))

        if global_acc > prev_acc:
            utility = float(global_acc + np.exp(-distance))
        else:
            utility = float(1.0 - np.exp(-distance))

        if len(malicious_scores) == 0:
            penalty = 0.0
        else:
            m = np.abs(np.asarray(malicious_scores, dtype=np.float32))
            m_min = float(np.min(m))
            m_max = float(np.max(m))

            # Paper score is bounded; normalize raw detector outputs to [0, 1].
            if m_max - m_min > eps:
                m = (m - m_min) / (m_max - m_min + eps)
            else:
                m = np.zeros_like(m)

            penalty = float(np.mean(1.0 - np.exp(-m)))

        reward = alpha * prev_reward + beta * utility - lam * penalty

        return float(reward)


# =========================
# Store experience
# =========================

    def remember(self, state, action, reward, next_state, done=False):
        state_arr = self._ensure_state_dim(np.asarray(state, dtype=np.float32))
        next_state_arr = self._ensure_state_dim(np.asarray(next_state, dtype=np.float32))
        action_mask = self._action_to_mask(action)
        self.memory.push(state_arr, action_mask, float(reward), next_state_arr, float(done))


# =========================
# Stateful transition update
# =========================

    def update_transition(self, curr_state, action, reward, next_state, done=False):
        self.remember(curr_state, action, reward, next_state, done=done)
        self.prev_state = np.asarray(next_state, dtype=np.float32)
        self.prev_action_mask = self._action_to_mask(action)


# =========================
# Train DQN
# =========================

    def train(self, batch_size=32):

        if len(self.memory) < batch_size:
            return None

        states, action_masks, rewards, next_states, dones = self.memory.sample(batch_size)

        states = states.to(self.device)
        action_masks = action_masks.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Q_primary(s, *)
        q_values = self.primary_q_net(states)

        # Q(s, A_t): trung bình Q-value trên top-P client đã chọn
        selected_count = torch.clamp(action_masks.sum(dim=1), min=1.0)
        q_sa = (q_values * action_masks).sum(dim=1) / selected_count

        # target = r + gamma * max_A' Q_target(s', A')
        # xấp xỉ bằng trung bình top-P q-values ở s'
        with torch.no_grad():
            next_q_all = self.target_q_net(next_states)
            topk_next_q = torch.topk(next_q_all, k=self.select_num, dim=1).values
            next_q = topk_next_q.mean(dim=1)

        target = rewards + (1.0 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # epsilon decay
        self.epsilon = max(self.epsilon * self.epsilon_decay,
                           self.epsilon_min)

        # update target network
        self.step_count += 1

        if self.step_count % self.update_target_steps == 0:
            self.target_q_net.load_state_dict(self.primary_q_net.state_dict())

        return float(loss.item())