import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy

from malicious_score import (
    flatten_weights,
    compute_malicious_scores,
    update_attacker_history
)


# =========================
# NORMAL CLIENT
# =========================

class Client:

    def __init__(self,
                 client_id,
                 dataset,
                 model,
                 device="cuda",
                 batch_size=32):

        self.client_id = client_id
        self.dataset   = dataset
        self.device    = device

        self.model = deepcopy(model).to(device)

        self.criterion = nn.CrossEntropyLoss()

        # SGD với momentum + weight decay — chuẩn thực nghiệm CIFAR
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=5e-4,
        )

        self.data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            # ⚠️ Windows: num_workers > 0 gây hang/deadlock — luôn dùng 0
            # Linux/Mac: có thể tăng lên 4 để tăng tốc
            num_workers=0,
            pin_memory=False,  # chỉ có lợi khi num_workers > 0
        )

        # tiện cho mô phỏng/đánh giá detection
        self.is_malicious = False


# =========================
# SET GLOBAL MODEL
# =========================

    def set_global_weights(self, global_weights):
        self.model.load_state_dict(global_weights)


# =========================
# TRAIN (BENIGN)
# =========================

    def train(self, epochs=1):

        self.model.train()

        prev_state = deepcopy(self.model.state_dict())

        for _ in range(epochs):
            for images, labels in self.data_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        new_state = self.model.state_dict()

        # trả về delta update
        weight_update = {}
        for key in new_state.keys():
            weight_update[key] = new_state[key] - prev_state[key]

        return weight_update


# =========================
# MALICIOUS CLIENT
# =========================

class MaliciousClient(Client):

    def __init__(self,
                 client_id,
                 dataset,
                 model,
                 attack_type="label_flipping",
                 device="cuda",
                 batch_size=32):

        super().__init__(client_id, dataset, model, device, batch_size=batch_size)
        self.attack_type = attack_type
        self.is_malicious = True


# =========================
# ATTACK LOGIC
# =========================

    def manipulate_data(self, images, labels):

        if self.attack_type == "label_flipping":
            labels = (labels + 1) % 10

        elif self.attack_type == "backdoor":
            images = self.add_trigger(images)
            labels[:] = 0  # target label

        return images, labels


    def add_trigger(self, images):
        images = images.clone()
        images[:, :, -3:, -3:] = 1.0
        return images


# =========================
# OVERRIDE TRAIN
# =========================

    def train(self, epochs=1):

        self.model.train()

        prev_state = deepcopy(self.model.state_dict())

        for _ in range(epochs):
            for images, labels in self.data_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                # APPLY ATTACK
                images, labels = self.manipulate_data(images, labels)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        new_state = self.model.state_dict()

        # trả về delta update
        weight_update = {}
        for key in new_state.keys():
            weight_update[key] = new_state[key] - prev_state[key]

        # Additive noise attack
        if self.attack_type == "noise":
            weight_update = self.add_noise(weight_update)

        return weight_update


# =========================
# NOISE ATTACK
# =========================

    def add_noise(self, weight_update, scale=0.1):

        noisy_update = {}

        for k, v in weight_update.items():
            noise = torch.randn_like(v) * scale
            noisy_update[k] = v + noise

        return noisy_update


# =========================
# CLIENT MANAGER
# =========================

class ClientManager:

    def __init__(self, clients, dqn_agent=None):

        self.clients = clients
        self.dqn_agent = dqn_agent

        # lịch sử attacker cho toàn bộ client theo global client_id
        self.client_history = [0] * len(clients)

        # Memory bank: latest flattened delta update cho TOÀN BỘ clients.
        # - selected round hiện tại: cập nhật vector mới
        # - không selected: giữ vector cũ
        self.all_client_updates = [None] * len(clients)


# =========================
# SELECT CLIENTS — fallback (private)
# =========================

    def _select_clients_fallback(self, state=None):
        """
        Fallback selection khi train_clients() không nhận selected_ids.
        Dùng DQN nếu có, ngược lại random.
        Không bao giờ bị monkey-patch từ bên ngoài (Issue #8).
        """
        if self.dqn_agent is None:
            return np.random.choice(
                len(self.clients),
                size=max(1, int(len(self.clients) * 0.5)),
                replace=False
            ).tolist()

        return self.dqn_agent.select_action(state).tolist()


# =========================
# UTILS — detection summary
# =========================

    @property
    def malicious_ratio(self):
        """Tỷ lệ clients thực sự là attacker (ground truth)."""
        n_mal = sum(1 for c in self.clients if c.is_malicious)
        return n_mal / max(1, len(self.clients))

# =========================
# UTILS
# =========================

    @staticmethod
    def delta_to_absolute(global_weights, delta_weights):
        abs_weights = {}
        for k in global_weights.keys():
            abs_weights[k] = global_weights[k] + delta_weights[k]
        return abs_weights

# =========================
# TRAIN CLIENTS
# =========================

    def train_clients(self,
                      global_weights,
                      round_idx,
                      local_epochs=1,
                      selected_ids=None):
        """
        Parameters
        ----------
        selected_ids : list[int], optional
            Danh sách client IDs đã được chọn bởi DQN ở train.py.
            Nếu None → fallback tự chọn qua _select_clients_fallback().
            Truyền trực tiếp giúp loại bỏ monkey-patch anti-pattern (Issue #8).
        """

        # ── Step 1: Broadcast global model đến TẤT CẢ clients ──
        for client in self.clients:
            client.set_global_weights(global_weights)

        # ── Step 2: Resolve selection ──
        if selected_ids is not None:
            # Caller (train.py / DQN) đã chọn — dùng trực tiếp
            selected_ids = [int(i) for i in selected_ids]
        else:
            # Fallback: tự chọn (standalone / no-DQN mode)
            selected_ids = [int(i) for i in self._select_clients_fallback()]

        selected_clients = [self.clients[i] for i in selected_ids]

        # ── Step 3: Local training ──
        local_updates       = []
        delta_weights_list  = []

        for client in selected_clients:
            delta_w = client.train(epochs=local_epochs)

            local_updates.append({
                "client_id":    client.client_id,
                "client":       client,
                "weights":      delta_w,           # delta w_i_t theo paper
                "data_size":    len(client.dataset),
                "is_malicious": client.is_malicious,
            })
            delta_weights_list.append(flatten_weights(delta_w))

        # ── Step 3.5: Update all-client memory bank ──
        for local_idx, cid in enumerate(selected_ids):
            self.all_client_updates[cid] = delta_weights_list[local_idx]

        # ── Step 4: Malicious scoring ──
        selected_history = [self.client_history[cid] for cid in selected_ids]

        malicious_scores, md_scores, att_ip = compute_malicious_scores(
            delta_weights_list,
            selected_history,
            round_idx,
            all_client_updates=self.all_client_updates,
            selected_ids=selected_ids,
        )

        # Cập nhật lịch sử attacker (chỉ với clients được chọn)
        md_feature_dim = int(delta_weights_list[0].size) if len(delta_weights_list) > 0 else 1
        updated_history = update_attacker_history(
            md_scores,
            selected_history,
            threshold_mode="chi2_95",
            n_features=md_feature_dim,
        )
        for local_idx, cid in enumerate(selected_ids):
            self.client_history[cid] = updated_history[local_idx]

        # Gắn scores vào từng update dict
        for i in range(len(local_updates)):
            local_updates[i]["malicious_score"] = float(malicious_scores[i])
            local_updates[i]["md_score"]         = float(md_scores[i])
            local_updates[i]["attacker_prob"]    = float(att_ip[i])

        return {
            "updates":      local_updates,
            "selected_ids": selected_ids,
        }