import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
from sklearn.decomposition import PCA

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
        self.attack_type  = attack_type
        self.is_malicious = True
        # Mỗi malicious client có target label khác nhau (phân tán hướng tấn công)
        # client_id % 10 đảm bảo 10 class CIFAR-10 đều có thể là target
        self.target_label = client_id % 10


# =========================
# ATTACK LOGIC
# =========================

    def manipulate_data(self, images, labels):

        if self.attack_type == "label_flipping":
            labels = (labels + 1) % 10

        elif self.attack_type == "backdoor":
            # Partial backdoor: chỉ poison 50% batch — 50% còn lại train sạch
            # → giữ được accuracy task chính, update trông "bình thường" hơn
            n = images.size(0)
            n_poison = max(1, n // 2)   # 50%
            idx = torch.randperm(n, device=images.device)[:n_poison]
            images = images.clone()
            images[idx] = self.add_trigger(images[idx])
            labels = labels.clone()
            labels[idx] = self.target_label   # mỗi client có target label riêng

        # --- Combined attacks: poisoning + large weight noise ---
        # Goal: push weight delta further from benign distribution
        # → higher Mahalanobis Distance → malicious score tăng → dễ detect hơn.
        elif self.attack_type == "noise_label_flipping":
            labels = (labels + 1) % 10  # label flip (data poisoning)
            # weight noise is injected in train() after computing the delta

        elif self.attack_type == "noise_backdoor":
            n = images.size(0)
            n_poison = max(1, n // 2)
            idx = torch.randperm(n, device=images.device)[:n_poison]
            images = images.clone()
            images[idx] = self.add_trigger(images[idx])
            labels = labels.clone()
            labels[idx] = self.target_label
            # weight noise is injected in train() after computing the delta

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

        # Additive noise attack (standalone)
        if self.attack_type == "noise":
            weight_update = self.add_noise(weight_update)

        # Combined poisoning + large weight noise:
        # The extra noise pushes the delta further from the benign distribution
        # → higher MD score → malicious score m_i tăng → easier for DQN to detect.
        elif self.attack_type in ("noise_label_flipping", "noise_backdoor"):
            weight_update = self.add_noise(weight_update)  # reuse same scale=0.1

        return weight_update


# =========================
# NOISE ATTACK
# =========================

    def add_noise(self, weight_update, scale=0.1):
        """Absolute additive Gaussian noise (scale=0.1 by default).

        Used by:
        - attack_type=="noise"               : standalone noise attack
        - attack_type=="noise_label_flipping": flip + noise → high MD
        - attack_type=="noise_backdoor"      : backdoor + noise → high MD
        """
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

        # Memory bank: latest PCA-reduced delta update cho TOÀN BỘ clients.
        # Lưu PCA vectors (≤100 dims) thay vì raw flattened (~600k dims).
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
                      selected_ids=None,
                      pca_model=None,
                      pca_output_dim=100):
        """
        Parameters
        ----------
        selected_ids : list[int], optional
            Danh sách client IDs đã được chọn bởi DQN ở train.py.
            Nếu None → fallback tự chọn qua _select_clients_fallback().
            Truyền trực tiếp giúp loại bỏ monkey-patch anti-pattern (Issue #8).
        pca_model : sklearn PCA, optional
            PCA model đã fitted. Dùng để transform raw delta → PCA space
            trước khi lưu vào memory bank và tính MD.
        pca_output_dim : int
            Số chiều output PCA (dùng cho fallback khi chưa có pca_model).
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
        raw_delta_list      = []    # raw flattened deltas (~600k dims)

        for client in selected_clients:
            delta_w = client.train(epochs=local_epochs)

            local_updates.append({
                "client_id":    client.client_id,
                "client":       client,
                "weights":      delta_w,           # delta w_i_t theo paper
                "data_size":    len(client.dataset),
                "is_malicious": client.is_malicious,
            })
            raw_delta_list.append(flatten_weights(delta_w))

        # ── Step 3.5: PCA transform + update memory bank (PCA space) ──
        pca_vectors = self._transform_to_pca(
            raw_delta_list, pca_model, pca_output_dim
        )
        for local_idx, cid in enumerate(selected_ids):
            self.all_client_updates[cid] = pca_vectors[local_idx]

        # ── Step 4: Malicious scoring (trên PCA space) ──
        selected_history = [self.client_history[cid] for cid in selected_ids]

        malicious_scores, md_scores, att_ip = compute_malicious_scores(
            pca_vectors,              # PCA-reduced vectors cho fallback
            selected_history,
            round_idx,
            all_client_updates=self.all_client_updates,
            selected_ids=selected_ids,
        )

        # Cập nhật lịch sử attacker — Dual-threshold:
        #   Malicious client: MD >= 7.8 → history += 1  (ngưỡng thấp hơn, bắt nhiều attacker hơn)
        #   Benign   client : MD >= 8.5 → history += 1  (ngưỡng cao hơn, giảm false alarm)
        #   Còn lại         : history += 0 (giữ nguyên)
        # Các chế độ cũ (comment để tham khảo):
        # pca_dim = int(pca_vectors[0].size) if len(pca_vectors) > 0 else pca_output_dim
        # updated_history = update_attacker_history(md_scores, selected_history,
        #                       threshold_mode="mean",     n_features=pca_dim)
        # updated_history = update_attacker_history(md_scores, selected_history,
        #                       threshold_mode="median",   n_features=pca_dim)
        # updated_history = update_attacker_history(md_scores, selected_history,
        #                       threshold_mode="chi2_95",  n_features=pca_dim)
        M_THR = 7.8   # ngưỡng cho malicious
        B_THR = 8.5   # ngưỡng cho benign
        for local_idx, cid in enumerate(selected_ids):
            md_val = float(md_scores[local_idx])
            thr    = M_THR if self.clients[cid].is_malicious else B_THR
            if md_val >= thr:
                self.client_history[cid] += 1
            # else: += 0, không thay đổi

        # Gắn scores vào từng update dict
        for i in range(len(local_updates)):
            local_updates[i]["malicious_score"] = float(malicious_scores[i])
            local_updates[i]["md_score"]         = float(md_scores[i])
            local_updates[i]["attacker_prob"]    = float(att_ip[i])
            # Giữ raw delta để train.py dùng cho DQN state / reward
            local_updates[i]["raw_delta"]        = raw_delta_list[i]

        return {
            "updates":      local_updates,
            "selected_ids": selected_ids,
        }


    @staticmethod
    def _transform_to_pca(raw_delta_list, pca_model, output_dim):
        """Transform raw deltas → PCA space.

        Nếu có pca_model fitted → dùng transform().
        Nếu chưa có → fit_transform per-batch (fallback cho early rounds).
        """
        if len(raw_delta_list) == 0:
            return []

        x = np.stack(raw_delta_list).astype(np.float32)
        n_samples, n_features = x.shape

        if pca_model is not None:
            # Dùng PCA model đã fitted
            x_pca = pca_model.transform(x).astype(np.float32)
            curr_dim = x_pca.shape[1]
            if curr_dim < output_dim:
                pad = np.zeros((n_samples, output_dim - curr_dim), dtype=np.float32)
                x_pca = np.concatenate([x_pca, pad], axis=1)
            elif curr_dim > output_dim:
                x_pca = x_pca[:, :output_dim]
        else:
            # Fallback: fit_transform per-batch (early rounds)
            max_comp = min(output_dim, n_samples, n_features)
            if max_comp >= 1 and n_samples >= 2:
                pca_tmp = PCA(n_components=max_comp, svd_solver="randomized",
                              random_state=42)
                x_pca = pca_tmp.fit_transform(x).astype(np.float32)
            else:
                x_pca = x[:, :max_comp].astype(np.float32)
            if max_comp < output_dim:
                pad = np.zeros((n_samples, output_dim - max_comp), dtype=np.float32)
                x_pca = np.concatenate([x_pca, pad], axis=1)

        return [x_pca[i] for i in range(n_samples)]