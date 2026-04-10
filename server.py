import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import get_model


# =========================
# Federated Server
# =========================

# =========================
# Normalize statistics
# =========================

_NORMALIZE = {
    "cifar10":      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2616)),
    "cifar100":     transforms.Normalize((0.5071, 0.4865, 0.4409),
                                         (0.2673, 0.2564, 0.2762)),
    "fashionmnist": transforms.Normalize((0.2860,), (0.3530,)),
}


# =========================
# Federated Server
# =========================

class FederatedServer:

    def __init__(self,
                 num_clients=100,
                 device="cuda",
                 dataset_name="cifar10",
                 model=None):
        """
        Parameters
        ----------
        model : nn.Module, optional
            Global model instance.  If None, one is created via get_model().
            Passing the model from train.py avoids the double-allocation
            anti-pattern (Issue #7 / Issue #6).
        """

        self.num_clients = num_clients
        self.device      = device
        self.dataset_name = dataset_name.lower()

        # ── Global model (Issue #7): use caller-supplied model if provided ──
        if model is not None:
            self.global_model = model.to(device)
        else:
            self.global_model = get_model(self.dataset_name).to(device)

        self.criterion = nn.CrossEntropyLoss()

        # ── Test DataLoader with proper normalization ──
        norm = _NORMALIZE.get(self.dataset_name)
        test_tf = transforms.Compose(
            [transforms.ToTensor(), norm] if norm else [transforms.ToTensor()]
        )

        _DS_MAP = {
            "cifar10":      datasets.CIFAR10,
            "cifar100":     datasets.CIFAR100,
            "fashionmnist": datasets.FashionMNIST,
        }
        if self.dataset_name not in _DS_MAP:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")

        test_dataset = _DS_MAP[self.dataset_name](
            root="./data",
            train=False,
            download=True,
            transform=test_tf,
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=2,
            pin_memory=(device == "cuda"),
        )

        self.global_accuracy      = 0.0
        self.prev_global_accuracy = 0.0
        self.last_round_info      = None


# =========================
# Broadcast model to clients
# =========================

    def broadcast_model(self):

        return self.global_model.state_dict()


# =========================
# Parse updates from clients
# =========================

    def receive_local_updates(self, updates_pack):

        """
        Hỗ trợ 2 kiểu input:
        1) list[dict] trực tiếp (legacy)
        2) dict có key "updates" và metadata khác (new)
        """

        if isinstance(updates_pack, dict):
            client_updates = updates_pack.get("updates", [])
            selected_ids = updates_pack.get("selected_ids", [])
        else:
            client_updates = updates_pack
            selected_ids = [u.get("client_id", i) for i, u in enumerate(client_updates)]

        if client_updates is None:
            client_updates = []

        local_weights = []
        local_sizes = []
        malicious_scores = []
        client_ids = []

        for i, update in enumerate(client_updates):
            if "weights" not in update:
                raise KeyError("Each client update must contain 'weights'.")

            data_size = int(update.get("data_size", 0))
            if data_size < 0:
                raise ValueError("'data_size' must be non-negative.")

            local_weights.append(update["weights"])
            local_sizes.append(data_size)
            malicious_scores.append(float(update.get("malicious_score", 0.0)))
            client_ids.append(int(update.get("client_id", selected_ids[i] if i < len(selected_ids) else i)))

        return {
            "client_updates": client_updates,
            "local_weights": local_weights,
            "local_sizes": local_sizes,
            "malicious_scores": malicious_scores,
            "client_ids": client_ids,
            "selected_ids": [int(i) for i in selected_ids],
        }


# =========================
# FedAvg Aggregation
# =========================

    def aggregate(self, delta_weights_list, local_data_sizes):
        """
        Aggregation theo công thức bài báo (incremental update):
        θ_t^g = θ_{t-1}^g + Σ (D_i / Σ D_j) * w_i_t

        delta_weights_list: list of delta weight dicts (w_i_t)
        local_data_sizes: list of D_i
        """

        if len(delta_weights_list) == 0:
            raise ValueError("No delta weights provided for aggregation.")

        total_data = sum(local_data_sizes)
        if total_data <= 0:
            raise ValueError("Sum of client data sizes must be > 0 for aggregation.")

        # θ_{t-1}^g
        prev_global = self.global_model.state_dict()

        new_weights = {}

        for key in delta_weights_list[0].keys():

            # Σ (D_i / Σ D_j) * w_i_t
            weighted_delta = None
            for i in range(len(delta_weights_list)):
                coeff = local_data_sizes[i] / total_data
                term = delta_weights_list[i][key] * coeff
                weighted_delta = term if weighted_delta is None else (weighted_delta + term)

            # θ_t^g = θ_{t-1}^g + weighted_delta
            new_weights[key] = prev_global[key] + weighted_delta

        self.global_model.load_state_dict(new_weights)

        return new_weights


# =========================
# Evaluate Global Model
# =========================

    def evaluate(self):

        self.global_model.eval()

        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():

            for images, labels in self.test_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.global_model(images)
                loss = self.criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_loss += loss.item() * labels.size(0)

        accuracy = correct / total
        avg_loss = total_loss / total

        self.global_accuracy = accuracy

        return {
            "accuracy": accuracy,
            "loss": avg_loss
        }


# =========================
# Build DQN feedback
# =========================

    def build_dqn_feedback(self,
                           parsed_updates,
                           eval_result,
                           aggregated_weights):

        acc = float(eval_result["accuracy"])
        prev_acc = float(self.prev_global_accuracy)

        feedback = {
            "round_accuracy": acc,
            "prev_accuracy": prev_acc,
            "accuracy_gain": acc - prev_acc,
            "round_loss": float(eval_result["loss"]),
            "client_ids": parsed_updates["client_ids"],
            "selected_ids": parsed_updates["selected_ids"],
            "data_sizes": parsed_updates["local_sizes"],
            "malicious_scores": parsed_updates["malicious_scores"],
            "aggregated_weights": aggregated_weights,
        }

        return feedback


# =========================
# One FL Round
# =========================

    def training_round(self, client_updates):
        """
        Runs one FL round on the server side:
          1. Parse client updates
          2. Aggregate (incremental update — paper formula)
          3. Evaluate global model
          4. Build DQN feedback
        """

        parsed = self.receive_local_updates(client_updates)

        # FedAvg
        new_global_weights = self.aggregate(
            parsed["local_weights"],
            parsed["local_sizes"]
        )

        # evaluate global model
        eval_result = self.evaluate()

        dqn_feedback = self.build_dqn_feedback(
            parsed_updates=parsed,
            eval_result=eval_result,
            aggregated_weights=new_global_weights
        )

        self.last_round_info = dqn_feedback
        self.prev_global_accuracy = eval_result["accuracy"]

        return {
            "global_weights": self.global_model.state_dict(),
            "accuracy": eval_result["accuracy"],
            "loss": eval_result["loss"],
            "selected_ids": parsed["selected_ids"],
            "dqn_feedback": dqn_feedback
        }


# (training loop is driven externally by train.py)