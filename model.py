import torch
import torch.nn as nn


# =========================
# CNN Model (paper)
# =========================

class CNNModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(CNNModel, self).__init__()

        # ===== Feature extractor =====
        self.conv = nn.Sequential(

            # Conv 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # ===== Classifier =====
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# =========================
# Model factory
# =========================

def get_model(dataset_name="cifar10"):

    if dataset_name == "cifar10":
        return CNNModel(in_channels=3, num_classes=10)

    elif dataset_name == "cifar100":
        return CNNModel(in_channels=3, num_classes=100)

    elif dataset_name == "fashionmnist":
        return CNNModel(in_channels=1, num_classes=10)

    else:
        raise ValueError("Dataset not supported")


# =========================
# Flatten model weights
# =========================

def flatten_model_weights(model):

    return torch.cat([
        param.data.view(-1)
        for param in model.parameters()
    ])


# =========================
# Get weight update (w_i)
# =========================

def get_weight_update(new_model, old_model):

    update = []

    for new_p, old_p in zip(new_model.parameters(), old_model.parameters()):
        update.append((new_p.data - old_p.data).view(-1))

    return torch.cat(update)


# =========================
# Copy model
# =========================

def copy_model(model):

    new_model = type(model)()
    new_model.load_state_dict(model.state_dict())
    return new_model