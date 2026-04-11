import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms


# =========================
# Per-dataset transform configs
# =========================
#
# TRAIN:  augmentation  → random crop + flip + normalize
# EVAL:  no augmentation → only ToTensor + normalize
#
# Mean/Std từ dataset gốc (full training set statistics).
# Dùng nhất quán ở cả client (train) và server (evaluate).

_TRANSFORMS = {
    "cifar10": {
        "train": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std =(0.2470, 0.2435, 0.2616),
            ),
        ]),
        "eval": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std =(0.2470, 0.2435, 0.2616),
            ),
        ]),
    },

    "cifar100": {
        "train": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4865, 0.4409),
                std =(0.2673, 0.2564, 0.2762),
            ),
        ]),
        "eval": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4865, 0.4409),
                std =(0.2673, 0.2564, 0.2762),
            ),
        ]),
    },

    "fashionmnist": {
        "train": transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.2860,),
                std =(0.3530,),
            ),
        ]),
        "eval": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.2860,),
                std =(0.3530,),
            ),
        ]),
    },
}

_DS_MAP = {
    "cifar10":      datasets.CIFAR10,
    "cifar100":     datasets.CIFAR100,
    "fashionmnist": datasets.FashionMNIST,
}


# =========================
# Public API for transforms
# =========================

def get_train_transform(name: str) -> transforms.Compose:
    """Trả về transform có augmentation dùng cho training."""
    name = name.lower()
    if name not in _TRANSFORMS:
        raise ValueError(f"Unsupported dataset: {name}. Supported: {list(_TRANSFORMS)}")
    return _TRANSFORMS[name]["train"]


def get_eval_transform(name: str) -> transforms.Compose:
    """Trả về transform KHÔNG có augmentation dùng cho evaluation/test."""
    name = name.lower()
    if name not in _TRANSFORMS:
        raise ValueError(f"Unsupported dataset: {name}. Supported: {list(_TRANSFORMS)}")
    return _TRANSFORMS[name]["eval"]


# =========================
# Load dataset (train split)
# =========================

def load_dataset(name: str = "cifar10"):
    """
    Tải training dataset với augmentation đúng chuẩn.

    Augmentation pipeline (theo paper & best practice cho CIFAR):
      CIFAR-10/100 : RandomCrop(32, pad=4) + HorizontalFlip + ColorJitter + Normalize
      FashionMNIST : RandomCrop(28, pad=4) + HorizontalFlip + Normalize

    Note: transform chỉ áp dụng khi DataLoader đọc ảnh, không ảnh hưởng đến
    chỉ số mẫu (indices) dùng cho IID/Non-IID split.
    """
    name = name.lower()

    if name not in _DS_MAP:
        raise ValueError(f"Dataset '{name}' not supported. Supported: {list(_DS_MAP)}")

    train_tf = get_train_transform(name)

    return _DS_MAP[name](
        root="./data",
        train=True,
        download=True,
        transform=train_tf,
    )


# =========================
# IID split
# =========================

def split_iid(dataset, num_clients: int):
    """Chia đều ngẫu nhiên dataset cho num_clients clients."""

    num_items  = len(dataset) // num_clients
    all_indices = np.random.permutation(len(dataset))

    client_datasets = []

    for i in range(num_clients):
        start   = i * num_items
        end     = start + num_items
        indices = all_indices[start:end]
        client_datasets.append(Subset(dataset, indices))

    return client_datasets


# =========================
# Non-IID (Dirichlet)
# =========================

def split_noniid(dataset, num_clients: int, alpha: float = 0.5):
    """
    Chia Non-IID theo Dirichlet distribution Dir(alpha).

    alpha nhỏ → phân phối lệch (heterogeneous)
    alpha lớn → gần IID
    """

    labels      = np.array(dataset.targets)
    num_classes = len(np.unique(labels))

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):

        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]

        split = np.split(class_indices, proportions)

        for i in range(num_clients):
            client_indices[i].extend(split[i].tolist())

    return [Subset(dataset, client_indices[i]) for i in range(num_clients)]


# =========================
# Wrapper function (main entry)
# =========================

def get_datasets(name: str    = "cifar10",
                 num_clients: int   = 100,
                 iid: bool          = True,
                 alpha: float       = 0.5):
    """
    Entry point chính.

    Returns
    -------
    list[Subset]  — một Subset per client, dùng training transform (có augmentation).
    """

    dataset = load_dataset(name)

    if iid:
        return split_iid(dataset, num_clients)
    else:
        return split_noniid(dataset, num_clients, alpha)