import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


# =========================
# Load dataset
# =========================

def load_dataset(name="cifar10"):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )

    elif name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )

    elif name == "fashionmnist":
        train_dataset = datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )

    else:
        raise ValueError("Dataset not supported")

    return train_dataset


# =========================
# IID split
# =========================

def split_iid(dataset, num_clients):

    num_items = len(dataset) // num_clients
    all_indices = np.random.permutation(len(dataset))

    client_datasets = []

    for i in range(num_clients):
        start = i * num_items
        end = start + num_items

        indices = all_indices[start:end]

        client_datasets.append(Subset(dataset, indices))

    return client_datasets


# =========================
# Non-IID (Dirichlet)
# =========================

def split_noniid(dataset, num_clients, alpha=0.5):

    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):

        class_indices = np.where(labels == c)[0]

        np.random.shuffle(class_indices)

        # Dirichlet phân phối
        proportions = np.random.dirichlet(
            np.repeat(alpha, num_clients)
        )

        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]

        split = np.split(class_indices, proportions)

        for i in range(num_clients):
            client_indices[i].extend(split[i])

    client_datasets = [
        Subset(dataset, client_indices[i])
        for i in range(num_clients)
    ]

    return client_datasets


# =========================
# Wrapper function
# =========================

def get_datasets(name="cifar10",
                 num_clients=100,
                 iid=True,
                 alpha=0.5):

    dataset = load_dataset(name)

    if iid:
        client_datasets = split_iid(dataset, num_clients)
    else:
        client_datasets = split_noniid(dataset, num_clients, alpha)

    return client_datasets