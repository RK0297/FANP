"""
CIFAR-10 / CIFAR-100 data loaders with standard augmentation.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_loaders(
    data_dir: str = "./data/downloads",
    batch_size: int = 128,
    num_workers: int = 4,
    val_split: float = 0.1,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader) for CIFAR-10.

    - Train: random crop + horizontal flip (standard augmentation)
    - Val:   taken from training set, no augmentation
    - Test:  no augmentation
    """
    # CIFAR-10 channel mean and std (pre-computed over training set)
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Download full training set with train augmentation and clean copy for val
    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    full_val   = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=eval_transform)
    test_set   = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_transform)

    # Deterministic split: last val_split fraction becomes validation
    n_total = len(full_train)                          # 50 000
    n_val   = int(n_total * val_split)                 # 5 000
    n_train = n_total - n_val                          # 45 000

    indices    = list(range(n_total))
    train_idx  = indices[:n_train]
    val_idx    = indices[n_train:]

    train_subset = torch.utils.data.Subset(full_train, train_idx)
    val_subset   = torch.utils.data.Subset(full_val,   val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader


def get_cifar100_loaders(
    data_dir: str = "./data/downloads",
    batch_size: int = 128,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Returns (train_loader, test_loader) for CIFAR-100."""
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR100(root=data_dir, train=True,  download=True, transform=train_transform)
    test_set  = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=eval_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader
