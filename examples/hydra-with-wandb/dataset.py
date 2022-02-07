import torch
import torchvision
from torchvision import transforms


def get_transform_pair(mean, std):
    fwd_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    inv_transform = transforms.Compose(
        [transforms.Normalize(-mean / std, 1 / std), transforms.ToPILImage()]
    )
    return fwd_transform, inv_transform


def load_mnist(directory, train_batch_size, test_batch_size):
    mean = torch.tensor([0.1307])
    std = torch.tensor([0.3081])
    fwd_transform, inv_transform = get_transform_pair(mean, std)

    train_data = torchvision.datasets.MNIST(
        root=directory, transform=fwd_transform, train=True, download=True
    )
    test_data = torchvision.datasets.MNIST(
        root=directory, transform=fwd_transform, train=False, download=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=test_batch_size, shuffle=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=train_batch_size, shuffle=True
    )
    return ((train_loader, test_loader), (fwd_transform, inv_transform))


def load_cifar10(directory, train_batch_size, test_batch_size):
    mean = torch.tensor([0.49139968, 0.48215827, 0.44653124])
    std = torch.tensor([0.24703233, 0.24348505, 0.26158768])
    fwd_transform, inv_transform = get_transform_pair(mean, std)

    train_data = torchvision.datasets.CIFAR10(
        root=directory, transform=fwd_transform, train=True, download=True
    )
    test_data = torchvision.datasets.CIFAR10(
        root=directory, transform=fwd_transform, train=False, download=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=test_batch_size, shuffle=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=train_batch_size, shuffle=True
    )
    return ((train_loader, test_loader), (fwd_transform, inv_transform))


def get_dataset(dataset_name, batch_size, train_batch, test_batch):
    dataset_loader_fn = globals()[f"load_{dataset_name}"]
    return dataset_loader_fn(batch_size, train_batch, test_batch)
