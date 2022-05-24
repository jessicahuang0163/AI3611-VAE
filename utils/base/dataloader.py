from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WOKERS = 16


def loading_data(path, img_size=28, batch_size=128):
    """MNIST dataloader with (32, 32) sized images."""
    # Resize images so they are a power of 2
    all_transforms = transforms.Compose(
        [transforms.Resize(img_size), transforms.ToTensor(),]
    )
    # Get train and test data
    train_data = datasets.MNIST(
        path + "/data", train=True, download=True, transform=all_transforms
    )
    test_data = datasets.MNIST(path + "/data", train=False, transform=all_transforms)
    # Create dataloaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=NUM_WOKERS
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=NUM_WOKERS
    )
    return train_loader, test_loader
