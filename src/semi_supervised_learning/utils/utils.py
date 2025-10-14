import torch


def weak_augmentation(data: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
    """
    Adds Gaussian noise to a tensor.

    Args:
        data (torch.Tensor): The input tensor of tabular data.
        noise_std (float): The standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: The augmented tensor with added noise.
    """
    noise = torch.randn_like(data) * noise_std
    return data + noise


def strong_augmentation(data: torch.Tensor, drop_rate: float = 0.3) -> torch.Tensor:
    """
    Randomly sets a percentage of features to zero.

    Args:
        data (torch.Tensor): The input tensor of tabular data.
        drop_rate (float): The percentage of features to drop (set to zero).

    Returns:
        torch.Tensor: The augmented tensor with some features set to zero.
    """
    mask = torch.rand_like(data) > drop_rate
    return data * mask
