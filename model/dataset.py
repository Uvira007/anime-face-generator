"""
Dataset utilities for Anime Face Generator

This module provides dataset loading, preprocessing and augmentation
functionalities for training the Anime Face Generator model.

The primary dataset is the Anime Face Dataset from kaggle, which contains
~63000 anime character face images under CC0 license.

Dataset URL: https://www.kaggle.com/datasets/splcher/animefacedataset
License: CC0 (Public Domain)

Author: Arivirku Iniyan
License: MIT
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class AnimeFaceDataset(Dataset):
    """Custom Dataset for loading Anime Face images.
    
    This dataset class loads images from the specified directory,
    applies necessary transformations and augmentations, and provides
    an interface compatible with PyTorch DataLoader.
    
    Attributes:
        root_dir (str): Directory containing anime face images.
        transform (Callable, optional): Transformations to apply to images.
        image_files (List[Path]): List of image file paths in the dataset.

    Examples:
        >>> dataset = AnimeFaceDataset(
        ...     root_dir="./data/anime_faces",
        ...     transform=transforms.Compose([
    """
    SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    def __init__(self, root_dir: str, transform: Optional[Callable] = None, 
                 max_samples: Optional[int] = None):
        """
        Initializes the AnimeFaceDataset.
        
        Args:
            root_dir (str): Directory containing anime face images.
            transform (Callable, optional): Transformations to apply to images.
            max_samples (int, optional): Maximum number of samples to load from the dataset.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # collect all valid images from the root dir
        self.image_paths = self._collect_image_paths()
        
        #limit samples if specified
        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]

        print(f"Loaded {len(self.image_paths)} images from {self.root_dir}")


    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)
    

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves the image at the specified index after applying transformations.
        
            Args:
                idx (int): Index of the image to retrieve.

            Returns:
                torch.Tensor: The transformed image tensor.
        """

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Apply transform if specified
        if self.transform:
            image = self.transform(image)
            return image

        return transforms.ToTensor()(image)

    def _collect_image_paths(self) -> list[Path]:
        """Retrieves all image file paths from the root directory."""
        image_files = []
        for ext in self.SUPPORTED_IMAGE_EXTENSIONS:
            image_files.extend(self.root_dir.rglob(f"*{ext}"))
            image_files.extend(self.root_dir.rglob(f"*{ext.upper()}"))
        return sorted(image_files)
    
    # Transformation pipeline for preprocessing images
def get_transforms(image_size: int, normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                   normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5), 
                   augment: bool = False) -> transforms.Compose:
    
    """
    Create image transformation pipeline. The transforms normalizes the image to [-1, 1] range
    which is optimal for training with tanh activation    
    
    Args:
        image_size (int): Desired size to resize images to (image_size x image_size).
        normalize_mean (Tuple[float, float, float]): Mean for normalization.
        normalize_std (Tuple[float, float, float]): Standard deviation for normalization.
        augment (bool): Whether to include data augmentation (random horizontal flip).
    
    Returns:
        transforms.Compose: Composed transformations.
    """
    transform_list = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size)
    ]

    # optional augmentations
    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomRotation(degrees = 15)
        ])

    # Final transforms
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean = normalize_mean, std = normalize_std)
    ])

    return transforms.Compose(transform_list)


def create_dataloader(root_dir: str, image_size: int = 64, batch_size: int = 128,
                      shuffle: bool = True, num_workers: int = 4,
                      pin_memory: bool = True, augment: bool = False,
                      max_samples: Optional[int] = None) -> DataLoader:
    """
    Creates a DataLoader for the Anime Face Dataset.
    
    Args:
        root_dir (str): Directory containing anime face images.
        image_size (int): Desired size to resize images to (image_size x image_size).
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of subprocesses to use for data loading.
        pin_memory (bool): Whether to use pin_memory in DataLoader for faster GPU transfer.
        augment (bool): Whether to include data augmentation.
        max_samples (int, optional): Maximum number of samples to load from the dataset.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    transform = get_transforms(image_size=image_size, augment=augment)
    dataset = AnimeFaceDataset(root_dir=root_dir, transform=transform, max_samples=max_samples)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    
    return dataloader


def download_dataset(output_dir: str = "./data/anime_faces", 
                     kaggle_dataset: str = "splcher/animefacedataset") -> str:
    """
    Download the anime dataset from Kaggle.

    Prerequisites:
        - Kaggle API credentials configured (~/.kaggle/kaggle.json)
        - kaggle package installed (pip install kaggle)

    Args:
        output_dir (str): Directory to save the downloaded dataset.
        kaggle_dataset (str): Kaggle dataset identifier.

    Returns:
        str: Path to the downloaded dataset directory.

    Note: If you don't have kaggle package installed or prefer manual download, please visit
    https://www.kaggle.com/datasets/splcher/animefacedataset to download the dataset manually.
    """
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        import kaggle
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        api_token = os.getenv('KAGGLE_API_TOKEN')
        print(f"Downloading dataset '{kaggle_dataset}' to '{output_path.resolve()}'...")

        # download and unzip the dataset
        kaggle.api.dataset_download_files(kaggle_dataset, path=str(output_path), unzip=True)
        print(f"Dataset downloaded and extracted to '{output_path.resolve()}'.")
        return str(output_path)
    
    except ImportError:
        print("Kaggle API not installed. Please install it using 'pip install kaggle'.")
        raise
    
    except Exception as e:
        print(f"An error occurred while downloading the dataset: {e}")
        raise

def visualize_batch(dataloader: DataLoader, num_images: int = 64, save_path: Optional[str] = None,
                    ) -> None:
    """
    Visualizes a batch of images from the dataloader.

    Args:
        dataloader (DataLoader): DataLoader to fetch images from.
        num_images (int): Number of images to visualize.

    Returns:
        np.ndarray: Array of images visualized.
    """
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils

    # Get a batch of images
    data_iter = iter(dataloader)
    batch = next(data_iter)
    images = batch[:num_images]

    # Create a grid of images
    img_grid = vutils.make_grid(images, normalize=True, scale_each=True)

    # Plot the images
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Sample Images from Anime Face Dataset")
    plt.imshow(img_grid.permute(1, 2, 0).cpu().numpy())
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi = 150)
        print(f"Saved visualization to {save_path}")
    plt.show()

if __name__ == "__main__":
    temp = AnimeFaceDataset()