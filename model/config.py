"""
Configuration setting for the Anime Face Generator application.

This module contains all the hyperparameters and congiguration settings
required for training the DCGAN model for anime face generation.
Settings are optimized for the anime face dataset that contains 64 x 64 pixel images
and can be adjusted for different hardware configurations.

Author: Arivirku Iniyan
License: MIT
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, Optional

@dataclass
class DCGANConfig:
    """Configuration settings for DCGAN model 
    
    This dataclass holds all the hyperparameters and paths needed for
    training the DGCAN model. Default values are optimized for 
    the Anime Face Dataset (64x64 images).
    
    Attributes:
        image_size (int): Size of the input images (assumed square).
        nc (int): Number of image channels (3 for RGB).
        nz (int): Dimension of the latent vector (input to generator).
        ngf (int): Base feature map size for generator.
        ndf (int): Base feature map size for discriminator.
        batch_size (int): Number of samples per training batch.
        lr (float): Learning rate for optimizers.
        beta1 (float): Beta1 hyperparameter for Adam optimizer.
        num_epochs (int): Number of training epochs.
        device (str): Device to run the model on ('cuda' or 'cpu').
        workers: Number of data loading workers.
    """
    #==================== Model Architecture ========================
    image_size: int = 64  # Size of the input images (assumed square)
    nc: int = 3  # Number of image channels (3 for RGB)
    nz: int = 100  # Dimension of the latent vector (input to generator)
    ngf: int = 64  # Base feature map size for generator
    ndf: int = 64  # Base feature map size for discriminator

    #==================== Training Hyperparameters =======================
    num_epochs: int = 50  # Number of training epochs
    batch_size: int = 128  # Number of samples per training batch
    lr: float = 0.0002  # Learning rate for optimizers
    beta1: float = 0.5  # Beta1 hyperparameter for Adam optimizer
    beta2: float = 0.999  # Beta2 hyperparameter for Adam optimizer
    
    #==================== DataLoader Settings =========================
    workers: int = 4  # Number of data loading workers
    pin_memory: bool = True  # Whether to use pin_memory in DataLoader for faster GPU transfer
    device: str = field(default_factory=lambda: "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")

    #==================== Path Settings ==============================
    data_dir: str = "./data/anime_faces"  # Directory containing the anime face dataset
    checkpoint_dir: str = "./checkpoints"  # Directory to save model checkpoints
    log_dir: str = "./logs"  # Directory to save training logs
    output_dir: str = "./outputs"  # Directory to save generated images
    
    #==================== Checkpoint Settings ==========================
    save_every: int = 10  # Save model checkpoint every n epochs
    sample_every: int = 5  # Generate sample images every n epochs
    num_samples: int = 64  # Number of sample images to generate during training

    #==================== ONNX Export Settings =======================
    onnx_output_path: str = "./exports/generator.onnx"  # Path to save the exported ONNX model
    onnx_opset_version: int = 14  # ONNX opset version for export
    onnx_dynamic_axes: bool = True  # Dynamic batch size

    #==================== Miscellaneous Settings ======================
    seed: int = 42  # Random seed for reproducibility
    
    def __post_init__(self):
        """Create necessary directories if they don't exist."""
        for dir_path in [self.checkpoint_dir, self.log_dir, 
                         self.output_dir, os.path.dirname(self.onnx_output_path)]:
            os.makedirs(dir_path, exist_ok=True)

    @property
    def generator_input_shape(self) -> Tuple[int, int, int, int]:
        """Returns the expected input shape for the generator input tensor."""
        return (1, self.nz, 1, 1)
    
    @property
    def image_shape(self) -> Tuple[int, int, int]:
        """Returns the shape of the generated images (C, H, W)."""
        return (self.nc, self.image_size, self.image_size)
    

@dataclass
class DatabricksConfig(DCGANConfig):
    """Configuration optimized for Azure Databricks Standard_L8s_v2.

    standard_L8s_v2 specs:
    - 8 vCPUs
    - 64 GB RAM
    - Local NVMe SSD Storage

    Settings are adjusted for optimal performance on this instance
    """
    #override for better GPU utilization
    batch_size: int = 256
    workers: int = 8

    #Databricks specific paths (using DBFS)
    data_dir: str = "/dbfs/FileStore/anime_faces"  # Databricks file system path
    checkpoint_dir: str = "/dbfs/FileStore/checkpoints"
    log_dir: str = "/dbfs/FileStore/logs"
    output_dir: str = "/dbfs/FileStore/outputs"
    onnx_output_path: str = "/dbfs/FileStore/exports/generator.onnx"

# Default configuration instance
default_config = DCGANConfig()