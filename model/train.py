"""
DCGAN training script for Anime Face Generator
This script provides the complete training pipeline for the DCGAN model.
It supports both local training and Azure databricks environments.

Usage:
    Local training:
        python model/train.py --data_dir ./data/anime_faces --epochs 100
        
    Azure Databricks:
        python model/train.py --config config.yaml
        
Training Algorithm:
    The DCGAN training alternates between
    1. Training Discriminator: Maximizing log(D(x)) + log(1 - D(G(z)))
    2. Training Generator: Minimizing log(D(G(z))) via maximizing log(D(G(z)))
    
    where D is the Discriminator, G is the Generator, x is real data and z is random noise.
    
Author: Arivirku Iniyan
License: MIT
"""

from typing import Tuple, Optional
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from config import DCGANConfig
from dc_gan_architecture import Generator, Discriminator, create_dcgan, model_summary
from dataset import AnimeFaceDataset

class DCGANTrainer:
    """
    Trainer class for DCGAN model.
    This class encapsulates the training loop, model saving, checkpointing and 
    logging functionalities.
    
    Attributes:
        config (DCGANConfig): Configuration settings for training.
        generator (Generator): The generator model.
        discriminator (Discriminator): The discriminator model.
        optimizer_G (torch.optim.Optimizer): Optimizer for the generator.
        optimizer_D (torch.optim.Optimizer): Optimizer for the discriminator.
        criterion (torch.nn.Module): Loss function (Binary Cross Entropy).
        device (str): Device to run the model on ('cuda' or 'cpu').
        fixed_noise (torch.Tensor): Fixed noise vector for generating sample images during training.
    """

    def __init__(self, config: DCGANConfig):
        """
        Initializes the DCGANTrainer with models, optimizers and loss function.
        
        Args:
            config (DCGANConfig): Configuration settings for training.
        """
        self.config = config
        self.device = config.device

        self._set_seed(config.seed)

        # Initialize Generator and Discriminator
        self.generator, self.discriminator = create_dcgan(
            nc=config.nc, nz=config.nz, ngf=config.ngf, ndf=config.ndf, device=str(self.device)
        )

        model_summary(self.generator, self.discriminator)

        # Initialize optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2)
        )

        # Loss function
        self.criterion = nn.BCELoss()

        # Fixed noise for generating sample images
        self.fixed_noise = torch.randn(config.num_samples, config.nz, 1, 1, device=self.device)

        # Training history
        self.history = {
            "G_losses": [],
            "D_losses": [],
            "D_x": [],
            "D_G_z1": [],
            "D_G_z2": []
        }

        self.real_label = 1.0
        self.fake_label = 0.0

    def _set_seed(self, seed: int):
        """Sets the random seed for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train_step(self, real_images: torch.Tensor) -> Tuple[float, float, float, float, float]:
        """
        Performs a single training step for both Discriminator and Generator.

        This implements the standard GAN training procedure.
        1. Update discriminator with real and fake images
        2. Update generator to fool the discriminator
        
        Args:
            real_images (torch.Tensor): Batch of real images from the dataset.

        Returns:
            Tuple[float, float, float, float, float]: 
                - D_loss: Discriminator loss
                - G_loss: Generator loss
                - D_x: Discriminator output on real images
                - D_G_z1: Discriminator output on fake images (before generator update)
                - D_G_z2: Discriminator output on fake images (after generator update)
        """
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)

        # Create labels
        real_labels = torch.full((batch_size,), self.real_label, device=self.device)
        fake_labels = torch.full((batch_size,), self.fake_label, device=self.device)
        #==================== Train Discriminator ====================

        # Zero gradients
        self.discriminator.zero_grad()
        # ------ Train with real images -----------
        # Forward pass real batch through D
        output_real = self.discriminator(real_images).view(-1)
        D_x = output_real.mean().item()

        # Calculate loss on real batch
        loss_D_real = self.criterion(output_real, real_labels)
        loss_D_real.backward()

        # ------ Train with fake images -----------
        # Generate fake images
        noise = torch.randn(batch_size, self.config.nz, 1, 1, device=self.device)
        fake_images = self.generator(noise)

        # Forward pass fake batch through D
        output_fake = self.discriminator(fake_images.detach()).view(-1)
        D_G_z1 = output_fake.mean().item()

        # Calculate loss on fake batch
        loss_D_fake = self.criterion(output_fake, fake_labels)
        loss_D_fake.backward()

        # Update Discriminator
        loss_D = loss_D_real + loss_D_fake
        self.optimizer_D.step()

        #==================== Train Generator ====================
        # Goal: maximize log(D(G(z))) = minimize log(1 - D(G(z)))
        # Zero gradients
        self.generator.zero_grad()

        # Forward pass fake images through D again
        output_fake2 = self.discriminator(fake_images).view(-1)
        D_G_z2 = output_fake2.mean().item()

        # Calculate Generator loss
        loss_G = self.criterion(output_fake2, real_labels)  # We want to fool the discriminator
        loss_G.backward()

        # Update Generator
        self.optimizer_G.step()

        return loss_D.item(), loss_G.item(), D_x, D_G_z1, D_G_z2
    
    def train():
        """ TODO: Start from this function"""