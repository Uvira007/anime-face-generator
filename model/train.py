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

import argparse
from typing import Tuple, Optional, Dict, List, Union
import time
import json


import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import DCGANConfig, DatabricksConfig
from dc_gan_architecture import Generator, Discriminator, create_dcgan, model_summary
from dataset import create_dataloader

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
    
    def train(self, dataloader:DataLoader) -> Dict[str, List[float]]:
        """
        Train the DCGAN for the specified number of epochs
        
        Args:
            dataloader (DataLoader): DataLoader for the training images.

        Returns:
            Dict[str, List[float]]: Training history containing losses and metrics.
        """
        print("\n" + "="* 60)
        print("Starting DCGAN Training...")
        print("="* 60 + "\n")
        start_time = time.time()
        iteration = 0

        for epoch in range(1, self.config.num_epochs):
            epoch_start_time = time.time()
            epoch_D_loss = 0.0
            epoch_G_loss = 0.0
            num_batches = 0

            # Progress bar for current epoch
            pbar = tqdm(dataloader, 
                        desc=f"Epoch {epoch}/{self.config.num_epochs}", unit="batch", leave = True)
            
            for batch_idx, real_images in enumerate(pbar):
                # Perform Training step
                loss_D, loss_G, D_x, D_G_z1, D_G_z2 = self.train_step(real_images)

                # Save losses and metrics
                self.history["D_losses"].append(loss_D)
                self.history["G_losses"].append(loss_G)
                self.history["D_x"].append(D_x)
                self.history["D_G_z1"].append(D_G_z1)
                self.history["D_G_z2"].append(D_G_z2)

                iteration += 1
                epoch_D_loss += loss_D
                epoch_G_loss += loss_G
                num_batches += 1
                pbar.set_postfix({
                    "D_Loss": f"{loss_D:.4f}",
                    "G_Loss": f"{loss_G:.4f}",
                    "D(x)": f"{D_x:.4f}",
                    "D(G(z))": f"{D_G_z1:.4f}/{D_G_z2:.4f}"
                })

            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_D_loss = epoch_D_loss / num_batches
            avg_G_loss = epoch_G_loss / num_batches

            print(f"Epoch [{epoch+1}/{self.config.num_epochs}] "
                  f"Loss_D: {avg_D_loss:.4f} Loss_G: {avg_G_loss:.4f} "
                  f"Time: {(epoch_time):.2f}s")
            
            # Save samples periodically
            if (epoch + 1) % self.config.sample_every == 0:
                self._save_samples(epoch + 1)

            # Save checkpoints and sample images at intervals
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(epoch + 1)
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "="* 60)
        print(f"\nTraining complete in {total_time/60:.2f} minutes.")
        print("="* 60 + "\n")

        # Save final models
        self._save_checkpoint('final')
        self._save_samples('final')
        self._save_training_curves()

        return self.history
    
    def _save_samples(self, epochs: Union[int, str]) -> None:
        """Generates and saves sample images from the fixed noise vector."""
        self.generator.eval()
        with torch.no_grad():
            fake_images = self.generator(self.fixed_noise).detach().cpu()
        
        self.generator.train()
        
        # Create grid of images
        grid = vutils.make_grid(fake_images, padding=2, normalize=True)
        save_path = f"{self.config.output_dir}//samples_epoch_{epochs}.png"

        plt.figure(figsize=(10,10))
        plt.axis("off")
        plt.title(f"Generated Anime faces at Epoch {epochs}")
        plt.imshow(np.transpose(grid.numpy(), (1,2,0)))
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f" Sample images saved to {save_path}")

    def _save_checkpoint(self, epoch: Union[int, str]) -> None:
        """Saves the model checkpoints for Generator and Discriminator."""
        checkpoint_path = f"{self.config.checkpoint_dir}//checkpoint_epoch_{epoch}.pth"

        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'history': self.history,
            'config': {
                'nc': self.config.nc,
                'nz': self.config.nz,
                'ngf': self.config.ngf,
                'ndf': self.config.ndf,
                'image_size': self.config.image_size
            }
            }
        torch.save(checkpoint, checkpoint_path)
        print(f" Model checkpoint saved to {checkpoint_path}")

        # Also save generator model separately for easy loading during inference
        gen_path = f"{self.config.checkpoint_dir}//generator_epoch_{epoch}.pth"
        torch.save(self.generator.state_dict(), gen_path)

    def _save_training_curves(self) -> None:
        """Saves the training loss curves for Generator and Discriminator."""
        save_path = f"{self.config.output_dir}//training_curves.png"
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curves
        axes[0].plot(self.history['G_losses'], label='Generator Loss', color='red', alpha = 0.7)
        axes[0].plot(self.history['D_losses'], label='Discriminator Loss', color='blue', alpha = 0.7)
        axes[0].set_title('Training Losses')
        axes[0].set_xlabel('Iterations')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha = 0.3)

        # D(x) and D(G(z)) curves
        axes[1].plot(self.history['D_x'], label='D(x) - Real', color='green', alpha = 0.7)
        axes[1].plot(self.history['D_G_z1'], label='D(G(z)) - Fake', color='orange', alpha = 0.7)
        axes[1].axhline(y=0.5, color='red', linestyle='--', alpha = 0.5, label='Ideal Threshold')
        axes[1].set_title('Discriminator predictions')
        axes[1].set_xlabel('Iterations')
        axes[1].set_ylabel('Discriminator Output')
        axes[1].legend()
        axes[1].grid(True, alpha = 0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f" Training curves saved to {save_path}")

        # Also save history as json
        history_path = f"{self.config.output_dir}//training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Loads model and optimizer states from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        
        Returns:
            int: The epoch number to resume training from.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.history = checkpoint['history']
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']

        print(f" Loaded checkpoint from {checkpoint_path}")
        
        return checkpoint['epoch']
    
def parse_args() -> argparse.Namespace:
    """Parses command line arguments for training configuration."""

    parser = argparse.ArgumentParser(description="DCGAN Training Script for Anime Face Generator")

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data/anime_faces',
                        help='Directory containing the anime face dataset.')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of samples per training batch.')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate for optimizers.')

    # Model arguments
    parser.add_argument('--nz', type=int, default=100,
                        help='Dimension of the latent vector (input to generator).')
    parser.add_argument('--ngf', type=int, default=64,
                        help='Base feature map size for generator.')
    parser.add_argument('--ndf', type=int, default=64,
                        help='Base feature map size for discriminator.')

    # Other arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a checkpoint file to resume training from.')
    parser.add_argument('--device', type=str, default='cuda', choices = ['cuda', 'cpu'],
                        help='Device to run the model on (cuda or cpu).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers.')
    parser.add_argument('--databricks', action='store_true',
                        help='Flag to indicate running in Azure Databricks environment.'
                        )
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()

    # Create configuration
    if args.databricks:
        config = DatabricksConfig()
    else:
        config = DCGANConfig()

    # Override config with command line args
    config.data_dir = args.data_dir
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.nz = args.nz
    config.ngf = args.ngf
    config.ndf = args.ndf
    config.device = args.device
    config.seed = args.seed
    config.workers = args.workers

    #Create directories
    config.__post_init__()

    # Create DataLoader
    dataloader = create_dataloader(
        root_dir = config.data_dir,
        image_size = config.image_size,
        batch_size = config.batch_size,
        shuffle = True,
        num_workers = config.workers,
        pin_memory = config.pin_memory,
        augment = True
    )

    # Initialize Trainer
    trainer = DCGANTrainer(config)

    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Start training
    history = trainer.train(dataloader)

if __name__ == "__main__":
    main()