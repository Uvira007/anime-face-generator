"""
Deep Convolutional Generative Adversarial Network (DCGAN) Implementation.

This module defines the architecture for both the Generator and Discriminator following the architrecture
guidelines established in the DCGAN paper by Radford et al. (2015).
References:
- Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. arXiv preprint arXiv:1511.06434. 

Key architetural guidelines from the paper:
1. Replace any pooling layers with strided convolutions (discriminator) 
   and fractional-strided convolutions (generator).
2. Use batch normalization in both the generator and the discriminator (except output layer of G
   and input layer of D).
3. Remove fully connected hidden layers for deeper architectures.
4. Use ReLU activation in the generator for all layers except for the output, which uses Tanh.
5. Use LeakyReLU activation in the discriminator for all layers.

Author: Arivirku Iniyan
License: MIT
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
class Generator(nn.Module):
    """
    DC GAN Generator Architecture.

    The generator takes a latent vector z from a random normal distribution and 
    transforms it into a near-realistic image through a series of transposed convolutional layers.

    Architecture:
        Input: (batch_size, nz, 1, 1) - Latent vector

        Layer 1: ConvTranspose2d(nz, ngf*8, 4, 1, 0)       -> (batch_size, ngf*8, 4, 4)
        Layer 2: ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1)    -> (batch_size, ngf*4, 8, 8)
        Layer 3: ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1)    -> (batch_size, ngf*2, 16, 16)
        Layer 4: ConvTranspose2d(ngf*2, ngf, 4, 2, 1)      -> (batch_size, ngf, 32, 32)
        Layer 5: ConvTranspose2d(ngf, nc, 4, 2, 1)         -> (batch_size, nc, 64, 64)

        Output: (batch_size, nc, 64, 64) - Generated Image in [-1, 1] range

    Attributes:
        nz (int): Size of the latent vector(z).
        ngf (int): Size of feature maps in the generator.
        nc (int): Number of channels in the output image (3 for RGB).
        main: Sequential container for the generator layers.

    Example:
        >>> generator = Generator(nc = 3, nz = 100, ngf = 64)
        >>> z = torch.randn(16, 100, 1, 1)
        >>> fake_images = generator(z)
        >>> print(fake_images.shape) # torch.Size([16, 3, 64, 64])
    """

    def __init__(self, nc: int = 3, ngf: int = 64, nz: int = 100):
        """
        Initialize the DCGAN Generator.

        Args:
            nc (int): Number of channels in the output image (3 for RGB).
            ngf (int): Size of feature maps in the generator.
            nz (int): Size of the latent vector (dimensionality of noise).
        """
        super().__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.generator_network = nn.Sequential(
            # Layer 1: Project latent vector and reshape
            # Input: (batch_size, nz, 1, 1)
            # Output: (batch_size, ngf*8, 4, 4)
            nn.ConvTranspose2d(in_channels=nz, out_channels = ngf * 8, kernel_size = 4, 
                               stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace = True),
            # Layer 2: Upsample to 8 x 8
            # Input : (batch_size, ngf*8, 4, 4)
            # Output: (batch_size, ngf*4, 8, 8)
            nn.ConvTranspose2d(in_channels = ngf * 8, out_channels = ngf * 4, kernel_size = 4,
                               stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace = True),
            # Layer 3: Upsample to 16 x 16
            # Input : (batch_size, ngf*4, 8, 8)
            # Output: (batch_size, ngf * 2, 16, 16)
            nn.ConvTranspose2d(in_channels = ngf * 4, out_channels = ngf * 2, kernel_size = 4,
                               stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace = True),
            # Layer 4: Upsample to 32 x 32
            # Input : (batch_size, ngf*2, 16, 16)
            # Output: (batch_size, ngf, 32, 32)
            nn.ConvTranspose2d(in_channels = ngf * 2, out_channels = ngf, kernel_size = 4,
                                 stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace = True),
            # Output Layer: Generated image 64 x 64 
            # Input : (batch_size, ngf, 32, 32)
            # Output: (batch_size, nc, 64, 64)
            # Tanh activation to scale output to [-1, 1]
            nn.ConvTranspose2d(in_channels = ngf, out_channels = 3, kernel_size = 4,
                               stride = 2, padding = 1, bias = False),
            nn.Tanh()                     
        )

        # Initialize weights
        self.apply(self.__init_weights)

    def __init_weights(self, module: nn.Module):
        """
        Initialize weights for convolutional and batch normalization layers.
        
        Conv layers: Normal distribution with mean = 0 and std = 0.02
        Batch Norm layers: Normal distribution with mean = 1 and std = 0.02 and 
        constant 0 for biases

        Args:
            module (nn.Module): Module to initialize weights for.
        """
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(module.weight, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(module.weight, 1, 0.02)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the generator network.

        Args:
            z (torch.Tensor): Input latent vector of shape (batch_size, nz, 1, 1).

        Returns:
            torch.Tensor: Generated image tensor of shape (batch_size, nc, 64, 64).
        """
        return self.generator_network(z)
    
    def generate(self, num_images: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Generate random images from random latent vectors.

        Args:
            num_images (int): Number of images to generate.
            device (str, optional): Device to perform computation on ('cpu' or 'cuda'). 
                                    If None, uses the device of the model parameters.
        
        Returns:
            Generated Image of shape (num_images, nc x 64 x 64)
        """
        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(num_images, self.nz, 1, 1, device = device)

        with torch.no_grad():
            return self.forward(z)

        


