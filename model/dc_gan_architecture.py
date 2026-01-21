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
import torchvision.utils as vutils
from typing import Tuple, Optional
from torchinfo import summary

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
        
class Discriminator(nn.Module):
    """
    DC GAN Discriminator Architecture.

    The discriminator takes an image as input and outputs a probability indicating whether the image is real or fake.

    Architecture:
        Input: (batch_size, nc, 64, 64) - Input Image

        Layer 1: Conv2d(nc, ndf, 4, 2, 1)        -> (batch_size, ndf, 32, 32)
        Layer 2: Conv2d(ndf, ndf*2, 4, 2, 1)     -> (batch_size, ndf*2, 16, 16)
        Layer 3: Conv2d(ndf*2, ndf*4, 4, 2, 1)   -> (batch_size, ndf*4, 8, 8)
        Layer 4: Conv2d(ndf*4, ndf*8, 4, 2, 1)   -> (batch_size, ndf*8, 4, 4)
        Layer 5: Conv2d(ndf*8, 1, 4, 1, 0)       -> (batch_size, 1, 1, 1)

        Output: (batch_size, 1) - Probability of the input image being real
    
    Attributes:
        ndf (int): Size of feature maps in the discriminator.
        nc (int): Number of channels in the input image (3 for RGB).
        main: Sequential container for the discriminator layers.
        
    Example:
        >>> discriminator = Discriminator(nc = 3, ndf = 64)
        >>> images = torch.randn(16, 3, 64, 64)
        >>> predictions = discriminator(images)
        >>> print(predictions.shape) # torch.Size([16, 1])
    """

    def __init__(self, nc: int = 3, ndf: int = 64):
        """
        Initialize the DCGAN Discriminator.

        Args:
            nc (int): Number of channels in the input image (3 for RGB).
            ndf (int): Size of feature maps in the discriminator.
        """
        super().__init__()
        self.ndf = ndf
        self.nc = nc
        self.discriminator_network = nn.Sequential(
            # Layer 1: Downsample to 32 x 32
            # Input: (batch_size, nc, 64, 64)
            # Output: (batch_size, ndf, 32, 32)
            nn.Conv2d(in_channels = nc, out_channels = ndf, kernel_size = 4,
                      stride = 2, padding = 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            # Layer 2: Downsample to 16 x 16
            # Input: (batch_size, ndf, 32, 32)
            # Output: (batch_size, ndf*2, 16, 16)
            nn.Conv2d(in_channels = ndf, out_channels = ndf * 2, kernel_size = 4,
                      stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace = True),
            # Layer 3: Downsample to 8 x 8
            # Input: (batch_size, ndf*2, 16, 16)
            # Output: (batch_size, ndf*4, 8, 8)
            nn.Conv2d(in_channels = ndf * 2, out_channels = ndf * 4, kernel_size = 4,
                      stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace = True),
            # Layer 4: Downsample to 4 x 4
            # Input: (batch_size, ndf*4, 8, 8)
            # Output: (batch_size, ndf*8, 4, 4)
            nn.Conv2d(in_channels = ndf * 4, out_channels = ndf * 8, kernel_size = 4,
                      stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace = True),
            # Layer 5: Final layer
            # Input: (batch_size, ndf*8, 4, 4)
            # Output: (batch_size, 1, 1, 1)
            nn.Conv2d(in_channels = ndf * 8, out_channels = 1, kernel_size = 4,
                      stride = 1, padding = 0, bias = False),
            nn.Sigmoid()
        )
    
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        Initialize network weights as per DCGAN paper guidelines.
        
        Args: 
            module (nn.Module): The module to initialize.
        """
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(module.weight, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the discriminator network.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, nc, 64, 64).
            
        Returns:
            torch.Tensor: Probability tensor of shape (batch_size, 1, 1, 1) with values in [0, 1]
              indicating real/fake.
        """
        return self.discriminator_network(x)

# Factory function to create DCGAN with both generator and discriminator
def create_dcgan(nc: int = 3, ngf: int = 64, ndf: int = 64, nz: int = 100, 
                 device: str = 'cuda') -> Tuple[Generator, Discriminator]:
    """
    This is a convenience function to create a DCGAN model with both generator and discriminator
    with matching configurations and move them to specified device.

    Args: 
        nc (int): Number of channels in the images (3 for RGB).
        ngf (int): Size of feature maps in the generator.
        ndf (int): Size of feature maps in the discriminator.
        nz (int): Size of the latent vector (dimensionality of noise).
        device (str): Device to move the models to ('cpu' or 'cuda').
    
    Returns:
        Tuple[Generator, Discriminator]: A tuple containing the generator and discriminator models.

    Example:
        >>> generator, discriminator = create_dcgan(nc = 3, ngf = 64, ndf = 64, nz = 100, device = 'cuda')
        >>> print(generator)
        >>> print(discriminator)
    """
    generator = Generator(nc = nc, ngf = ngf, nz = nz).to(device)
    discriminator = Discriminator(nc = nc, ndf = ndf).to(device)
    
    return generator, discriminator

def count_parameters(module: nn.Module) -> int:
    """
    Count the number of trainable parameters in a pyTorch model.

    Args:
        module (nn.Module): The model to count parameters for.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def model_summary(generator: Generator, discriminator: Discriminator) -> None:
    """
    Print a summary of the model architecture using torchinfo.

    Args:
        generator (Generator): The generator model to summarize.
        discriminator (Discriminator): The discriminator model to summarize.
    """
    print("=" * 60)
    print("Generator Summary:")
    print("=" * 60)

    print("\n[Generator]")
    print(f" Latent Vector Size (nz): {generator.nz}")
    print(f" Number of Channels (nc): {generator.nc}")
    print(f" Feature Map Size (ngf): {generator.ngf}")
    print(f" Trainable Parameters: {count_parameters(generator)}")

    summary(generator, input_size = (1, generator.nz, 1, 1))

    print("=" * 60)
    print("Discriminator Summary:")
    print("=" * 60)

    print("\n[Discriminator]")
    print(f" Number of Channels (nc): {discriminator.nc}")
    print(f" Feature Map Size (ndf): {discriminator.ndf}")
    print(f" Trainable Parameters: {count_parameters(discriminator)}")

    summary(discriminator, input_size = (1, discriminator.nc, 64, 64))

if __name__ == "__main__":
    # local test of models
    print("Testing DCGAN Implementation")

    # Create models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen, disc = create_dcgan(device = device)
    
    # Print model summaries
    model_summary(gen, disc)

    # Test forward passes
    print("\nTesting [Generator] Forward Passes:")
    z = torch.randn(4, gen.nz, 1, 1, device = device)
    fake_images = gen(z)
    print(f" Input Shape: {z.shape}")
    print(f" Output Shape: {fake_images.shape}")
    vutils.save_image(fake_images, r"model/test_output/test_generated_images.png", normalize = True, nrow = 2)
    print(" Generated images saved to 'test_output/test_generated_images.png'")

    print("\nTesting [Discriminator] Forward Passes:")
    predictions = disc(fake_images)
    print(f" Input Shape: {fake_images.shape}")
    print(f" Output Shape: {predictions.shape}")
    print(f" Output Range: [{predictions.min():.3f}, {predictions.max():.3f}]")

    print("\n✅ DCGAN Implementation Test Completed.")