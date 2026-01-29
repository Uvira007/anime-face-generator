"""
ONNX Inference Utilities for Anime Face Generator.

This module provides optimized inference capabilities using ONNX Runtime
for generating anime faces from the trained DCGAN model.

Features:
- Efficient ONNX Runtime inference
- Batch generation support
- Image post-processing and encoding
- Latent space interpolation

Author: [Your Name]
License: MIT
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple, Union
from io import BytesIO
import base64

import numpy as np
import onnxruntime as ort
from PIL import Image


class AnimeFaceGenerator:
    """
    Anime Face Generator using ONNX Runtime.
    
    This class provides high-performance inference for generating
    anime character faces using the ONNX-exported DCGAN Generator.
    
    Attributes:
        session: ONNX Runtime inference session
        input_name: Name of the model input tensor
        output_name: Name of the model output tensor
        nz: Latent vector dimension
        
    Example:
        >>> generator = AnimeFaceGenerator("generator.onnx")
        >>> images = generator.generate(num_images=4)
        >>> generator.save_images(images, "output/")
    """
    
    def __init__(
        self,
        model_path: str,
        nz: int = 100,
        use_gpu: bool = False
    ):
        """
        Initialize the Anime Face Generator.
        
        Args:
            model_path: Path to the ONNX model file
            nz: Latent vector dimension (must match training config)
            use_gpu: Whether to use GPU for inference (requires onnxruntime-gpu)
        """
        self.model_path = model_path
        self.nz = nz
        
        # Configure execution providers
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # Session options for optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = os.cpu_count() // 2 or 1
        
        # Create inference session
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"[Generator] Loaded model from {model_path}")
        print(f"[Generator] Using providers: {self.session.get_providers()}")
        print(f"[Generator] Latent dimension: {nz}")
    
    def generate(
        self,
        num_images: int = 1,
        seed: Optional[int] = None,
        latent_vectors: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate anime face images.
        
        Args:
            num_images: Number of images to generate (ignored if latent_vectors provided)
            seed: Random seed for reproducibility (optional)
            latent_vectors: Pre-defined latent vectors of shape (N, nz, 1, 1)
            
        Returns:
            Generated images as numpy array of shape (N, H, W, 3) with values in [0, 255]
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Generate or use provided latent vectors
        if latent_vectors is not None:
            z = latent_vectors.astype(np.float32)
            if z.ndim == 2:
                z = z.reshape(-1, self.nz, 1, 1)
        else:
            z = np.random.randn(num_images, self.nz, 1, 1).astype(np.float32)
        
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: z}
        )[0]
        
        # Post-process: Convert from [-1, 1] to [0, 255] RGB images
        images = self._postprocess(outputs)
        
        return images
    
    def generate_from_seed(self, seed: int, num_images: int = 1) -> np.ndarray:
        """
        Generate images with a specific seed for reproducibility.
        
        Args:
            seed: Random seed
            num_images: Number of images to generate
            
        Returns:
            Generated images
        """
        return self.generate(num_images=num_images, seed=seed)
    
    def interpolate(
        self,
        z1: np.ndarray,
        z2: np.ndarray,
        num_steps: int = 10
    ) -> np.ndarray:
        """
        Interpolate between two latent vectors.
        
        This creates a smooth transition between two generated faces,
        useful for visualizing the latent space.
        
        Args:
            z1: Starting latent vector of shape (nz,) or (1, nz, 1, 1)
            z2: Ending latent vector of shape (nz,) or (1, nz, 1, 1)
            num_steps: Number of interpolation steps
            
        Returns:
            Array of interpolated images
        """
        # Normalize input shapes
        z1 = z1.flatten()
        z2 = z2.flatten()
        
        # Linear interpolation in latent space
        alphas = np.linspace(0, 1, num_steps)
        latent_vectors = np.array([
            (1 - alpha) * z1 + alpha * z2 
            for alpha in alphas
        ])
        
        # Reshape for inference
        latent_vectors = latent_vectors.reshape(-1, self.nz, 1, 1)
        
        return self.generate(latent_vectors=latent_vectors)
    
    def _postprocess(self, images: np.ndarray) -> np.ndarray:
        """
        Post-process model outputs to RGB images.
        
        Converts from (N, C, H, W) in [-1, 1] to (N, H, W, C) in [0, 255].
        
        Args:
            images: Model output of shape (N, C, H, W) with values in [-1, 1]
            
        Returns:
            Processed images of shape (N, H, W, C) with values in [0, 255]
        """
        # Denormalize from [-1, 1] to [0, 1]
        images = (images + 1) / 2.0
        
        # Clip to valid range
        images = np.clip(images, 0, 1)
        
        # Convert to uint8 [0, 255]
        images = (images * 255).astype(np.uint8)
        
        # Transpose from (N, C, H, W) to (N, H, W, C)
        images = np.transpose(images, (0, 2, 3, 1))
        
        return images
    
    def to_pil(self, images: np.ndarray) -> List[Image.Image]:
        """
        Convert numpy images to PIL Images.
        
        Args:
            images: Array of shape (N, H, W, C) with values in [0, 255]
            
        Returns:
            List of PIL Image objects
        """
        return [Image.fromarray(img) for img in images]
    
    def to_base64(
        self,
        images: np.ndarray,
        format: str = 'PNG'
    ) -> List[str]:
        """
        Convert images to base64-encoded strings.
        
        Useful for sending images via API responses.
        
        Args:
            images: Array of shape (N, H, W, C) with values in [0, 255]
            format: Image format ('PNG' or 'JPEG')
            
        Returns:
            List of base64-encoded image strings
        """
        pil_images = self.to_pil(images)
        base64_images = []
        
        for img in pil_images:
            buffer = BytesIO()
            img.save(buffer, format=format)
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            base64_images.append(base64_str)
        
        return base64_images
    
    def save_images(
        self,
        images: np.ndarray,
        output_dir: str,
        prefix: str = 'anime_face',
        format: str = 'png'
    ) -> List[str]:
        """
        Save generated images to disk.
        
        Args:
            images: Array of shape (N, H, W, C)
            output_dir: Directory to save images
            prefix: Filename prefix
            format: Image format (png, jpg)
            
        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        pil_images = self.to_pil(images)
        saved_paths = []
        
        for i, img in enumerate(pil_images):
            path = os.path.join(output_dir, f"{prefix}_{i:04d}.{format}")
            img.save(path)
            saved_paths.append(path)
        
        return saved_paths
    
    def create_grid(
        self,
        images: np.ndarray,
        nrow: int = 8,
        padding: int = 2
    ) -> Image.Image:
        """
        Create a grid of images.
        
        Args:
            images: Array of shape (N, H, W, C)
            nrow: Number of images per row
            padding: Padding between images
            
        Returns:
            PIL Image of the grid
        """
        n, h, w, c = images.shape
        ncol = (n + nrow - 1) // nrow
        
        # Calculate grid dimensions
        grid_h = ncol * (h + padding) + padding
        grid_w = nrow * (w + padding) + padding
        
        # Create grid (white background)
        grid = np.ones((grid_h, grid_w, c), dtype=np.uint8) * 255
        
        # Place images
        for idx in range(n):
            row = idx // nrow
            col = idx % nrow
            
            y = row * (h + padding) + padding
            x = col * (w + padding) + padding
            
            grid[y:y+h, x:x+w] = images[idx]
        
        return Image.fromarray(grid)


def load_generator(
    model_path: str = "generator.onnx",
    nz: int = 100
) -> AnimeFaceGenerator:
    """
    Factory function to load the generator.
    
    Args:
        model_path: Path to ONNX model
        nz: Latent vector dimension
        
    Returns:
        Configured AnimeFaceGenerator instance
    """
    return AnimeFaceGenerator(model_path=model_path, nz=nz)


if __name__ == "__main__":
    # Test the generator
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='generator.onnx')
    parser.add_argument('--num', type=int, default=16)
    parser.add_argument('--output', type=str, default='./generated')
    args = parser.parse_args()
    
    # Load generator
    generator = load_generator(args.model)
    
    # Generate images
    print(f"Generating {args.num} images...")
    images = generator.generate(num_images=args.num, seed=42)
    print(f"Generated images shape: {images.shape}")
    
    # Save images
    paths = generator.save_images(images, args.output)
    print(f"Saved {len(paths)} images to {args.output}")
    
    # Create and save grid
    grid = generator.create_grid(images, nrow=4)
    grid.save(f"{args.output}/grid.png")
    print(f"Saved grid to {args.output}/grid.png")