"""
Proper ONNX Export script for the trained generator

This script exports the pyTorch model to ONNX format correctly
"""

import os
import sys

# CRITICAL: Disable dynamo BEFORE importing torch
os.environ["TORCH_ONNX_USE_EXPERIMENTAL_LOGIC"] = "0"

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class Generator(nn.Module):
    """DCGAN generator Network - must match training architecture"""

    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3):
        super(Generator, self).__init__()

        self.nc = nc
        self.nz = nz
        self.ngf = ngf

        self.generator_network = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.generator_network(input)
    
def export_to_onnx(
        checkpoint_path: str = "./checkpoints/generator_epoch_final.pth",
        output_path: str = "../api/models/generator.onnx",
        nz: int = 100):
    """
    Exports the trained generator model to ONNX format.

    Default outut path is ../api/models/generator.onnx, which is accessible by the API and 
    mounted to docker.

    Args:
        checkpoint_path (str): Path to the trained generator checkpoint.
        output_path (str): Path to save the exported ONNX model.
        nz (int): Dimension of the latent vector (input to generator).
    """

    print("=" * 60)
    print("Exporting Generator to ONNX format")
    print("=" * 60)

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok = True)
    print(f"\nOutput directory: {output_dir}")

    # Create model
    print(f"\n[1/5] Creating Generator model...")
    generator = Generator(nc = 3, nz=nz, ngf=64)

    # Load training weights
    print(f"[2/5] Loading weights from checkpoint: {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only = True)
    generator.load_state_dict(state_dict)
    print(f"    Loaded {len(state_dict)} parameter tensors")

    #CRITICAL: Set to eval mode for inference
    print(f"[3/5] Setting model to evaluation mode...")
    generator.eval()

    # Verify batchNorm is in eval mode
    for name, module in generator.named_modules():
        if isinstance(module, nn.BatchNorm2d):
           print(f"    BatchNorm layer '{name}' - training mode: {module.training}")

    # Create dummy input
    print(f"[4/5] Creating dummy input tensor...")
    dummy_input = torch.randn(1, nz, 1, 1)

    # Test the model first
    with torch.no_grad():
        test_output = generator(dummy_input)
        print(f"    Test inference successful - output shape: {test_output.shape}")
        print(f"    Output tensor stats - min: {test_output.min().item():.4f}, max: {test_output.max().item():.4f}, mean: {test_output.mean().item():.4f}")


    # Export to ONNX
    try:
        torch.onnx.export(
            generator,
            (dummy_input,),
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['latent_vector'],
            output_names=['generated_image'],
            dynamic_axes={'latent_vector': {0: 'batch_size'},    # Dynamic batch size
                          'generated_image': {0: 'batch_size'}
                          },
            dynamo = False,
            verbose = False
        )
    except TypeError:
        # Fallback for older pyTorch versions without fynamo parameter
        torch.onnx.export(
            generator,
            (dummy_input,),
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['latent_vector'],
            output_names=['generated_image'],
            dynamic_axes={
                'latent_vector': {0: 'batch_size'},
                'generated_image': {0: 'batch_size'}
            },
            training = torch.onnx.TrainingMode.EVAL,
            verbose = False 
        )

    # Validate
    print(f"[5/5] Validating ONNX Model")

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Test with ONNX runtime
    session = ort.InferenceSession(output_path, providers = ['CPUExecutionProvider'])
    test_input = np.random.randn(1, nz, 1, 1).astype(np.float32)
    onnx_output = session.run(None, {'latent_vector': test_input})[0]

    print(f"    ONNX output shape: {onnx_output.shape}")
    print(f"    ONNX output range: {onnx_output.min():.3f}, {onnx_output.max():.3f}")

    # check if output looks reasonable
    output_std = np.std(onnx_output)
    if output_std < 0.1:
        print(f"    WARNING: Output has low variance of {output_std:.4f} - May produce grayscale images!")
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print("ONNX model exported successfully")
    print(f"    Path: {output_path}")
    print(f"    Size: {file_size:.4f} MB")
    print(f"{'=' * 60}")

    return output_path    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exported trained generator to ONNX")
    parser.add_argument('--checkpoint', type = str, 
                        default='./checkpoints/generator_epoch_final.pth',
                        help='Path to the trained generator weights')
    parser.add_argument('--output', type=str, default='../api/models/generator.onnx',
                        help='Output ONNX path')
    
    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output)

    



    

