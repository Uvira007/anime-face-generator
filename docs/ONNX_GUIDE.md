# ONNX Export & Inference Guide

This guide provides comprehensive documentation on exporting the DCGAN Generator to ONNX format and using it for inference.

## Table of Contents

1. [What is ONNX?](#what-is-onnx)
2. [Why Use ONNX?](#why-use-onnx)
3. [Exporting the Model](#exporting-the-model)
4. [Model Specifications](#model-specifications)
5. [Inference with ONNX Runtime](#inference-with-onnx-runtime)
6. [Optimization Techniques](#optimization-techniques)
7. [Troubleshooting](#troubleshooting)

---

## What is ONNX?

**ONNX (Open Neural Network Exchange)** is an open-source format for representing machine learning models. It defines a common set of operators and a standard file format, enabling models to be transferred between different frameworks and tools.

### Key Features

- **Interoperability**: Train in PyTorch, deploy with ONNX Runtime, TensorFlow, or other frameworks
- **Optimization**: ONNX Runtime applies graph-level optimizations automatically
- **Hardware Acceleration**: Supports CPU, GPU (CUDA, DirectML), and specialized accelerators
- **Cross-Platform**: Works on Windows, Linux, macOS, iOS, Android, and web browsers

---

## Why Use ONNX?

| Benefit | Description |
|---------|-------------|
| **Faster Inference** | ONNX Runtime is highly optimized, often 2-3x faster than native PyTorch |
| **Smaller Deployment** | No need to ship PyTorch with your application |
| **Framework Freedom** | Train with any framework, deploy with ONNX |
| **Production Ready** | Battle-tested by Microsoft, Facebook, Amazon, and others |

### Performance Comparison

```
PyTorch (CPU):     ~45ms per image
ONNX Runtime (CPU): ~15ms per image  (3x faster)
ONNX Runtime (GPU): ~2ms per image   (20x faster)
```

---

## Exporting the Model

### Prerequisites

```bash
pip install torch onnx onnxruntime
```

### Export Script

```python
import torch
import onnx
from dcgan import Generator

# Load trained generator
generator = Generator(nc=3, nz=100, ngf=64)
generator.load_state_dict(torch.load('generator.pth', map_location='cpu'))
generator.eval()

# Create dummy input
dummy_input = torch.randn(1, 100, 1, 1)

# Export to ONNX
torch.onnx.export(
    generator,
    dummy_input,
    'generator.onnx',
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['latent_vector'],
    output_names=['generated_image'],
    dynamic_axes={
        'latent_vector': {0: 'batch_size'},
        'generated_image': {0: 'batch_size'}
    }
)

# Validate
onnx_model = onnx.load('generator.onnx')
onnx.checker.check_model(onnx_model)
print("✓ Model exported successfully!")
```

### Using the Export Script

```bash
cd model
python export_onnx.py \
    --checkpoint ./checkpoints/generator_final.pth \
    --output ./exports/generator.onnx \
    --validate \
    --benchmark
```

---

## Model Specifications

### Input Tensor

| Property | Value |
|----------|-------|
| Name | `latent_vector` |
| Shape | `(batch_size, 100, 1, 1)` |
| Data Type | `float32` |
| Value Range | Normal distribution (mean=0, std=1) |

### Output Tensor

| Property | Value |
|----------|-------|
| Name | `generated_image` |
| Shape | `(batch_size, 3, 64, 64)` |
| Data Type | `float32` |
| Value Range | `[-1, 1]` (normalized) |

### Post-Processing

To convert output to displayable RGB images:

```python
# Output is in [-1, 1] range, shape (N, C, H, W)
images = (output + 1) / 2.0  # Convert to [0, 1]
images = np.clip(images, 0, 1)
images = (images * 255).astype(np.uint8)
images = np.transpose(images, (0, 2, 3, 1))  # NCHW -> NHWC
```

---

## Inference with ONNX Runtime

### Python Example

```python
import numpy as np
import onnxruntime as ort
from PIL import Image

# Create inference session
session = ort.InferenceSession(
    'generator.onnx',
    providers=['CPUExecutionProvider']
)

# Generate random latent vector
z = np.random.randn(1, 100, 1, 1).astype(np.float32)

# Run inference
output = session.run(None, {'latent_vector': z})[0]

# Post-process
image = (output[0] + 1) / 2.0  # [-1, 1] -> [0, 1]
image = np.clip(image, 0, 1)
image = (image * 255).astype(np.uint8)
image = np.transpose(image, (1, 2, 0))  # CHW -> HWC

# Save
Image.fromarray(image).save('generated.png')
```

### Batch Generation

```python
# Generate 16 images at once
batch_size = 16
z = np.random.randn(batch_size, 100, 1, 1).astype(np.float32)
outputs = session.run(None, {'latent_vector': z})[0]
```

### With GPU Acceleration

```python
# Install: pip install onnxruntime-gpu

session = ort.InferenceSession(
    'generator.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# Check which provider is being used
print(session.get_providers())
```

---

## Optimization Techniques

### 1. Graph Optimization

ONNX Runtime automatically applies optimizations:

```python
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession('generator.onnx', sess_options)
```

### 2. Quantization (Optional)

Reduce model size with minimal accuracy loss:

```python
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    model_input='generator.onnx',
    model_output='generator_quantized.onnx',
    weight_type=QuantType.QUInt8
)
```

### 3. Threading Configuration

```python
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4  # Parallel ops within nodes
sess_options.inter_op_num_threads = 2  # Parallel execution of nodes
```

### 4. Memory Optimization

```python
sess_options = ort.SessionOptions()
sess_options.enable_mem_pattern = True
sess_options.enable_cpu_mem_arena = True
```

---

## Using with FastAPI

The API uses ONNX Runtime for inference:

```python
# api/inference.py
import onnxruntime as ort

class AnimeFaceGenerator:
    def __init__(self, model_path: str, nz: int = 100):
        self.nz = nz
        
        # Optimized session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
    
    def generate(self, num_images: int = 1) -> np.ndarray:
        z = np.random.randn(num_images, self.nz, 1, 1).astype(np.float32)
        outputs = self.session.run(None, {'latent_vector': z})[0]
        return self._postprocess(outputs)
```

---

## Troubleshooting

### Common Issues

#### 1. Model Validation Failed

```
onnx.checker.ValidationError: Node ...
```

**Solution**: Ensure opset version is compatible (use 14+):

```python
torch.onnx.export(..., opset_version=14)
```

#### 2. Shape Mismatch

```
InvalidArgument: Got invalid dimensions for input
```

**Solution**: Check input shape is exactly `(N, 100, 1, 1)`:

```python
z = z.reshape(-1, 100, 1, 1).astype(np.float32)
```

#### 3. Missing Provider

```
EP Error: CUDA provider is not enabled
```

**Solution**: Install GPU version:

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

#### 4. Slow Performance

**Solutions**:
- Enable graph optimization
- Use GPU provider if available
- Batch multiple requests together
- Check CPU/memory utilization

---

## File Sizes

| Model Variant | Size |
|---------------|------|
| PyTorch (.pth) | ~13 MB |
| ONNX (.onnx) | ~13 MB |
| ONNX Optimized | ~12 MB |
| ONNX Quantized | ~4 MB |

---

## Resources

- [ONNX Official Documentation](https://onnx.ai/onnx/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
- [ONNX Model Zoo](https://github.com/onnx/models)

---

## Summary

1. **Export**: Use `torch.onnx.export()` with opset 14+
2. **Validate**: Run `onnx.checker.check_model()`
3. **Optimize**: Enable graph optimizations in ONNX Runtime
4. **Deploy**: Use `onnxruntime.InferenceSession()` for inference

The ONNX format provides a reliable, optimized way to deploy your DCGAN model in production environments.

