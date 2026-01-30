# Training Guide

This guide provides comprehensive documentation on training the DCGAN model for anime face generation.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Dataset Setup](#dataset-setup)
3. [Local Training](#local-training)
4. [Azure Databricks Training](#azure-databricks-training)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Monitoring Training](#monitoring-training)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

| Environment | Minimum | Recommended |
|-------------|---------|-------------|
| Local (GPU) | 8GB VRAM | 16GB VRAM |
| Local (CPU) | 16GB RAM | 32GB RAM |
| Databricks | Standard_L8s_v2 | Standard_NC6s_v3 |

### Software Requirements

```bash
# Python 3.10+
python --version

# Install dependencies
pip install -r model/requirements.txt
```

---

## Dataset Setup

### Option 1: Kaggle API (Recommended)

```bash
# Install Kaggle CLI
pip install kaggle

# Configure credentials
# 1. Go to kaggle.com → Account → Create New API Token
# 2. Save kaggle.json to ~/.kaggle/kaggle.json (Linux/Mac)
#    or C:\Users\<username>\.kaggle\kaggle.json (Windows)

# Download dataset
kaggle datasets download -d splcher/animefacedataset -p ./data --unzip
```

### Option 2: Manual Download

1. Visit [Kaggle Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset)
2. Download the ZIP file (~340MB)
3. Extract to `./data/anime_faces/`

### Dataset Structure

```
data/
└── anime_faces/
    ├── 00001.jpg
    ├── 00002.jpg
    ├── ...
    └── 63565.jpg  (approximately)
```

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Images | ~63,000 |
| Image Format | JPG/PNG |
| Original Resolution | Various (64x64 to 512x512) |
| License | CC0 (Public Domain) |

---

## Local Training

### Quick Start

```bash
cd model

# Train with default settings
python train.py --data_dir ./data/anime_faces --epochs 100

# Train with custom settings
python train.py \
    --data_dir ./data/anime_faces \
    --epochs 200 \
    --batch_size 128 \
    --lr 0.0002 \
    --device cuda
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `./data/anime_faces` | Path to training images |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `128` | Training batch size |
| `--lr` | `0.0002` | Learning rate |
| `--nz` | `100` | Latent vector size |
| `--ngf` | `64` | Generator feature maps |
| `--ndf` | `64` | Discriminator feature maps |
| `--workers` | `4` | Data loading workers |
| `--device` | `cuda` | Training device (cuda/cpu) |
| `--seed` | `42` | Random seed |
| `--checkpoint` | `None` | Resume from checkpoint |

### Training Duration

| Configuration | Epochs | Approximate Time |
|---------------|--------|------------------|
| GPU (RTX 3080) | 100 | ~30 minutes |
| GPU (RTX 2060) | 100 | ~60 minutes |
| CPU (8 cores) | 100 | ~8 hours |

---

## Azure Databricks Training

### 1. Cluster Configuration

Create a cluster with the following specifications:

| Setting | Value |
|---------|-------|
| Runtime | 14.3 LTS (includes Apache Spark 3.5.0, Scala 2.12) |
| Worker Type | Standard_L8s_v2 (or GPU instance) |
| Driver Type | Same as worker |
| Workers | 1 (single-node for training) |

### 2. Upload Dataset

**Option A: Via Databricks UI**
1. Navigate to Data → Create Table → Upload Files
2. Upload the anime faces ZIP file
3. Extract to `/FileStore/anime_faces/`

**Option B: Via Azure Blob Storage**
```python
# Mount Azure Blob Storage
dbutils.fs.mount(
    source="wasbs://container@account.blob.core.windows.net/",
    mount_point="/mnt/anime-data",
    extra_configs={"fs.azure.account.key.account.blob.core.windows.net": "YOUR_KEY"}
)
```

### 3. Run Training Notebook

1. Import `model/notebooks/train_dcgan_databricks.py`
2. Attach to your cluster
3. Run all cells

### 4. Download Trained Model

```python
# Copy from DBFS to local
dbutils.fs.cp(
    "dbfs:/FileStore/dcgan_checkpoints/generator.onnx",
    "file:/tmp/generator.onnx"
)

# Or download via Databricks UI:
# Data → FileStore → dcgan_checkpoints → generator.onnx
```

---

## Hyperparameter Tuning

### Key Hyperparameters

| Parameter | Range | Notes |
|-----------|-------|-------|
| `batch_size` | 32-256 | Larger = more stable, needs more VRAM |
| `lr` | 0.0001-0.0003 | Too high = mode collapse |
| `beta1` | 0.0-0.9 | DCGAN paper uses 0.5 |
| `nz` | 50-200 | Latent space dimensionality |
| `ngf/ndf` | 32-128 | Model capacity |

### Recommended Configurations

**Faster Training (Lower Quality)**
```python
batch_size = 256
lr = 0.0003
num_epochs = 50
ngf = 32
ndf = 32
```

**Better Quality (Slower)**
```python
batch_size = 64
lr = 0.0001
num_epochs = 200
ngf = 128
ndf = 128
```

### Training Tips

1. **Monitor D(x) and D(G(z))**
   - D(x) should hover around 0.7-0.9
   - D(G(z)) should hover around 0.3-0.5
   - If D(x) → 1.0 and D(G(z)) → 0.0: discriminator winning (mode collapse)

2. **Label Smoothing** (optional)
   ```python
   real_label = 0.9  # Instead of 1.0
   fake_label = 0.1  # Instead of 0.0
   ```

3. **Feature Matching** (advanced)
   - Match intermediate layer activations instead of output

---

## Monitoring Training

### Training Outputs

```
./checkpoints/
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_20.pth
├── ...
├── generator_epoch_final.pth
└── checkpoint_epoch_final.pth

./outputs/
├── samples_epoch_5.png
├── samples_epoch_10.png
├── ...
├── training_curves.png
└── training_history.json
```

### Sample Images

During training, sample images are saved every N epochs:

```
samples_epoch_5.png   - Early training (noisy)
samples_epoch_20.png  - Starting to form faces
samples_epoch_50.png  - Recognizable faces
samples_epoch_100.png - Final quality
```

### Training Curves

The `training_curves.png` shows:
- Generator loss over time
- Discriminator loss over time
- D(x) and D(G(z)) values

**Healthy Training Signs:**
- Losses oscillate but trend downward
- D(x) ≈ 0.7-0.8, D(G(z)) ≈ 0.3-0.4
- Sample images improve progressively

### TensorBoard (Optional)

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/dcgan')

# In training loop
writer.add_scalar('Loss/Generator', loss_G, iteration)
writer.add_scalar('Loss/Discriminator', loss_D, iteration)
writer.add_images('Generated', fake_images, iteration)
```

```bash
tensorboard --logdir runs/
```

---

## Troubleshooting

### Issue 1: Mode Collapse

**Symptoms:**
- Generator produces same/similar images
- D(G(z)) → 0 consistently
- Generator loss increases

**Solutions:**
1. Decrease learning rate
2. Add noise to discriminator inputs
3. Use label smoothing
4. Increase batch size

### Issue 2: Discriminator Wins

**Symptoms:**
- D(x) → 1.0, D(G(z)) → 0.0
- Generator loss explodes
- No improvement in samples

**Solutions:**
1. Train generator more (2 G updates per D update)
2. Use one-sided label smoothing
3. Decrease discriminator learning rate

### Issue 3: Training Instability

**Symptoms:**
- Wild loss oscillations
- NaN/Inf values
- Gradient explosion

**Solutions:**
1. Decrease learning rate
2. Add gradient clipping
3. Check for data preprocessing issues
4. Ensure correct normalization (-1 to 1)

### Issue 4: Out of Memory

**Symptoms:**
- CUDA OOM errors
- Training crashes

**Solutions:**
1. Reduce batch size
2. Reduce image size
3. Use gradient accumulation
4. Use mixed precision (fp16)

### Issue 5: Slow Training

**Solutions:**
1. Increase batch size
2. Reduce num_workers if I/O bound
3. Use pin_memory=True
4. Use GPU if available

---

## After Training

### Export to ONNX

```bash
python export_onnx.py \
    --checkpoint ./checkpoints/generator_epoch_final.pth \
    --output ./exports/generator.onnx \
    --validate \
    --benchmark
```

### Test Generation

```python
from dcgan import Generator

generator = Generator(nc=3, nz=100, ngf=64)
generator.load_state_dict(torch.load('generator_final.pth'))
generator.eval()

# Generate images
z = torch.randn(16, 100, 1, 1)
with torch.no_grad():
    fake = generator(z)
```

---

## Summary

1. **Setup**: Download dataset, install dependencies
2. **Configure**: Adjust hyperparameters as needed
3. **Train**: Run training script or notebook
4. **Monitor**: Check samples and loss curves
5. **Export**: Convert to ONNX for deployment

Happy training! 🎨

