# 🎨 Anime Face Generator

<div align="center">

![DCGAN](https://img.shields.io/badge/Model-DCGAN-ff69b4?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=for-the-badge&logo=onnx)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi)
![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Generate unique anime character faces using Deep Convolutional Generative Adversarial Networks**

[Demo](#demo) • [Features](#features) • [Quick Start](#quick-start) • [Documentation](#documentation)

</div>

---

## 📖 Overview

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate unique anime character faces. The model is trained on the [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset) containing ~63,000 images under CC0 license.

### What is a DCGAN?

DCGANs are a class of neural networks that learn to generate realistic images by training two networks simultaneously:
- **Generator**: Creates fake images from random noise
- **Discriminator**: Tries to distinguish real images from fakes

Through adversarial training, the generator learns to produce increasingly realistic images.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎨 **Image Generation** | Generate unique 64x64 anime faces from random latent vectors |
| 🔄 **Latent Interpolation** | Morph smoothly between two faces |
| 🚀 **ONNX Optimized** | Fast inference using ONNX Runtime |
| 🌐 **REST API** | FastAPI backend with interactive docs |
| 💻 **Modern UI** | Beautiful React frontend with animations |
| 🐳 **Docker Ready** | One-command deployment with Docker Compose |
| 📊 **Databricks Support** | Train on Azure Databricks clusters |

---

## 🖼️ Demo

### Generated Samples

<div align="center">
<i>After 100 epochs of training, the generator produces anime faces like these:</i>

![Generated anime face (50 epochs)](docs/images/generated_anime_image.png)
</div>

### Latent Space Interpolation

<div align="center">
<i>Smooth transition between two faces by interpolating in latent space:</i>

![Face morphing sequence](docs/images/interpolation_sequence.png)
</div>

---

## 🏗️ Project Structure

```
anime-face-generator/
├── 📁 model/                    # Training & model code
│   ├── config.py                # Configuration classes
│   ├── dataset.py               # Dataset utilities
│   ├── dcgan.py                 # Generator & Discriminator
│   ├── train.py                 # Training script
│   ├── export_onnx.py           # ONNX export utilities
│   ├── requirements.txt
│   └── 📁 notebooks/
│       └── train_dcgan_databricks.py
│
├── 📁 api/                      # FastAPI backend
│   ├── main.py                  # API endpoints
│   ├── inference.py             # ONNX inference
│   ├── requirements.txt
│   └── Dockerfile
│
├── 📁 frontend/                 # React frontend
│   ├── src/
│   │   ├── App.jsx
│   │   └── components/
│   ├── package.json
│   └── Dockerfile
│
├── 📁 docs/                     # Documentation
│   ├── TRAINING_GUIDE.md
│   ├── ONNX_GUIDE.md
│   └── DOCKER_GUIDE.md
│
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start

### Option 1: Using Pre-trained Model (Fastest)

1. **Download the pre-trained model** (after you train it)
2. **Run with Docker:**

```bash
# Clone the repository
git clone https://github.com/yourusername/anime-face-generator.git
cd anime-face-generator

# Place your trained model
mkdir -p api/models
cp /path/to/generator.onnx api/models/

# Start the application
docker compose up -d

# Open http://localhost:3000
```

### Option 2: Train Your Own Model

```bash
# 1. Install dependencies
cd model
pip install -r requirements.txt

# 2. Download dataset (requires Kaggle API)
kaggle datasets download -d splcher/animefacedataset -p ./data --unzip

# 3. Train the model
python train.py --data_dir ./data/anime_faces --epochs 100

# 4. Export to ONNX
python export_onnx.py --checkpoint ./checkpoints/generator_final.pth

# 5. Start the application
cd ..
docker compose up -d
```

---

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| [Training Guide](docs/TRAINING_GUIDE.md) | How to train the model locally or on Databricks |
| [ONNX Guide](docs/ONNX_GUIDE.md) | Exporting and optimizing the model |
| [Docker Guide](docs/DOCKER_GUIDE.md) | Deployment with Docker |

---

## 🔧 API Reference

### Generate Images

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"num_images": 4, "seed": 42}'
```

### Interpolate Faces

```bash
curl -X POST "http://localhost:8000/interpolate" \
  -H "Content-Type: application/json" \
  -d '{"seed1": 42, "seed2": 123, "num_steps": 10}'
```

### Get Random Image

```bash
curl "http://localhost:8000/random?seed=42" --output anime_face.png
```

**Interactive API Docs:** http://localhost:8000/docs

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|------------|
| **Model Training** | PyTorch 2.0+ |
| **Model Inference** | ONNX Runtime |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | React 18 + Tailwind CSS + Framer Motion |
| **Deployment** | Docker + Docker Compose |
| **Training Platform** | Azure Databricks (optional) |

---

## 📊 Model Architecture

### Generator

```
Input: (batch, 100, 1, 1) - Random latent vector

Layer 1: ConvTranspose2d(100, 512, 4, 1, 0) → BatchNorm → ReLU  → (batch, 512, 4, 4)
Layer 2: ConvTranspose2d(512, 256, 4, 2, 1) → BatchNorm → ReLU  → (batch, 256, 8, 8)
Layer 3: ConvTranspose2d(256, 128, 4, 2, 1) → BatchNorm → ReLU  → (batch, 128, 16, 16)
Layer 4: ConvTranspose2d(128, 64, 4, 2, 1)  → BatchNorm → ReLU  → (batch, 64, 32, 32)
Layer 5: ConvTranspose2d(64, 3, 4, 2, 1)    → Tanh              → (batch, 3, 64, 64)

Output: (batch, 3, 64, 64) - RGB image in [-1, 1]
```

### Discriminator

```
Input: (batch, 3, 64, 64) - RGB image

Layer 1: Conv2d(3, 64, 4, 2, 1)    → LeakyReLU(0.2)  → (batch, 64, 32, 32)
Layer 2: Conv2d(64, 128, 4, 2, 1)  → BN → LeakyReLU  → (batch, 128, 16, 16)
Layer 3: Conv2d(128, 256, 4, 2, 1) → BN → LeakyReLU  → (batch, 256, 8, 8)
Layer 4: Conv2d(256, 512, 4, 2, 1) → BN → LeakyReLU  → (batch, 512, 4, 4)
Layer 5: Conv2d(512, 1, 4, 1, 0)   → Sigmoid         → (batch, 1, 1, 1)

Output: Probability that input is real [0, 1]
```

---

## 🎛️ Configuration

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_size` | 64 | Generated image size |
| `nz` | 100 | Latent vector dimension |
| `ngf` | 64 | Generator feature maps |
| `ndf` | 64 | Discriminator feature maps |
| `num_epochs` | 100 | Training epochs |
| `batch_size` | 128 | Training batch size |
| `lr` | 0.0002 | Learning rate |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./models/generator.onnx` | Path to ONNX model |
| `LATENT_DIM` | `100` | Latent vector size |
| `CORS_ORIGINS` | `http://localhost:3000` | Allowed CORS origins |

---

## 📈 Training Results

### Expected Metrics

| Epoch | D(x) | D(G(z)) | Sample Quality |
|-------|------|---------|----------------|
| 10 | ~0.9 | ~0.1 | Noisy blobs |
| 30 | ~0.8 | ~0.3 | Basic shapes |
| 60 | ~0.7 | ~0.4 | Recognizable faces |
| 100 | ~0.6 | ~0.5 | Good quality |

### Training Time

| Environment | 100 Epochs |
|-------------|------------|
| RTX 3080 | ~30 min |
| RTX 2060 | ~60 min |
| Azure Standard_NC6s_v3 | ~45 min |
| CPU (8 cores) | ~8 hours |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [DCGAN Paper](https://arxiv.org/abs/1511.06434) by Radford et al.
- [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset) (CC0 License)
- [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [ONNX Runtime](https://onnxruntime.ai/)

---

## 📞 Contact

Your Name - [uvira007](https://linkedin.com/uvira007)

Project Link: [https://github.com/uvira007/anime-face-generator](https://github.com/uvira007/anime-face-generator)

---

<div align="center">

**Built with 💜 for learning and portfolio demonstration**

⭐ Star this repo if you found it helpful!

</div>

