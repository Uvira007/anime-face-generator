# Docker Deployment Guide

This guide provides comprehensive documentation on deploying the Anime Face Generator using Docker containers.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Architecture Overview](#architecture-overview)
4. [Building Images](#building-images)
5. [Running Containers](#running-containers)
6. [Configuration](#configuration)
7. [Production Deployment](#production-deployment)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Docker | 20.10+ | Container runtime |
| Docker Compose | 2.0+ | Multi-container orchestration |

### Installation

**Windows:**
Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

**Linux (Ubuntu/Debian):**
```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt-get install docker-compose-plugin
```

**macOS:**
Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Verify Installation

```bash
docker --version
docker compose version
```

---

## Quick Start

### 1. Prepare the Model

Before running, ensure you have the trained ONNX model:

```bash
# Create models directory
mkdir -p api/models

# Copy your trained model
cp /path/to/generator.onnx api/models/
```

### 2. Build and Run

```bash
# Build all images
docker compose build

# Start all services
docker compose up -d

# View logs
docker compose logs -f
```

### 3. Access the Application

- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐         ┌──────────────────────────┐     │
│   │   Frontend   │         │      API Backend         │     │
│   │  (Nginx)     │ ──────► │      (FastAPI)           │     │
│   │  Port: 3000  │         │      Port: 8000          │     │
│   └──────────────┘         └──────────────────────────┘     │
│         │                           │                        │
│         │                           ▼                        │
│         │                  ┌──────────────────────┐         │
│         │                  │  ONNX Model Volume   │         │
│         │                  │  /app/models/        │         │
│         │                  │  generator.onnx      │         │
│         │                  └──────────────────────┘         │
│         │                                                    │
│         ▼                                                    │
│   ┌──────────────────────────────────────────────────┐      │
│   │              anime-face-network                   │      │
│   │              (Docker Bridge Network)              │      │
│   └──────────────────────────────────────────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Building Images

### Build All Services

```bash
docker compose build
```

### Build Individual Services

```bash
# Build API only
docker compose build api

# Build frontend only
docker compose build frontend
```

### Build with No Cache

```bash
docker compose build --no-cache
```

### View Image Sizes

```bash
docker images | grep anime-face
```

Expected sizes:
- `anime-face-api`: ~500MB
- `anime-face-frontend`: ~25MB

---

## Running Containers

### Start All Services

```bash
# Start in detached mode
docker compose up -d

# Start with build
docker compose up -d --build

# Start specific service
docker compose up -d api
```

### View Running Containers

```bash
docker compose ps
```

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f api

# Last 100 lines
docker compose logs --tail 100
```

### Stop Services

```bash
# Stop all
docker compose down

# Stop and remove volumes
docker compose down -v

# Stop specific service
docker compose stop frontend
```

### Restart Services

```bash
docker compose restart
docker compose restart api
```

---

## Configuration

### Environment Variables

#### API Service

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/app/models/generator.onnx` | Path to ONNX model |
| `LATENT_DIM` | `100` | Latent vector dimension |
| `CORS_ORIGINS` | `http://localhost:3000` | Allowed CORS origins |
| `PORT` | `8000` | API port |

#### Frontend Service

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_URL` | `http://localhost:8000` | Backend API URL |

### Override Configuration

Create a `docker-compose.override.yml`:

```yaml
version: '3.8'

services:
  api:
    environment:
      - LATENT_DIM=128
      - CORS_ORIGINS=*
    ports:
      - "9000:8000"  # Different port
  
  frontend:
    ports:
      - "8080:80"  # Different port
```

### Volume Mounts

```yaml
services:
  api:
    volumes:
      # Read-only model mount
      - ./api/models:/app/models:ro
      
      # For development - mount source code
      - ./api:/app:ro
```

---

## Production Deployment

### 1. Create Production Compose File

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  api:
    build:
      context: ./api
      dockerfile: Dockerfile
    restart: always
    environment:
      - MODEL_PATH=/app/models/generator.onnx
      - CORS_ORIGINS=${ALLOWED_ORIGINS}
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    restart: always
    depends_on:
      api:
        condition: service_healthy
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 256M
```

### 2. Use with Reverse Proxy (Nginx/Traefik)

Example with Traefik:

```yaml
services:
  api:
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.api.rule=Host(`api.yourdomain.com`)"
      - "traefik.http.services.api.loadbalancer.server.port=8000"

  frontend:
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.frontend.rule=Host(`yourdomain.com`)"
      - "traefik.http.services.frontend.loadbalancer.server.port=80"
```

### 3. Deploy

```bash
# Using production compose file
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## GPU Support (Optional)

### Prerequisites

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Update Dockerfile

```dockerfile
# api/Dockerfile.gpu
FROM python:3.10-slim

# Install ONNX Runtime GPU
RUN pip install onnxruntime-gpu
```

### Update Compose File

```yaml
services:
  api:
    build:
      dockerfile: Dockerfile.gpu
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Troubleshooting

### Common Issues

#### 1. Container Won't Start

```bash
# Check logs
docker compose logs api

# Common causes:
# - Model file not found
# - Port already in use
# - Memory limit too low
```

#### 2. Model Not Found

```
Error: Model file not found at /app/models/generator.onnx
```

**Solution:**
```bash
# Ensure model exists
ls -la api/models/

# Verify volume mount
docker compose exec api ls -la /app/models/
```

#### 3. CORS Errors

```
Access-Control-Allow-Origin header missing
```

**Solution:**
```yaml
environment:
  - CORS_ORIGINS=http://localhost:3000,http://your-frontend-url
```

#### 4. Out of Memory

```
Container killed due to OOM
```

**Solution:**
```yaml
deploy:
  resources:
    limits:
      memory: 4G
```

#### 5. Port Conflict

```
bind: address already in use
```

**Solution:**
```bash
# Find process using port
lsof -i :8000

# Or change port in docker-compose.yml
ports:
  - "9000:8000"
```

### Useful Commands

```bash
# Enter container shell
docker compose exec api /bin/bash

# Check container resource usage
docker stats

# Clean up unused images
docker image prune

# Clean up everything
docker system prune -a

# View container details
docker inspect anime-face-api
```

---

## Health Checks

The stack includes built-in health checks:

### API Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "/app/models/generator.onnx",
  "latent_dim": 100
}
```

### Frontend Health Check

```bash
curl http://localhost:3000/
```

---

## Summary

| Action | Command |
|--------|---------|
| Build | `docker compose build` |
| Start | `docker compose up -d` |
| Stop | `docker compose down` |
| Logs | `docker compose logs -f` |
| Restart | `docker compose restart` |
| Status | `docker compose ps` |
| Shell | `docker compose exec api /bin/bash` |

The Docker setup provides an easy, reproducible way to deploy the Anime Face Generator in any environment.

