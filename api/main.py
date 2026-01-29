"""
FastAPI Backend for Anime Face Generator.

This module provides a REST API for generating anime faces using
the ONNX-exported DCGAN Generator model.

Endpoints:
- GET /: API information and health check
- POST /generate: Generate anime face images
- POST /interpolate: Interpolate between two faces
- GET /health: Health check endpoint

Author: [Your Name]
License: MIT
"""

import os
import time
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import numpy as np
from io import BytesIO

from inference import AnimeFaceGenerator, load_generator


# ==================== Configuration ====================

class Settings:
    """Application settings."""
    
    # Model settings
    MODEL_PATH: str = os.getenv("MODEL_PATH", "./models/generator_50epoch.onnx")
    LATENT_DIM: int = int(os.getenv("LATENT_DIM", "100"))
    
    # API settings
    MAX_IMAGES_PER_REQUEST: int = int(os.getenv("MAX_IMAGES_PER_REQUEST", "16"))
    MAX_INTERPOLATION_STEPS: int = int(os.getenv("MAX_INTERPOLATION_STEPS", "20"))
    
    # CORS settings
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS", 
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000"
    ).split(",")


settings = Settings()


# ==================== Pydantic Models ====================

class GenerateRequest(BaseModel):
    """Request model for image generation."""
    
    num_images: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Number of images to generate (1-16)"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    format: str = Field(
        default="base64",
        pattern="^(base64|grid)$",
        description="Output format: 'base64' for individual images, 'grid' for combined grid"
    )


class GenerateResponse(BaseModel):
    """Response model for image generation."""
    
    success: bool
    num_images: int
    seed: Optional[int]
    images: Optional[List[str]] = None  # Base64 encoded images
    grid: Optional[str] = None  # Base64 encoded grid image
    generation_time_ms: float


class InterpolateRequest(BaseModel):
    """Request model for latent space interpolation."""
    
    seed1: int = Field(
        description="Random seed for the first face"
    )
    seed2: int = Field(
        description="Random seed for the second face"
    )
    num_steps: int = Field(
        default=10,
        ge=2,
        le=20,
        description="Number of interpolation steps (2-20)"
    )


class InterpolateResponse(BaseModel):
    """Response model for interpolation."""
    
    success: bool
    num_steps: int
    images: List[str]  # Base64 encoded images
    generation_time_ms: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str
    model_loaded: bool
    model_path: str
    latent_dim: int
    version: str = "1.0.0"


class APIInfo(BaseModel):
    """API information model."""
    
    name: str = "Anime Face Generator API"
    version: str = "1.0.0"
    description: str = "Generate anime character faces using DCGAN"
    endpoints: dict


# ==================== Global State ====================

generator: Optional[AnimeFaceGenerator] = None


# ==================== Lifespan Management ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    
    Loads the ONNX model on startup and cleans up on shutdown.
    """
    global generator
    
    # Startup
    print("[API] Starting Anime Face Generator API...")
    print(f"[API] Loading model from: {settings.MODEL_PATH}")
    
    try:
        generator = load_generator(
            model_path=settings.MODEL_PATH,
            nz=settings.LATENT_DIM
        )
        print("[API] Model loaded successfully!")
    except Exception as e:
        print(f"[API] Error loading model: {e}")
        print("[API] API will start but model endpoints will fail")
    
    yield
    
    # Shutdown
    print("[API] Shutting down...")
    generator = None


# ==================== FastAPI App ====================

app = FastAPI(
    title="Anime Face Generator API",
    description="""
    Generate anime character faces using a Deep Convolutional 
    Generative Adversarial Network (DCGAN).
    
    ## Features
    - Generate single or multiple anime face images
    - Reproducible generation with seed control
    - Latent space interpolation between faces
    - Output as base64 images or combined grid
    
    ## Model
    This API uses an ONNX-optimized DCGAN Generator trained on 
    the Anime Face Dataset (~63,000 images, CC0 license).
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Helper Functions ====================

def ensure_model_loaded():
    """Raise exception if model is not loaded."""
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the ONNX model file exists."
        )


# ==================== API Endpoints ====================

@app.get("/", response_model=APIInfo)
async def root():
    """
    Get API information and available endpoints.
    """
    return APIInfo(
        endpoints={
            "GET /": "API information (this endpoint)",
            "GET /health": "Health check and model status",
            "POST /generate": "Generate anime face images",
            "POST /interpolate": "Interpolate between two faces",
            "GET /docs": "Interactive API documentation (Swagger)",
            "GET /redoc": "Alternative API documentation (ReDoc)"
        }
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check API health and model status.
    """
    return HealthResponse(
        status="healthy" if generator is not None else "degraded",
        model_loaded=generator is not None,
        model_path=settings.MODEL_PATH,
        latent_dim=settings.LATENT_DIM
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_images(request: GenerateRequest):
    """
    Generate anime face images.
    
    **Parameters:**
    - `num_images`: Number of images to generate (1-16)
    - `seed`: Optional random seed for reproducibility
    - `format`: Output format - 'base64' for individual images, 'grid' for combined
    
    **Returns:**
    - Base64-encoded PNG images or a grid image
    
    **Example Request:**
    ```json
    {
        "num_images": 4,
        "seed": 42,
        "format": "base64"
    }
    ```
    """
    ensure_model_loaded()
    
    start_time = time.time()
    
    try:
        # Generate images
        images = generator.generate(
            num_images=request.num_images,
            seed=request.seed
        )
        
        generation_time = (time.time() - start_time) * 1000
        
        if request.format == "grid":
            # Return as grid image
            grid = generator.create_grid(images, nrow=4)
            buffer = BytesIO()
            grid.save(buffer, format='PNG')
            grid_base64 = __import__('base64').b64encode(
                buffer.getvalue()
            ).decode('utf-8')
            
            return GenerateResponse(
                success=True,
                num_images=request.num_images,
                seed=request.seed,
                grid=f"data:image/png;base64,{grid_base64}",
                generation_time_ms=generation_time
            )
        else:
            # Return as individual base64 images
            base64_images = generator.to_base64(images)
            base64_images = [
                f"data:image/png;base64,{img}" 
                for img in base64_images
            ]
            
            return GenerateResponse(
                success=True,
                num_images=request.num_images,
                seed=request.seed,
                images=base64_images,
                generation_time_ms=generation_time
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating images: {str(e)}"
        )


@app.post("/interpolate", response_model=InterpolateResponse)
async def interpolate_faces(request: InterpolateRequest):
    """
    Interpolate between two anime faces in latent space.
    
    Creates a smooth transition between two faces by linearly
    interpolating their latent vectors.
    
    **Parameters:**
    - `seed1`: Random seed for the first face
    - `seed2`: Random seed for the second face
    - `num_steps`: Number of interpolation steps (2-20)
    
    **Returns:**
    - Array of base64-encoded images showing the transition
    
    **Example Request:**
    ```json
    {
        "seed1": 42,
        "seed2": 123,
        "num_steps": 10
    }
    ```
    """
    ensure_model_loaded()
    
    start_time = time.time()
    
    try:
        # Generate latent vectors from seeds
        np.random.seed(request.seed1)
        z1 = np.random.randn(generator.nz).astype(np.float32)
        
        np.random.seed(request.seed2)
        z2 = np.random.randn(generator.nz).astype(np.float32)
        
        # Interpolate
        images = generator.interpolate(z1, z2, num_steps=request.num_steps)
        
        generation_time = (time.time() - start_time) * 1000
        
        # Convert to base64
        base64_images = generator.to_base64(images)
        base64_images = [
            f"data:image/png;base64,{img}" 
            for img in base64_images
        ]
        
        return InterpolateResponse(
            success=True,
            num_steps=request.num_steps,
            images=base64_images,
            generation_time_ms=generation_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during interpolation: {str(e)}"
        )


@app.get("/random")
async def get_random_image(seed: Optional[int] = Query(None)):
    """
    Get a single random anime face as a PNG image.
    
    This endpoint returns the raw PNG image directly,
    useful for embedding in img tags or downloading.
    
    **Parameters:**
    - `seed`: Optional random seed for reproducibility
    """
    ensure_model_loaded()
    
    try:
        images = generator.generate(num_images=1, seed=seed)
        pil_image = generator.to_pil(images)[0]
        
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={
                "Content-Disposition": "inline; filename=anime_face.png"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating image: {str(e)}"
        )


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )