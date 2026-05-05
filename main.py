import asyncio
import io
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from fastembed import ImageEmbedding, TextEmbedding

from config import logger

IMAGE_MODEL = os.getenv("IMAGE_MODEL", "Qdrant/clip-ViT-B-32-vision")
TEXT_MODEL = os.getenv("TEXT_MODEL", "Qdrant/clip-ViT-B-32-text")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
image_model = None
text_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global image_model, text_model
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, _load_models)
    yield
    executor.shutdown(wait=True)


def _load_models():
    global image_model, text_model
    try:
        logger.info(f"Loading image model: {IMAGE_MODEL}")
        image_model = ImageEmbedding(model_name=IMAGE_MODEL)
        logger.info(f"Loading text model: {TEXT_MODEL}")
        text_model = TextEmbedding(model_name=TEXT_MODEL)
        logger.info("FastEmbed models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load FastEmbed models: %s", e)
        raise


def _norm(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    return v / norm


def _embed_images_sync(images: list):
    if image_model is None:
        raise RuntimeError("Image model not loaded")
    emb = list(image_model.embed(images))
    return [_norm(e).tolist() for e in emb]


def _embed_texts_sync(texts: list[str]):
    if text_model is None:
        raise RuntimeError("Text model not loaded")
    emb = list(text_model.embed(texts))
    return [_norm(e).tolist() for e in emb]


class TextRequest(BaseModel):
    text: str


class TextResponse(BaseModel):
    embedding: list[float]


class BatchTextRequest(BaseModel):
    texts: list[str]


class BatchTextResponse(BaseModel):
    embeddings: list[list[float]]


class ImageResponse(BaseModel):
    embedding: list[float]


class SimilarityRequest(BaseModel):
    query_embedding: list[float]
    embeddings: list[list[float]]


class SimilarityResponse(BaseModel):
    similarities: list[float]


app = FastAPI(title="Embedding API", lifespan=lifespan)


@app.post("/embed/text", response_model=TextResponse)
async def embed_text(req: TextRequest):
    if text_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(executor, _embed_texts_sync, [req.text])
        return TextResponse(embedding=embedding[0])
    except Exception as e:
        logger.error(f"Text embedding error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/text/batch", response_model=BatchTextResponse)
async def embed_text_batch(req: BatchTextRequest):
    if text_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.texts:
        return BatchTextResponse(embeddings=[])
    try:
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(executor, _embed_texts_sync, req.texts)
        return BatchTextResponse(embeddings=embeddings)
    except Exception as e:
        logger.error(f"Batch text embedding error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/image", response_model=ImageResponse)
async def embed_image(file: UploadFile = File(...)):
    if image_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(executor, _embed_images_sync, [image])
        return ImageResponse(embedding=embedding[0])
    except Exception as e:
        logger.error(f"Image embedding error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similarity", response_model=SimilarityResponse)
async def similarity(req: SimilarityRequest):
    try:
        query = np.array(req.query_embedding, dtype=np.float32)
        embeddings = np.array(req.embeddings, dtype=np.float32)
        similarities = np.dot(embeddings, query).tolist()
        return SimilarityResponse(similarities=similarities)
    except Exception as e:
        logger.error(f"Similarity error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    status = "healthy" if text_model and image_model else "loading"
    return {"status": status, "models_loaded": text_model is not None and image_model is not None}