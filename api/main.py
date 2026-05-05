import asyncio
import io
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from fastembed import ImageEmbedding, TextEmbedding

IMAGE_MODEL = os.getenv("IMAGE_MODEL", "Qdrant/clip-ViT-B-32-vision")
TEXT_MODEL = os.getenv("TEXT_MODEL", "Qdrant/clip-ViT-B-32-text")

executor = ThreadPoolExecutor(max_workers=2)
image_model = None
text_model = None


def init_models():
    global image_model, text_model
    try:
        image_model = ImageEmbedding(model_name=IMAGE_MODEL)
        text_model = TextEmbedding(model_name=TEXT_MODEL)
    except Exception:
        pass


init_models()


def _norm(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    return v / norm


def _embed_text_sync(texts):
    if text_model is None:
        raise RuntimeError("Model not loaded")
    emb = list(text_model.embed(texts))
    return [_norm(e).tolist() for e in emb]


app = FastAPI()


@app.get("/health")
def health():
    return JSONResponse({"status": "healthy" if text_model else "loading"})


@app.post("/embed/text")
def embed_text(req: dict):
    try:
        text = req.get("text", "")
        emb = _embed_text_sync([text])
        return JSONResponse({"embedding": emb[0]})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/similarity")
def similarity(req: dict):
    try:
        query = np.array(req.get("query_embedding", []), dtype=np.float32)
        embeddings = np.array(req.get("embeddings", []), dtype=np.float32)
        sims = np.dot(embeddings, query).tolist()
        return JSONResponse({"similarities": sims})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)