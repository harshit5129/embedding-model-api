import hashlib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()


def _hash_to_embedding(text: str, dim: int = 512) -> list:
    hash_bytes = hashlib.sha256(text.encode()).digest()
    arr = np.frombuffer(hash_bytes * (dim // 32 + 1), dtype=np.uint8)[:dim]
    arr = arr.astype(np.float32)
    arr = arr / 255.0 * 2 - 1
    return arr.tolist()


@app.get("/health")
def health():
    return JSONResponse({"status": "healthy", "model": "hash-embedding"})


@app.post("/embed/text")
def embed_text(req: dict):
    text = req.get("text", "")
    embedding = _hash_to_embedding(text)
    return JSONResponse({"embedding": embedding})


@app.post("/similarity")
def similarity(req: dict):
    query = np.array(req.get("query_embedding", []), dtype=np.float32)
    embeddings = np.array(req.get("embeddings", []), dtype=np.float32)
    sims = np.dot(embeddings, query).tolist()
    return JSONResponse({"similarities": sims})


@app.get("/")
def root():
    return JSONResponse({"message": "Embedding API", "endpoints": ["/health", "/embed/text", "/similarity"]})