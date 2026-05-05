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


@app.get("/")
async def status_page():
    models_ready = text_model is not None and image_model is not None
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embedding API</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f0f0f; color: #fff; min-height: 100vh; }}
        .container {{ max-width: 900px; margin: 0 auto; padding: 2rem; }}
        h1 {{ font-size: 2rem; margin-bottom: 1rem; color: #00ff88; }}
        .status-bar {{ display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem; padding: 1rem; background: #1a1a1a; border-radius: 8px; }}
        .status-dot {{ width: 12px; height: 12px; border-radius: 50%; background: #ff4444; }}
        .status-dot.healthy {{ background: #00ff88; }}
        .section {{ background: #1a1a1a; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }}
        .section h2 {{ font-size: 1.2rem; margin-bottom: 1rem; color: #00ff88; }}
        .endpoint {{ display: flex; justify-content: space-between; align-items: center; padding: 1rem; background: #252525; border-radius: 8px; margin-bottom: 0.75rem; }}
        .endpoint-method {{ font-weight: bold; padding: 4px 12px; border-radius: 4px; font-size: 0.8rem; }}
        .method-get {{ background: #00ff88; color: #000; }}
        .method-post {{ background: #0088ff; color: #fff; }}
        .endpoint-path {{ color: #aaa; font-family: monospace; }}
        .test-area {{ margin-top: 1rem; }}
        textarea {{ width: 100%; padding: 0.75rem; background: #252525; border: 1px solid #333; border-radius: 8px; color: #fff; font-family: monospace; font-size: 0.9rem; resize: vertical; min-height: 80px; }}
        button {{ background: #00ff88; color: #000; border: none; padding: 0.75rem 1.5rem; border-radius: 8px; cursor: pointer; font-weight: bold; margin-top: 0.5rem; }}
        button:hover {{ opacity: 0.9; }}
        .response {{ margin-top: 1rem; padding: 1rem; background: #252525; border-radius: 8px; font-family: monospace; font-size: 0.85rem; max-height: 200px; overflow: auto; white-space: pre-wrap; word-break: break-all; }}
        .error {{ color: #ff4444; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Embedding API</h1>
        <div class="status-bar">
            <div class="status-dot" id="statusDot" class="{'healthy' if models_ready else ''}"></div>
            <span id="statusText">API Status: {'healthy' if models_ready else 'loading'}</span>
        </div>

        <div class="section">
            <h2>Embed Text</h2>
            <div class="endpoint">
                <span class="endpoint-method method-post">POST</span>
                <span class="endpoint-path">/embed/text</span>
            </div>
            <div class="test-area">
                <textarea id="textInput" placeholder='{{"text": "Hello world"}}'></textarea>
                <button onclick="testText()">Test</button>
                <div class="response" id="textResponse"></div>
            </div>
        </div>

        <div class="section">
            <h2>Similarity</h2>
            <div class="endpoint">
                <span class="endpoint-method method-post">POST</span>
                <span class="endpoint-path">/similarity</span>
            </div>
            <div class="test-area">
                <textarea id="simInput" placeholder='{{"query_embedding": [0.1, 0.2, ...], "embeddings": [[0.1, 0.2, ...], ...]}}'></textarea>
                <button onclick="testSimilarity()">Test</button>
                <div class="response" id="simResponse"></div>
            </div>
        </div>

        <div class="section">
            <h2>All Endpoints</h2>
            <div class="endpoint"><span class="endpoint-method method-get">GET</span><span class="endpoint-path">/health</span></div>
            <div class="endpoint"><span class="endpoint-method method-get">GET</span><span class="endpoint-path">/</span></div>
            <div class="endpoint"><span class="endpoint-method method-post">POST</span><span class="endpoint-path">/embed/text</span></div>
            <div class="endpoint"><span class="endpoint-method method-post">POST</span><span class="endpoint-path">/embed/text/batch</span></div>
            <div class="endpoint"><span class="endpoint-method method-post">POST</span><span class="endpoint-path">/embed/image</span></div>
            <div class="endpoint"><span class="endpoint-method method-post">POST</span><span class="endpoint-path">/similarity</span></div>
        </div>
    </div>

    <script>
        async function checkHealth() {{
            try {{
                const res = await fetch('/health');
                const data = await res.json();
                const dot = document.getElementById('statusDot');
                if (data.status === 'healthy') {{
                    dot.classList.add('healthy');
                    document.getElementById('statusText').textContent = 'API Status: healthy';
                }} else {{
                    document.getElementById('statusText').textContent = 'API Status: ' + data.status;
                }}
            }} catch (e) {{
                document.getElementById('statusText').textContent = 'API Status: offline';
            }}
        }}

        async function testText() {{
            const input = document.getElementById('textInput').value;
            const resp = document.getElementById('textResponse');
            try {{
                const data = JSON.parse(input);
                const res = await fetch('/embed/text', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify(data)
                }});
                const result = await res.json();
                resp.textContent = JSON.stringify(result, null, 2);
                resp.classList.remove('error');
            }} catch (e) {{
                resp.textContent = 'Error: ' + e.message;
                resp.classList.add('error');
            }}
        }}

        async function testSimilarity() {{
            const input = document.getElementById('simInput').value;
            const resp = document.getElementById('simResponse');
            try {{
                const data = JSON.parse(input);
                const res = await fetch('/similarity', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify(data)
                }});
                const result = await res.json();
                resp.textContent = JSON.stringify(result, null, 2);
                resp.classList.remove('error');
            }} catch (e) {{
                resp.textContent = 'Error: ' + e.message;
                resp.classList.add('error');
            }}
        }}

        checkHealth();
        setInterval(checkHealth, 10000);
    </script>
</body>
</html>"""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html)