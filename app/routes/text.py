from fastapi import APIRouter, HTTPException, Request
from app.schemas import (
    TextRequest,
    TextResponse,
    BatchTextRequest,
    BatchTextResponse,
)
from app.core.models import embed_texts_sync, is_models_loaded, get_executor
from app.core.exceptions import ModelNotLoadedError, EmbeddingError
from config import logger
import asyncio

router = APIRouter(prefix="/embed", tags=["embeddings"])


@router.post("/text", response_model=TextResponse)
async def embed_text(req: TextRequest, request: Request):
    if not is_models_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(
            get_executor(), embed_texts_sync, [req.text]
        )
        return TextResponse(embedding=embedding[0])
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="Model not loaded")
    except EmbeddingError as e:
        logger.error("Text embedding error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("Text embedding error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text/batch", response_model=BatchTextResponse)
async def embed_text_batch(req: BatchTextRequest):
    if not is_models_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.texts:
        return BatchTextResponse(embeddings=[])
    try:
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            get_executor(), embed_texts_sync, req.texts
        )
        return BatchTextResponse(embeddings=embeddings)
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="Model not loaded")
    except EmbeddingError as e:
        logger.error("Batch text embedding error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("Batch text embedding error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
