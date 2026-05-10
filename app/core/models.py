from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import numpy as np
from fastembed import ImageEmbedding, TextEmbedding

from app.core.config import settings
from app.core.exceptions import ModelLoadError, ModelNotLoadedError
from config import logger

executor: Optional[ThreadPoolExecutor] = None
image_model: Optional[ImageEmbedding] = None
text_model: Optional[TextEmbedding] = None


def _norm(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    return v / norm


def load_models() -> None:
    global image_model, text_model
    try:
        logger.info(f"Loading image model: {settings.image_model}")
        image_model = ImageEmbedding(model_name=settings.image_model)
        logger.info(f"Loading text model: {settings.text_model}")
        text_model = TextEmbedding(model_name=settings.text_model)
        logger.info("FastEmbed models loaded successfully")
    except Exception as e:
        logger.error("Failed to load FastEmbed models: %s", e)
        raise ModelLoadError(f"Failed to load models: {e}")


def get_executor() -> ThreadPoolExecutor:
    global executor
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=settings.max_workers)
    return executor


def shutdown_executor() -> None:
    global executor
    if executor:
        executor.shutdown(wait=True)
        executor = None


def embed_images_sync(images: list) -> list[list[float]]:
    if image_model is None:
        raise ModelNotLoadedError("Image model not loaded")
    emb = list(image_model.embed(images))
    return [_norm(e).tolist() for e in emb]


def embed_texts_sync(texts: list[str]) -> list[list[float]]:
    if text_model is None:
        raise ModelNotLoadedError("Text model not loaded")
    emb = list(text_model.embed(texts))
    return [_norm(e).tolist() for e in emb]


def is_models_loaded() -> bool:
    return text_model is not None and image_model is not None


def get_embedding_dimension() -> int:
    if text_model is not None:
        return text_model.dim
    return 512
