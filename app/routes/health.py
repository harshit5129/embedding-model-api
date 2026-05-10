from fastapi import APIRouter
from app.schemas import HealthResponse, MetricsResponse
from app.core.models import is_models_loaded, get_embedding_dimension
from app.core.config import settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health():
    loaded = is_models_loaded()
    return HealthResponse(
        status="healthy" if loaded else "loading",
        models_loaded=loaded,
        text_model=settings.text_model if loaded else None,
        image_model=settings.image_model if loaded else None,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def metrics():
    loaded = is_models_loaded()
    if not loaded:
        return MetricsResponse(
            status="loading",
            text_model=settings.text_model,
            image_model=settings.image_model,
            embedding_dimension=0,
            max_workers=settings.max_workers,
        )
    return MetricsResponse(
        status="healthy",
        text_model=settings.text_model,
        image_model=settings.image_model,
        embedding_dimension=get_embedding_dimension(),
        max_workers=settings.max_workers,
    )
