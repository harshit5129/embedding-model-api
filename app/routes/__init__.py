from .text import router as text_router
from .image import router as image_router
from .similarity import router as similarity_router
from .health import router as health_router
from .tts import router as tts_router

__all__ = ["text_router", "image_router", "similarity_router", "health_router", "tts_router"]
