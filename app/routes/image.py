from fastapi import APIRouter, File, HTTPException, UploadFile
from app.schemas import ImageResponse
from app.core.models import embed_images_sync, is_models_loaded, get_executor
from app.core.exceptions import ModelNotLoadedError, ImageProcessingError, EmbeddingError
from config import logger
import asyncio
import io
from PIL import Image

router = APIRouter(prefix="/embed", tags=["embeddings"])


@router.post("/image", response_model=ImageResponse)
async def embed_image(file: UploadFile = File(...)):
    if not is_models_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(
            get_executor(), embed_images_sync, [image]
        )
        return ImageResponse(embedding=embedding[0])
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="Model not loaded")
    except ImageProcessingError as e:
        logger.error("Image processing error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except EmbeddingError as e:
        logger.error("Image embedding error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("Image embedding error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
