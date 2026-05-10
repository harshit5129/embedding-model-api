from fastapi import APIRouter, HTTPException
import numpy as np
from app.schemas import SimilarityRequest, SimilarityResponse
from app.core.exceptions import ValidationError
from config import logger

router = APIRouter(prefix="", tags=["similarity"])


@router.post("/similarity", response_model=SimilarityResponse)
async def similarity(req: SimilarityRequest):
    try:
        query = np.array(req.query_embedding, dtype=np.float32)
        embeddings = np.array(req.embeddings, dtype=np.float32)
        similarities = np.dot(embeddings, query).tolist()
        return SimilarityResponse(similarities=similarities)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Similarity error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
