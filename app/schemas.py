from pydantic import BaseModel, Field, field_validator
from typing import Annotated


class TextRequest(BaseModel):
    text: Annotated[str, Field(min_length=1, max_length=10000)]


class TextResponse(BaseModel):
    embedding: list[float]


class BatchTextRequest(BaseModel):
    texts: Annotated[list[str], Field(min_length=1, max_length=1000)]

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        if not all(1 <= len(t) <= 10000 for t in v):
            raise ValueError("Each text must be between 1 and 10000 characters")
        return v


class BatchTextResponse(BaseModel):
    embeddings: list[list[float]]


class ImageResponse(BaseModel):
    embedding: list[float]


class SimilarityRequest(BaseModel):
    query_embedding: Annotated[list[float], Field(min_length=1, max_length=10000)]
    embeddings: Annotated[list[list[float]], Field(min_length=1, max_length=10000)]


class SimilarityResponse(BaseModel):
    similarities: list[float]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    text_model: str | None = None
    image_model: str | None = None


class MetricsResponse(BaseModel):
    status: str
    text_model: str
    image_model: str
    embedding_dimension: int
    max_workers: int
