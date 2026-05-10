from dataclasses import dataclass
import os


@dataclass
class Settings:
    image_model: str = os.getenv("IMAGE_MODEL", "Qdrant/clip-ViT-B-32-vision")
    text_model: str = os.getenv("TEXT_MODEL", "Qdrant/clip-ViT-B-32-text")
    max_workers: int = int(os.getenv("MAX_WORKERS", "4"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    batch_chunk_size: int = int(os.getenv("BATCH_CHUNK_SIZE", "100"))
    max_text_length: int = 10000
    max_batch_size: int = 1000


settings = Settings()
