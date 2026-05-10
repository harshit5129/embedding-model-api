class EmbeddingError(Exception):
    pass


class ModelNotLoadedError(EmbeddingError):
    pass


class ModelLoadError(EmbeddingError):
    pass


class ValidationError(EmbeddingError):
    pass


class RateLimitError(EmbeddingError):
    pass


class ImageProcessingError(EmbeddingError):
    pass
