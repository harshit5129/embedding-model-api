# Embedding Model API

FastAPI-based REST API for text and image embeddings using CLIP models.

## Features

- **Text Embeddings**: Generate vector embeddings from text
- **Image Embeddings**: Generate vector embeddings from images
- **Cosine Similarity**: Compute similarity between embeddings
- **Batch Processing**: Support for batch text embeddings
- **Status Page**: Built-in UI for testing endpoints at `/`

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn main:app --reload
```

### Docker

```bash
# Build image
docker build -t embedding-api .

# Run container
docker run -d -p 8000:8000 --restart always embedding-api
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_MODEL` | `Qdrant/clip-ViT-B-32-vision` | Image embedding model |
| `TEXT_MODEL` | `Qdrant/clip-ViT-B-32-text` | Text embedding model |
| `MAX_WORKERS` | `4` | Thread pool workers |
| `LOG_LEVEL` | `INFO` | Logging level |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Status page with testing UI |
| GET | `/health` | Health check |
| POST | `/embed/text` | Single text embedding |
| POST | `/embed/text/batch` | Batch text embeddings |
| POST | `/embed/image` | Image embedding (multipart) |
| POST | `/similarity` | Cosine similarity |

## Example Usage

### Text Embedding

```bash
curl -X POST http://localhost:8000/embed/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

### Image Embedding

```bash
curl -X POST http://localhost:8000/embed/image \
  -F "file=@image.jpg"
```

### Similarity

```bash
curl -X POST http://localhost:8000/similarity \
  -H "Content-Type: application/json" \
  -d '{"query_embedding": [0.1, 0.2], "embeddings": [[0.1, 0.2], [0.3, 0.4]]}'
```

## Deployment

### Railway (Recommended)

```bash
railway init
railway up
```

### VPS

```bash
docker build -t embedding-api .
docker run -d -p 8000:8000 --restart always embedding-api
```

## Tech Stack

- **FastAPI**: Web framework
- **FastEmbed**: CLIP embeddings
- **Pillow**: Image processing
- **NumPy**: Vector operations

## License

MIT