import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import time

from app.core.models import load_models, shutdown_executor
from app.routes import text_router, image_router, similarity_router, health_router
from config import logger

executor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global executor
    from app.core.models import get_executor
    executor = get_executor()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, load_models)
    yield
    shutdown_executor()


app = FastAPI(
    title="Embedding API",
    description="FastAPI-based embedding API with CLIP text/image embeddings",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(text_router)
app.include_router(image_router)
app.include_router(similarity_router)
app.include_router(health_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s"
    )
    return response


STATUS_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embedding API</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f0f0f; color: #fff; min-height: 100vh; }}
        .container {{ max-width: 900px; margin: 0 auto; padding: 2rem; }}
        h1 {{ font-size: 2rem; margin-bottom: 1rem; color: #00ff88; }}
        .status-bar {{ display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem; padding: 1rem; background: #1a1a1a; border-radius: 8px; }}
        .status-dot {{ width: 12px; height: 12px; border-radius: 50%; background: #ff4444; }}
        .status-dot.healthy {{ background: #00ff88; }}
        .section {{ background: #1a1a1a; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }}
        .section h2 {{ font-size: 1.2rem; margin-bottom: 1rem; color: #00ff88; }}
        .endpoint {{ display: flex; justify-content: space-between; align-items: center; padding: 1rem; background: #252525; border-radius: 8px; margin-bottom: 0.75rem; }}
        .endpoint-method {{ font-weight: bold; padding: 4px 12px; border-radius: 4px; font-size: 0.8rem; }}
        .method-get {{ background: #00ff88; color: #000; }}
        .method-post {{ background: #0088ff; color: #fff; }}
        .endpoint-path {{ color: #aaa; font-family: monospace; }}
        .test-area {{ margin-top: 1rem; }}
        textarea {{ width: 100%; padding: 0.75rem; background: #252525; border: 1px solid #333; border-radius: 8px; color: #fff; font-family: monospace; font-size: 0.9rem; resize: vertical; min-height: 80px; }}
        button {{ background: #00ff88; color: #000; border: none; padding: 0.75rem 1.5rem; border-radius: 8px; cursor: pointer; font-weight: bold; margin-top: 0.5rem; }}
        button:hover {{ opacity: 0.9; }}
        .response {{ margin-top: 1rem; padding: 1rem; background: #252525; border-radius: 8px; font-family: monospace; font-size: 0.85rem; max-height: 200px; overflow: auto; white-space: pre-wrap; word-break: break-all; }}
        .error {{ color: #ff4444; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Embedding API</h1>
        <div class="status-bar">
            <div class="status-dot" id="statusDot"></div>
            <span id="statusText">API Status: loading</span>
        </div>

        <div class="section">
            <h2>Embed Text</h2>
            <div class="endpoint">
                <span class="endpoint-method method-post">POST</span>
                <span class="endpoint-path">/embed/text</span>
            </div>
            <div class="test-area">
                <textarea id="textInput" placeholder='{{"text": "Hello world"}}'></textarea>
                <button onclick="testText()">Test</button>
                <div class="response" id="textResponse"></div>
            </div>
        </div>

        <div class="section">
            <h2>Similarity</h2>
            <div class="endpoint">
                <span class="endpoint-method method-post">POST</span>
                <span class="endpoint-path">/similarity</span>
            </div>
            <div class="test-area">
                <textarea id="simInput" placeholder='{{"query_embedding": [0.1, 0.2, ...], "embeddings": [[0.1, 0.2, ...], ...]}}'></textarea>
                <button onclick="testSimilarity()">Test</button>
                <div class="response" id="simResponse"></div>
            </div>
        </div>

        <div class="section">
            <h2>All Endpoints</h2>
            <div class="endpoint"><span class="endpoint-method method-get">GET</span><span class="endpoint-path">/health</span></div>
            <div class="endpoint"><span class="endpoint-method method-get">GET</span><span class="endpoint-path">/metrics</span></div>
            <div class="endpoint"><span class="endpoint-method method-get">GET</span><span class="endpoint-path">/</span></div>
            <div class="endpoint"><span class="endpoint-method method-post">POST</span><span class="endpoint-path">/embed/text</span></div>
            <div class="endpoint"><span class="endpoint-method method-post">POST</span><span class="endpoint-path">/embed/text/batch</span></div>
            <div class="endpoint"><span class="endpoint-method method-post">POST</span><span class="endpoint-path">/embed/image</span></div>
            <div class="endpoint"><span class="endpoint-method method-post">POST</span><span class="endpoint-path">/similarity</span></div>
        </div>
    </div>

    <script>
        async function checkHealth() {{
            try {{
                const res = await fetch('/health');
                const data = await res.json();
                const dot = document.getElementById('statusDot');
                if (data.status === 'healthy') {{
                    dot.classList.add('healthy');
                    document.getElementById('statusText').textContent = 'API Status: healthy';
                }} else {{
                    document.getElementById('statusText').textContent = 'API Status: ' + data.status;
                }}
            }} catch (e) {{
                document.getElementById('statusText').textContent = 'API Status: offline';
            }}
        }}

        async function testText() {{
            const input = document.getElementById('textInput').value;
            const resp = document.getElementById('textResponse');
            try {{
                const data = JSON.parse(input);
                const res = await fetch('/embed/text', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify(data)
                }});
                const result = await res.json();
                resp.textContent = JSON.stringify(result, null, 2);
                resp.classList.remove('error');
            }} catch (e) {{
                resp.textContent = 'Error: ' + e.message;
                resp.classList.add('error');
            }}
        }}

        async function testSimilarity() {{
            const input = document.getElementById('simInput').value;
            const resp = document.getElementById('simResponse');
            try {{
                const data = JSON.parse(input);
                const res = await fetch('/similarity', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify(data)
                }});
                const result = await res.json();
                resp.textContent = JSON.stringify(result, null, 2);
                resp.classList.remove('error');
            }} catch (e) {{
                resp.textContent = 'Error: ' + e.message;
                resp.classList.add('error');
            }}
        }}

        checkHealth();
        setInterval(checkHealth, 10000);
    </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def status_page():
    return HTMLResponse(content=STATUS_PAGE)
