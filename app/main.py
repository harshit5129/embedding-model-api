import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import time

from app.core.models import load_models, shutdown_executor
from app.routes import text_router, image_router, similarity_router, health_router, tts_router
from app.core.logging import logger

executor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global executor
    from app.core.models import get_executor
    from app.core.tts import load_tts_model
    
    executor = get_executor()
    loop = asyncio.get_event_loop()
    
    await loop.run_in_executor(executor, load_models)
    await loop.run_in_executor(executor, load_tts_model)
    
    yield
    shutdown_executor()


app = FastAPI(
    title="Embedding API + TTS",
    description="FastAPI-based API with CLIP embeddings and Pocket TTS voice synthesis",
    version="1.1.0",
    lifespan=lifespan,
)

app.include_router(text_router)
app.include_router(image_router)
app.include_router(similarity_router)
app.include_router(health_router)
app.include_router(tts_router)


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
        * { box-sizing: border-box; margin: 0; padding: 0; }
        :root { --primary: #10b981; --bg: #0a0a0a; --card: #171717; --border: #262626; --text: #fafafa; --text-dim: #a3a3a3; --error: #ef4444; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; min-height: 100vh; }
        .container { max-width: 960px; margin: 0 auto; padding: 3rem 1.5rem; }
        header { text-align: center; margin-bottom: 3rem; }
        h1 { font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; background: linear-gradient(135deg, var(--primary), #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
        .subtitle { color: var(--text-dim); font-size: 1.1rem; }
        .status-card { display: inline-flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1.5rem; background: var(--card); border: 1px solid var(--border); border-radius: 999px; margin-top: 1.5rem; }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--error); animation: pulse 2s infinite; }
        .status-dot.healthy { background: var(--primary); }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .grid { display: grid; gap: 1.5rem; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); }
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem; }
        .card-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; }
        .card-icon { width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.25rem; }
        .card-icon.text { background: #064e3b; }
        .card-icon.batch { background: #1e3a5f; }
        .card-icon.image { background: #4a1d5a; }
        .card-icon.similarity { background: #3b2f1e; }
        .card h2 { font-size: 1.1rem; font-weight: 600; }
        .endpoint { display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1rem; background: rgba(255,255,255,0.03); border-radius: 8px; margin-bottom: 1rem; font-family: monospace; font-size: 0.9rem; }
        .method { font-weight: 600; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; }
        .method-get { background: var(--primary); color: #000; }
        .method-post { background: #3b82f6; color: #fff; }
        .path { color: var(--text-dim); }
        .input-area { margin-top: 1rem; }
        .input-area label { display: block; font-size: 0.85rem; color: var(--text-dim); margin-bottom: 0.5rem; }
        textarea, input[type="file"] { width: 100%; padding: 0.875rem; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; color: var(--text); font-family: 'Fira Code', monospace; font-size: 0.875rem; }
        textarea { min-height: 100px; resize: vertical; }
        textarea:focus, input:focus { outline: none; border-color: var(--primary); }
        .btn-row { display: flex; gap: 0.75rem; margin-top: 0.75rem; }
        button { flex: 1; background: var(--primary); color: #000; border: none; padding: 0.875rem 1.5rem; border-radius: 8px; font-weight: 600; cursor: pointer; transition: opacity 0.2s; }
        button:hover { opacity: 0.9; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .response { margin-top: 1rem; padding: 1rem; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; font-family: 'Fira Code', monospace; font-size: 0.8rem; max-height: 250px; overflow: auto; white-space: pre-wrap; word-break: break-all; }
        .response.error { border-color: var(--error); color: var(--error); }
        .docs { margin-top: 3rem; }
        .docs h3 { font-size: 1.25rem; margin-bottom: 1rem; color: var(--text-dim); }
        .endpoints-list { display: flex; flex-direction: column; gap: 0.5rem; }
        .endpoint-row { display: flex; align-items: center; gap: 1rem; padding: 0.75rem 1rem; background: var(--card); border-radius: 8px; }
        .endpoint-row .method { min-width: 60px; text-align: center; }
        .endpoint-row .path { flex: 1; color: var(--text-dim); font-family: monospace; }
        .loading { opacity: 0.5; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Embedding API</h1>
            <p class="subtitle">CLIP-powered text and image embeddings</p>
            <div class="status-card">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">Loading...</span>
            </div>
        </header>

        <div class="grid">
            <div class="card">
                <div class="card-header">
                    <div class="card-icon text">T</div>
                    <h2>Text Embedding</h2>
                </div>
                <div class="endpoint">
                    <span class="method method-post">POST</span>
                    <span class="path">/embed/text</span>
                </div>
                <div class="input-area">
                    <label>Request Body</label>
                    <textarea id="textInput">{
  "text": "Hello world"
}</textarea>
                </div>
                <div class="btn-row">
                    <button onclick="testText()">Generate Embedding</button>
                </div>
                <div class="response" id="textResponse">Response will appear here...</div>
            </div>

            <div class="card">
                <div class="card-header">
                    <div class="card-icon batch">B</div>
                    <h2>Batch Text Embedding</h2>
                </div>
                <div class="endpoint">
                    <span class="method method-post">POST</span>
                    <span class="path">/embed/text/batch</span>
                </div>
                <div class="input-area">
                    <label>Request Body</label>
                    <textarea id="batchInput">{
  "texts": ["Hello", "World", "Embeddings"]
}</textarea>
                </div>
                <div class="btn-row">
                    <button onclick="testBatch()">Generate Batch</button>
                </div>
                <div class="response" id="batchResponse">Response will appear here...</div>
            </div>

            <div class="card">
                <div class="card-header">
                    <div class="card-icon image">I</div>
                    <h2>Image Embedding</h2>
                </div>
                <div class="endpoint">
                    <span class="method method-post">POST</span>
                    <span class="path">/embed/image</span>
                </div>
                <div class="input-area">
                    <label>Select Image File</label>
                    <input type="file" id="imageInput" accept="image/*">
                </div>
                <div class="btn-row">
                    <button onclick="testImage()">Generate Embedding</button>
                </div>
                <div class="response" id="imageResponse">Response will appear here...</div>
            </div>

            <div class="card">
                <div class="card-header">
                    <div class="card-icon similarity">S</div>
                    <h2>Cosine Similarity</h2>
                </div>
                <div class="endpoint">
                    <span class="method method-post">POST</span>
                    <span class="path">/similarity</span>
                </div>
                <div class="input-area">
                    <label>Request Body</label>
                    <textarea id="simInput">{
  "query_embedding": [0.1, 0.2, 0.3],
  "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
}</textarea>
                </div>
                <div class="btn-row">
                    <button onclick="testSimilarity()">Calculate Similarity</button>
                </div>
                <div class="response" id="simResponse">Response will appear here...</div>
            </div>

            <div class="card">
                <div class="card-header">
                    <div class="card-icon" style="background: #5a1d4a;">V</div>
                    <h2>Text-to-Speech</h2>
                </div>
                <div class="endpoint">
                    <span class="method method-post">POST</span>
                    <span class="path">/tts/synthesize</span>
                </div>
                <div class="endpoint" style="margin-bottom:1rem;">
                    <span class="method method-get">GET</span>
                    <span class="path">/tts/voices</span>
                </div>
                <div class="input-area">
                    <label>Text</label>
                    <textarea id="ttsText" style="min-height:60px;">Hello world, this is a test.</textarea>
                </div>
                <div class="input-area">
                    <label>Voice</label>
                    <select id="ttsVoice" style="width:100%;padding:0.875rem;background:var(--bg);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:0.875rem;">
                        <option value="alba">alba (en)</option>
                        <option value="giovanni">giovanni (it)</option>
                        <option value="lola">lola (es)</option>
                        <option value="juergen">juergen (de)</option>
                        <option value="rafael">rafael (pt)</option>
                        <option value="estelle">estelle (fr)</option>
                        <option value="anna">anna (en)</option>
                        <option value="michael">michael (en)</option>
                    </select>
                </div>
                <div class="btn-row">
                    <button onclick="testTTS()">Synthesize Speech</button>
                </div>
                <div class="response" id="ttsResponse">Response will appear here...</div>
            </div>
        </div>

        <div class="docs">
            <h3>All Endpoints</h3>
            <div class="endpoints-list">
                <div class="endpoint-row"><span class="method method-get">GET</span><span class="path">/</span><span style="color:var(--text-dim);font-size:0.85rem;">Status page</span></div>
                <div class="endpoint-row"><span class="method method-get">GET</span><span class="path">/health</span><span style="color:var(--text-dim);font-size:0.85rem;">Health check</span></div>
                <div class="endpoint-row"><span class="method method-get">GET</span><span class="path">/metrics</span><span style="color:var(--text-dim);font-size:0.85rem;">Prometheus metrics</span></div>
                <div class="endpoint-row"><span class="method method-post">POST</span><span class="path">/embed/text</span><span style="color:var(--text-dim);font-size:0.85rem;">Single text embedding</span></div>
                <div class="endpoint-row"><span class="method method-post">POST</span><span class="path">/embed/text/batch</span><span style="color:var(--text-dim);font-size:0.85rem;">Batch text embeddings</span></div>
                <div class="endpoint-row"><span class="method method-post">POST</span><span class="path">/embed/image</span><span style="color:var(--text-dim);font-size:0.85rem;">Image embedding</span></div>
                <div class="endpoint-row"><span class="method method-post">POST</span><span class="path">/similarity</span><span style="color:var(--text-dim);font-size:0.85rem;">Cosine similarity</span></div>
                <div class="endpoint-row"><span class="method method-get">GET</span><span class="path">/tts/voices</span><span style="color:var(--text-dim);font-size:0.85rem;">List available voices</span></div>
                <div class="endpoint-row"><span class="method method-post">POST</span><span class="path">/tts/synthesize</span><span style="color:var(--text-dim);font-size:0.85rem;">Synthesize speech</span></div>
            </div>
        </div>
    </div>

    <script>
        async function checkHealth() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                const dot = document.getElementById('statusDot');
                const text = document.getElementById('statusText');
                if (data.status === 'healthy') {
                    dot.classList.add('healthy');
                    text.textContent = 'API Healthy';
                } else {
                    text.textContent = 'Status: ' + data.status;
                }
            } catch (e) {
                document.getElementById('statusText').textContent = 'API Offline';
            }
        }

        async function testText() {
            const input = document.getElementById('textInput').value;
            const resp = document.getElementById('textResponse');
            resp.textContent = 'Loading...';
            try {
                const data = JSON.parse(input);
                const res = await fetch('/embed/text', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                const result = await res.json();
                resp.textContent = JSON.stringify(result, null, 2);
                resp.classList.remove('error');
            } catch (e) {
                resp.textContent = 'Error: ' + e.message;
                resp.classList.add('error');
            }
        }

        async function testBatch() {
            const input = document.getElementById('batchInput').value;
            const resp = document.getElementById('batchResponse');
            resp.textContent = 'Loading...';
            try {
                const data = JSON.parse(input);
                const res = await fetch('/embed/text/batch', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                const result = await res.json();
                resp.textContent = JSON.stringify(result, null, 2);
                resp.classList.remove('error');
            } catch (e) {
                resp.textContent = 'Error: ' + e.message;
                resp.classList.add('error');
            }
        }

        async function testImage() {
            const fileInput = document.getElementById('imageInput');
            const resp = document.getElementById('imageResponse');
            resp.textContent = 'Loading...';
            if (!fileInput.files[0]) {
                resp.textContent = 'Error: Please select an image file';
                resp.classList.add('error');
                return;
            }
            try {
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                const res = await fetch('/embed/image', {
                    method: 'POST',
                    body: formData
                });
                const result = await res.json();
                resp.textContent = JSON.stringify(result, null, 2);
                resp.classList.remove('error');
            } catch (e) {
                resp.textContent = 'Error: ' + e.message;
                resp.classList.add('error');
            }
        }

        async function testSimilarity() {
            const input = document.getElementById('simInput').value;
            const resp = document.getElementById('simResponse');
            resp.textContent = 'Loading...';
            try {
                const data = JSON.parse(input);
                const res = await fetch('/similarity', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                const result = await res.json();
                resp.textContent = JSON.stringify(result, null, 2);
                resp.classList.remove('error');
            } catch (e) {
                resp.textContent = 'Error: ' + e.message;
                resp.classList.add('error');
            }
        }

        async function testTTS() {
            const text = document.getElementById('ttsText').value;
            const voice = document.getElementById('ttsVoice').value;
            const resp = document.getElementById('ttsResponse');
            resp.textContent = 'Loading...';
            try {
                const res = await fetch('/tts/synthesize', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text, voice: voice})
                });
                if (res.ok) {
                    const blob = await res.blob();
                    const url = URL.createObjectURL(blob);
                    resp.innerHTML = '<audio controls src="' + url + '" style="width:100%;"></audio><br><span style="color:var(--primary);">Audio generated successfully!</span>';
                    resp.classList.remove('error');
                } else {
                    const result = await res.json();
                    resp.textContent = 'Error: ' + (result.detail || 'Failed to generate audio');
                    resp.classList.add('error');
                }
            } catch (e) {
                resp.textContent = 'Error: ' + e.message;
                resp.classList.add('error');
            }
        }

        checkHealth();
        setInterval(checkHealth, 10000);
    </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def status_page():
    return HTMLResponse(content=STATUS_PAGE)
