## Agriculture AI Assistant — Detailed README

An intelligent, multilingual AI assistant for agriculture that provides personalized irrigation advice using weather data, soil analysis, a rule engine, and an explainable AI stack (RAG + local LLM + cloud fallback). This guide is intentionally exhaustive and aims to be crystal-clear for macOS (M1/M2) and Windows users.

### Table of Contents
- [1. Overview & Goals](#1-overview--goals)
- [2. System Requirements (Hardware / OS)](#2-system-requirements-hardware--os)
- [3. Quickstart Summary (copy-paste commands)](#3-quickstart-summary-copy-paste-commands)
- [4. Install & Setup — macOS (detailed)](#4-install--setup--macos-m1--m1-pro--m2)
- [5. Install & Setup — Windows (detailed + WSL recommendation)](#5-install--setup--windows)
- [6. Downloading the LLaMA model (GGUF)](#6-downloading-the-llama-model-gguf)
- [7. Virtual environment & dependencies](#7-virtual-environment--dependencies)
- [8. Configuration (.env, config.py) explained](#8-configuration-env-and-configpy--explained)
- [9. Project structure & explanation of each module/file](#9-project-structure--explanation-file-by-file)
- [10. How the pipeline works (step-by-step with examples)](#10-how-the-pipeline-works--end-to-end)
- [11. Running the app (development / production / Docker)](#11-running-the-app)
- [12. API reference & example requests](#12-api-reference--example-requests)
- [13. Testing — unit, integration, manual tests](#13-testing--unit-integration-manual)
- [14. Performance tuning (macOS M1/M2, Intel, Windows)](#14-performance-tuning--practical-tips)
- [15. Troubleshooting & common errors (LLM, FAISS, Weather API, TTS)](#15-troubleshooting--common-errors)
- [16. Security, API keys, and best practices](#16-security--best-practices)
- [17. Deployment options (cloud, Docker, GPU VM)](#17-deployment-options)
- [18. Contributing, license, acknowledgements](#18-contributing-license-and-acknowledgements)
- [19. Appendix: useful commands & quick cheatsheet](#19-appendix--useful-commands-examples--snippets)

---

## 1. Overview & Goals

This project combines deterministic irrigation rules with retrieval-augmented generation (RAG) and a local LLM fallback to deliver explainable, locally-capable agricultural advice.

Key capabilities:
- **STT** (Whisper) and **TTS** for voice I/O
- **NLU** (intent & entity extraction)
- **Weather integration** (OpenWeatherMap)
- **Soil local DB** + irrigation heuristics
- **RAG** with FAISS + SentenceTransformers embeddings
- **Local LLaMA-3.1 8B** quantized model for offline inference (GGUF)
- **Cloud fallback** (OpenAI, Hugging Face Inference, or Ollama)
- **FastAPI backend** + lightweight frontend for demos

Primary use-case: actionable irrigation recommendations; secondary: generalized crop advice.

---

## 2. System Requirements (Hardware / OS)

Minimum:
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Disk**: 20 GB free
- **Python**: 3.9+

Recommended (for local LLaMA inference):
- **Apple Silicon (M1/M2)** with 16+ GB unified memory — excellent performance with `llama-cpp-python` + Metal
- Or **Linux/Windows** with 16+ GB RAM; discrete GPU preferred for larger models
- **SSD** recommended for fast mmap and indexing

Notes for Windows:
- Running `llama-cpp-python` natively is possible but often easier via **WSL2 (Ubuntu)** or Docker.
- For best local LLaMA performance on Windows, use **WSL2** with GPU passthrough or a discrete GPU.

---

## 3. Quickstart Summary (copy-paste commands)

Use this to verify everything runs before deep tuning.

```bash
# Clone
git clone <repo-url> agriculture-ai-assistant
cd agriculture-ai-assistant

# Create venv (macOS / Linux)
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell)
# python -m venv .venv
# .\.venv\Scripts\Activate.ps1

# Install base deps
pip install -r requirements.txt

# Install llama-cpp-python (macOS M1/M2 with Metal)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install --force-reinstall llama-cpp-python

# Make models dir and download GGUF (after hf login)
mkdir -p models
huggingface-cli login
huggingface-cli download \
  QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf \
  --local-dir models --local-dir-use-symlinks False

# Copy env
cp env.example .env
# Edit .env: OPENWEATHER_API_KEY, OPENAI_API_KEY(optional), LOCAL_LLM_PATH=models/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf

# Run tests
python test_system.py

# Start server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 4. Install & Setup — macOS (M1 / M1 Pro / M2)

Important: Apple Silicon users should install `llama-cpp-python` with Metal support for best local performance.

### Step-by-step (macOS)
1) Install Homebrew (if you don’t have it):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2) Install system packages (CMake, Python headers, ffmpeg for audio handling):

```bash
brew install cmake pkg-config ffmpeg
```

3) Python & venv
- Use system Python or install via `pyenv` or Homebrew.
- Create and activate venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

4) Upgrade pip

```bash
pip install --upgrade pip
```

5) Install core Python packages (excluding `llama-cpp-python` for now)

```bash
pip install -r requirements.txt
```

If you get errors with sentence-transformers on macOS M1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
```

6) Install `llama-cpp-python` with Metal

```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install --force-reinstall llama-cpp-python
```

If compile fails, try:

```bash
pip install --no-cache-dir llama-cpp-python
```

7) Download model (GGUF) — see section 6.

8) Edit `.env`:

```bash
OPENWEATHER_API_KEY=your_openweather_key
OPENAI_API_KEY=your_openai_key  # optional
USE_LOCAL_LLM=true
LOCAL_LLM_PATH=models/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf
```

9) Run tests & application

```bash
python test_system.py
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### M1 / M1 Pro tuning recommendations
- If you have 16GB:
  - `n_ctx=2048`, `n_threads=4`, `n_gpu_layers=20`
- If 32GB:
  - `n_ctx=4096`, `n_threads=8`, `n_gpu_layers=35`
- If you see OOM, reduce `n_ctx` first.

---

## 5. Install & Setup — Windows

Running local LLaMA on Windows is possible, but many users prefer **WSL2 (Ubuntu)**. Both routes are provided.

### Option A: Use WSL2 (recommended)
1) Enable WSL2: see [Microsoft WSL install](https://learn.microsoft.com/windows/wsl/install)

2) Install Ubuntu from Microsoft Store

3) Inside WSL terminal (Ubuntu):

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential cmake pkg-config ffmpeg python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# llama-cpp-python (Linux build)
pip install --force-reinstall llama-cpp-python
```

4) Download GGUF with `huggingface-cli` or `hf_hub_download`.

5) Run server from WSL:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

6) Access `http://localhost:8000` from Windows host.

### Option B: Native Windows (no WSL)
- You can attempt to install `llama-cpp-python` via `pip` on Windows (MSVC build). This is more fragile.
- Many Windows users instead use **Ollama** or **Hugging Face Inference / OpenAI** for inference.

### Windows tips
- If you cannot run local LLM, set `USE_LOCAL_LLM=false` and rely on OpenAI/Hugging Face/Ollama.
- For production, prefer a Linux VM with a GPU.

---

## 6. Downloading the LLaMA model (GGUF)

Important legal note: Ensure you comply with model license terms. Some Meta models require registration and acceptance of the license.

### Using Hugging Face CLI (recommended)

```bash
pip install -U huggingface_hub
huggingface-cli login
# Paste your HF token with Repo Read permissions

mkdir -p models
huggingface-cli download \
  QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf \
  --local-dir models --local-dir-use-symlinks False

ls -lh models/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf
```

### Alternative: `hf_hub_download` in Python

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF",
    filename="Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf",
    token="hf_xxx",
)
print(path)
```

If your app expects a specific filename, either point `.env` to the real file (preferred) or rename:

```bash
mv models/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf models/llama-8b-ggml-q4_0.bin
```

---

## 7. Virtual environment & dependencies

Base dependencies (examples in `requirements.txt`):
- `fastapi`, `uvicorn`, `pydantic`, `requests`
- `whisper` or `openai-whisper` (for STT)
- `soundfile`, `pydub`
- `llama-cpp-python` (for local LLaMA)
- `huggingface_hub`, `sentence-transformers`, `faiss-cpu` (or `faiss-gpu`)
- `transformers` (utilities)
- `pytest`, `pytest-asyncio` (tests)
- `python-dotenv` or `pydantic` for env management

Install:

```bash
pip install -r requirements.txt
```

If you plan to build Python wheels or compile packages, you may need:
- `cmake`, `build-essential` (Linux), Xcode CLI (macOS)

---

## 8. Configuration (.env and config.py) — explained

Typical `.env` (example):

```bash
# API keys
OPENWEATHER_API_KEY=your_openweather_key
OPENAI_API_KEY=your_openai_key

# LLM
USE_LOCAL_LLM=true
LOCAL_LLM_PROVIDER=llama_cpp  # or "ollama" or "openai"
LOCAL_LLM_PATH=models/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf

# Weather / caching
WEATHER_CACHE_TTL_SECONDS=3600

# RAG
CHUNK_SIZE=400
CHUNK_OVERLAP=50
TOP_K_RETRIEVAL=5

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=true
```

`config.py` (Pydantic settings) loads `.env` and defines:
- `OPENWEATHER_API_KEY` — required
- `OPENAI_API_KEY` — optional (fallback)
- `LOCAL_LLM_PATH` — path to GGUF model
- `USE_LOCAL_LLM` — boolean
- `LOCAL_LLM_PROVIDER` — `llama_cpp`, `ollama`, or `openai`
- RAG and crop thresholds

Always keep API keys secret; do not commit `.env`.

---

## 9. Project structure & explanation (file-by-file)

```text
agriculture-ai-assistant/
├── app/
│   ├── main.py                # FastAPI app & routers
│   ├── __init__.py
│   └── modules/
│       ├── stt.py             # Whisper STT wrapper
│       ├── nlu.py             # Intent+entity extraction
│       ├── weather.py         # OpenWeatherMap client + caching
│       ├── soil.py            # Soil DB loader & analysis
│       ├── rules.py           # Irrigation rule engine
│       ├── rag.py             # Document ingestion & FAISS index
│       ├── llm_local.py       # Local LLaMA wrapper (llama-cpp-python)
│       ├── llm_cloud.py       # Cloud fallback (OpenAI/HF/Ollama)
│       └── response.py        # Response formatting + TTS
├── static/
│   └── index.html             # Simple frontend
├── data/                      # sample CSVs (e.g., soil_data.csv)
├── models/
│   └── Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf
├── ingest_documents.py        # CLI for document ingestion to RAG
├── test_system.py             # end-to-end system tests
├── requirements.txt
├── env.example
└── README.md
```

Module explanations (high-level):
- `app/main.py`: Creates FastAPI app; registers routers `/ask`, `/ingest-docs`, `/health`, `/config`. Initializes services at startup (STT, NLU, Weather, Soil, RAG, Local/Cloud LLM). Health checks and DI.
- `app/modules/stt.py`: Wraps Whisper STT. `transcribe(file_path) -> str` (async). Handles audio and language detection.
- `app/modules/nlu.py`: Intent classification and entity extraction (crop, growth stage, DAS, lat/lon). Returns `{ intent, entities, confidence }`.
- `app/modules/weather.py`: Calls OpenWeatherMap (`/weather`, `/forecast`), caches results, computes irrigation metrics (rain_probability_24h, ET, rainfall 7d).
- `app/modules/soil.py`: Loads `data/soil_data.csv`. `get_soil_by_location` or `get_soil_by_field_id`. Computes moisture status and irrigation depth.
- `app/modules/rules.py`: Deterministic irrigation rules and final decision.
- `app/modules/rag.py`: Document ingestion with SentenceTransformers embeddings and FAISS index (persisted to disk).
- `app/modules/llm_local.py`: Loads local LLaMA via `llama_cpp.Llama`. Async-friendly via `asyncio.to_thread`.
- `app/modules/llm_cloud.py`: Wrapper for OpenAI/HF Inference/Ollama endpoints. Used when local is disabled/low-confidence.
- `app/modules/response.py`: Builds prompts, calls LLM, parses structured JSON `{answer, explanation, sources, confidence}`; optional TTS to `/audio`.

---

## 10. How the pipeline works — end-to-end

User → `POST /ask` with `query`, `lat`, `lon`, `lang` (or audio):
1) Input stage: if voice → STT transcribes; NLU analyzes intent/entities
2) Context retrieval: weather (current+forecast), soil lookup, RAG retrieval (top-k)
3) Rule engine: deterministic decision + reason + confidence
4) Prompt construction: language-specific template with system prompt, context, user query
5) LLM generation: try local first; if low-confidence/fail → fallback to cloud
6) Postprocessing: parse structured response, optional TTS, return JSON with weather/soil/rule outputs and metadata

Why RAG + Rules + LLM? RAG provides factual grounding; rules ensure safety and explainability; local LLM enables offline privacy; cloud fallback improves quality.

---

## 11. Running the app

### Development mode

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- `--reload` for dev only
- Visit `http://localhost:8000/docs` for OpenAPI UI

### Production (uvicorn + gunicorn example)

```bash
pip install gunicorn
gunicorn -k uvicorn.workers.UvicornWorker app.main:app -b 0.0.0.0:8000 --workers 3
```

### Using Docker

Dockerfile should include:
- System deps for `llama-cpp-python` (or use a suitable base image)
- Copy app; mount models directory as a volume
- Expose port 8000

Example run (volume for models):

```bash
docker build -t agriculture-ai-assistant .
docker run -p 8000:8000 -v $(pwd)/models:/app/models -e OPENWEATHER_API_KEY=... agriculture-ai-assistant
```

### Running test suite

```bash
pytest tests/
```

---

## 12. API reference & example requests

### POST `/ask`
Form fields:
- `query` (str) or `audio_file`
- `lat` (float), `lon` (float)
- `lang` (hi/en/hinglish) (optional)

Example:

```bash
curl -X POST "http://localhost:8000/ask" \
  -F "query=Should I irrigate my rice field today?" \
  -F "lat=26.9124" \
  -F "lon=75.7873" \
  -F "lang=en"
```

Response (example):

```json
{
  "answer": "...",
  "explanation": "...",
  "confidence": 0.82,
  "sources": ["weather", "soil", "doc:irrigation_guide.pdf"],
  "audio_url": "/audio/response_2025-08-18_12345.mp3",
  "weather_data": {"...": "..."},
  "soil_data": {"...": "..."},
  "rule_output": {"...": "..."}
}
```

### POST `/ingest-docs`
Upload PDF/TXT/CSV for RAG ingestion.

```bash
curl -X POST "http://localhost:8000/ingest-docs" -F "file=@some_pdf.pdf"
```

### GET `/health`
Returns basic statuses of services.

---

## 13. Testing — Unit, Integration, Manual

### Unit tests
- `tests/unit/test_<module>.py`

Run:

```bash
pytest tests/unit/
```

### Integration tests
- `tests/integration/test_end_to_end.py`
- May call external APIs (OpenWeatherMap). Use `responses` or `vcrpy` for mocking.

### Manual verification steps
1) Start server
2) Visit `/docs` — try `/ask`
3) Ingest a PDF and ask a question referencing it

---

## 14. Performance tuning & practical tips

### macOS (M1 / M2)
- Use `CMAKE_ARGS="-DLLAMA_METAL=on"` when installing `llama-cpp-python`
- Recommended defaults for M1 Pro (16GB):

```python
n_ctx=2048
n_threads=4
n_gpu_layers=20
use_mlock=True
use_mmap=True
```

- For 32GB M1/M2 Max: `n_ctx=4096`, `n_gpu_layers=35`, `n_threads=8`

### Linux / x86
- If GPU (NVIDIA) available, use `faiss-gpu` and appropriate llama-cpp GPU options if supported
- Otherwise use `faiss-cpu`

### Windows
- Prefer WSL2; reduce `n_ctx` to 1024–2048 initially

### RAG optimization
- Use `CHUNK_SIZE=400` and `CHUNK_OVERLAP=50` for readable chunks
- Persist FAISS index to disk after ingestion to save embedding time

---

## 15. Troubleshooting & common errors

1) LLaMA model not found
- Ensure `LOCAL_LLM_PATH` in `.env` points to the GGUF file
- Confirm file exists: `ls -lh models/*.gguf`

2) `llama_cpp` import error / build fails
- On macOS: install Xcode CLI tools: `xcode-select --install`
- Ensure `cmake` installed via Homebrew
- Reinstall: `CMAKE_ARGS="-DLLAMA_METAL=on" pip install --force-reinstall llama-cpp-python`

3) App hangs after `ggml_metal_init: skipping kernel_*`
- Often not fatal; first inference may take minutes
- If CPU usage is zero for >10 minutes, abort and reduce `n_ctx` to 2048 or 1024
- Check Activity Monitor or `top`

4) FAISS errors (index building)
- Ensure `faiss-cpu` matches Python version and OS
- If memory errors, reduce chunk size or ingest in smaller batches

5) OpenWeatherMap API errors
- Check `OPENWEATHER_API_KEY` in `.env`
- Confirm plan supports the endpoints you call

6) TTS audio not playable
- Ensure `ffmpeg` is installed and that the TTS engine outputs valid mp3/wav

7) Permission errors on model file
- `chmod 644 models/*.gguf` and run server as the same user

---

## 16. Security & best practices

- Never commit `.env` or API keys to git
- Use environment variable management in production (e.g., Kubernetes secrets, AWS Parameter Store)
- Enable rate limiting and authentication on `/ingest-docs`
- Sanitize user inputs if you expose UI widely
- Validate/log all third-party calls and handle retries gracefully

---

## 17. Deployment options

### Option 1: Docker (stateless)
- Use volumes for `models/` and `data/`
- Use environment variables for keys
- Prefer cloud VMs with 16+ GB for local LLM

### Option 2: Cloud LLM fallback (cheap)
- Disable local LLM on small machines: `USE_LOCAL_LLM=false`
- Use OpenAI/Hugging Face Inference or Ollama as fallback

### Option 3: Dedicated inference server
- Host llama-cpp inference on a GPU server or optimized instance (e.g., GCP/AWS)
- Expose internal API to frontend & rule engine

---

## 18. Contributing, License, and Acknowledgements

- **License**: MIT (see `LICENSE`)
- **Contributions**: Fork → branch → PR → tests
- **Acknowledgements**:
  - OpenAI (Whisper, GPT)
  - Meta (LLaMA)
  - Hugging Face & SentenceTransformers
  - FAISS, FastAPI

---

## 19. Appendix — Useful commands, examples & snippets

### Example LLM init (safer for M1 Pro)

```python
from llama_cpp import Llama

llm = Llama(
    model_path="models/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=20,
    use_mlock=True,
    verbose=False,
)
```

### Example prompt template (English)

```text
System: You are an expert agricultural irrigation advisor...
Context: <weather data> <soil data> <rule_output> <docs>
User: <user query>
Assistant:
```

### Warmup snippet (run after model load to avoid long first-use hang)

```python
# call in a background thread after load
res = llm.create_completion("Hello. Warmup test. Provide a short reply.", max_tokens=16, temperature=0.0)
print("Warmup done:", res["choices"][0]["text"])
```

### Curl example for ingesting docs

```bash
curl -X POST http://localhost:8000/ingest-docs -F "file=@agriculture_guide.pdf"
```

---

### Detailed explanations (extra clarity)

**Why RAG + LLM + Rules?**
- Rules grant deterministic, auditable safety (e.g., skip irrigation if heavy rain predicted)
- RAG brings verifiable, domain-specific knowledge into prompts
- Local LLM allows offline usage and privacy (farmer data stays local)
- Cloud LLM is an optional fallback for high-quality responses when local fails

**Prompt engineering & templates**
- Use a strong `system_prompt` to define role and constraints
- Include concise `context_info`—weather, soil, rule outputs—to reduce hallucination
- Use structured output instructions (Answer / Explanation / Sources) for reliable parsing

**How Confidence is computed (example approach)**
- Based on model initialization, response length, presence of explanation, and rule congruence
- If local LLM confidence is low or mismatches rule engine → escalate to cloud

### Example real-life workflow (farmer scenario)
1) Farmer asks via phone: “Should I irrigate my rice today?”
2) App transcribes (STT) → NLU finds `intent=irrigation`, field `lat/lon`
3) Weather API returns 60% rain probability
4) Soil service shows moisture above critical threshold
5) Rule engine: `skip_irrigation` (primary reason: forecast rain)
6) LLM constructs final advice: short answer + explanation + sources (weather & soil)
7) TTS produced; audio served; JSON returned with confidence ~0.87

---

### Final notes & recommendations
- Start with cloud LLM if hardware is limited, then add local LLM later
- Download GGUF only after reading and accepting model license terms
- Back up your FAISS index and model files; they take time to build/download
- For hackathons/demos: set `USE_LOCAL_LLM=true` for offline demos on M1 Pro (safe settings), or `USE_LOCAL_LLM=false` to rely on OpenAI for speed 
