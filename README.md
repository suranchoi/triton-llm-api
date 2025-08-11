# 🧠 LLM + Embedding Model API Server (Triton + FastAPI)

> This project serves GGUF-format LLM and embedding models based on Llama.cpp via Triton Inference Server,  
> and provides a REST API interface through FastAPI for inference and embedding tasks.

---

## 📁 Project Structure
This project integrates Triton Inference Server and FastAPI to serve LLM and embedding models with a clean API interface.

- The model_repository/ directory is used by Triton Inference Server to load and serve GGUF-format LLM and embedding models.
- The fastapi_server/ directory contains the REST API server code that handles client requests and communicates with the Triton backend.


```
llm-triton-api/
├── fastapi_server/
│   ├── embedding_api.py           # /embedding API route
│   └── llm_api.py                 # /generate API route
│
├── model_repository/
│   ├── bge-m3/
│   ├── multilingual-e5-large/
│   └── llm/
```

---

## 🚀 Key Features

- ✅ Serve GGUF models using Triton Python backend (based on Llama.cpp)
- ✅ REST API endpoints using FastAPI for LLM and embedding tasks
- ✅ Endpoints: `/generate` and `/embedding`

---

## 🐳 Build and Run Triton Inference Server (Docker)

### 1. Build the Docker image

```bash
docker build . -t triton-llm:latest
```

### 2. Run the container
| Note: Update the DOCSRAY_MODEL environment variable according to the path of your model directory.
```bash
docker run -d -it --name triton-llm-api \
  --gpus all \
  -e DOCSRAY_MODEL=/data2/huggingface/models/Docsray \
  --shm-size=1G \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /data2:/data2 \
  triton-llm:latest
```

### 3. Start Triton inside the container
Once inside the container shell, run:
```bash
tritonserver --model-repository /data2/llm/triton-llm-api/model_repository
```


---

### 2. Run FastAPI Server

```bash
cd fastapi_server
python llm_api.py 
python embedding_api.py 
```

---

## 🌐 API Examples

### 🔸 Text Generation (LLM)

```
POST /generate
{
  "prompt": "Hello, who are you?"
}
```

### 🔹 Text Embedding

```
POST /embedding
{
  "text": "What is artificial intelligence?",
  "is_query": false
}
```

---

## 🧪 curl Test Examples

### ✅ Text Generation

```bash
curl -X POST http://localhost:7000/generate \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "Hello, what time is it now?"
         }'
```

### ✅ Text Embedding

```bash
curl -X POST http://localhost:7000/embedding \
     -H "Content-Type: application/json" \
     -d '{
           "text": "What is AI?",
           "is_query": false
         }'
```
