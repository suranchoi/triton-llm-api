from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
import numpy as np
import httpx  
import uvicorn

TRITON_URL = "http://localhost:8000/v2/models"

app = FastAPI(
    title="Embedding Wrapper API",
    version="1.0.0",
    description="A thin API layer on top of NVIDIA Triton Inference Server"
)

EPS = 1e-8    

def _l2_normalize(arr):
    norm = np.linalg.norm(arr) + EPS
    return arr / norm

class EmbedRequest(BaseModel):
    text: Union[str, list]
    is_query: bool = False

# Triton 호출 함수
async def query_triton_model(model_name, text, is_query):
    payload = {
        "inputs": [
            {
                "name": "text_input",
                "shape": [1],
                "datatype": "BYTES",
                "data": [text]
            }
        ]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{TRITON_URL}/{model_name}/infer", json=payload)
        response.raise_for_status()
        return np.array(response.json()["outputs"][0]["data"], dtype=np.float32)


@app.post("/embedding")
async def embed(req: EmbedRequest):
    text = req.text.strip()
    is_query = req.is_query

    emb_1 = await query_triton_model("bge-m3", text, is_query)

    prefix = "query: " if is_query else "passage: "
    emb_2 = await query_triton_model("multilingual-e5-large", prefix + text, is_query)

    emb = np.concatenate([emb_1, emb_2])
    emb = _l2_normalize(emb)

    return {
        "embedding_combined": emb.tolist()
    }


@app.post("/embeddings")
async def embed_batch(req: EmbedRequest):
    is_query = req.is_query
    results = []

    for t in req.text:
        clean_text = t.strip()

        emb_1 = await query_triton_model("bge-m3", clean_text, is_query)

        prefix = "query: " if is_query else "passage: "
        emb_2 = await query_triton_model("multilingual-e5-large", prefix + clean_text, is_query)

        combined = np.concatenate([emb_1, emb_2])
        combined = _l2_normalize(combined)

        results.append({
            "text": clean_text,
            "embedding_combined": combined.tolist()
        })

    return {"results": results}


if __name__ == "__main__":
    uvicorn.run("embedding_api:app", host='0.0.0.0', port=7001, reload=True)        