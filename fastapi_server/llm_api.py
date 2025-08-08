from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import uvicorn

app = FastAPI(
    title="LLM Wrapper API",
    version="1.0.0",
    description="A thin API layer on top of NVIDIA Triton Inference Server"
)

TRITON_HTTP_URL = "http://localhost:8000"
MODEL_NAME = "llm"

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(request: GenerateRequest):
    prompt = request.prompt.strip()

    # Triton Inference용 payload 구성
    payload = {
        "inputs": [
            {
                "name": "text_input",
                "shape": [1],
                "datatype": "BYTES",
                "data": [prompt]
            }
        ]
    }

    url = f"{TRITON_HTTP_URL}/v2/models/{MODEL_NAME}/infer"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))

    result = response.json()
    generated = result["outputs"][0]["data"][0]  # 첫 번째 결과 텍스트 추출

    return {"result": generated}


if __name__ == "__main__":
    uvicorn.run("llm_api:app", host='0.0.0.0', port=7000, reload=True)        