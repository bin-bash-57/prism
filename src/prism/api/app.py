from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import uvicorn
from typing import List

from src.prism.inference.triton_client import TritonClient
from src.prism.database.milvus_client import MilvusClient
from src.prism.utils.logger import get_logger
from src.prism.api.routes import router
from contextlib import asynccontextmanager


logger = get_logger("API")

triton_client = None
milvus_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup Logic
    global triton_client, milvus_client
    try:
        triton_client = TritonClient(url="localhost:8000")
        milvus_client = MilvusClient()
        logger.info("Connected to Milvus & Triton successfully")
    except Exception as e:
        logger.error(f"Startup Failed: {e}")

    yield
    logger.info("Shutting down Prism API")


app = FastAPI(title="Prism Vision API", version="1.0.0", lifespan=lifespan)
app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
triton_client = TritonClient(url="localhost:8000")
milvus_client = None


@app.on_event("startup")
async def startup_event():
    """Connect to Database on Startup"""
    global milvus_client
    try:
        milvus_client = MilvusClient()
        logger.info("Connected to Milvus Database")
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")


@app.get("/health")
def health_check():
    """Simple check to see if API is running"""
    return {"status": "healthy", "service": "prism-api"}


@app.post("/search/similar")
async def search_similar_frames(file: UploadFile = File(...), top_k: int = 5):
    """
    1. Receives an uploaded image.
    2. Runs it through Triton (YOLO) to get the embedding.
    3. Searches Milvus for similar frames.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        img_resized = cv2.resize(img, (640, 640))
        img_input = img_resized.transpose((2, 0, 1))
        img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
        img_input /= 255.0

        yolo_output = triton_client.infer("yolov8", img_input)

        if yolo_output is None:
            raise HTTPException(status_code=500, detail="Inference Failed")

        query_vector = yolo_output.flatten()[:8400].tolist()
        results = milvus_client.search(query_vector, top_k=top_k)

        matches = []
        for hits in results:
            for hit in hits:
                matches.append(
                    {
                        "id": hit.id,
                        "score": hit.distance,
                        "timestamp": hit.entity.get("timestamp"),
                        "frame_id": hit.entity.get("frame_id"),
                    }
                )

        return {"query_status": "success", "matches": matches}

    except Exception as e:
        logger.error(f"Search Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
