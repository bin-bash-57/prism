from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import cv2
from src.prism.inference.triton_client import TritonClient
from src.prism.database.vector_store import VectorStore

router = APIRouter()

triton_client = TritonClient()
vector_store = VectorStore()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/search")
async def search(file: UploadFile = File(...), top_k: int = 5):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        img_resized = cv2.resize(img, (640, 640))
        img_input = img_resized.transpose((2, 0, 1))
        img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
        img_input /= 255.0

        vector_output = triton_client.infer("yolov8", img_input)
        query_vector = vector_output.flatten()[:8400].tolist()
        results = vector_store.search_image(query_vector, top_k)

        return {"matches": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
