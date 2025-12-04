from src.prism.database.milvus_client import MilvusClient
from typing import List, Dict, Any


class VectorStore:
    def __init__(self):
        self.client = MilvusClient()

    def save_frame(self, frame_id: int, timestamp: float, vector: List[float]):
        """High-level method to save a frame detection."""
        self.client.insert(frame_id, timestamp, vector)

    def search_image(
        self, query_vector: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """High-level method to find similar frames."""
        raw_results = self.client.search(query_vector, top_k)

        # Clean up the messy output from Milvus
        clean_results = []
        for hits in raw_results:
            for hit in hits:
                clean_results.append(
                    {
                        "score": hit.distance,
                        "timestamp": hit.entity.get("timestamp"),
                        "frame_id": hit.entity.get("frame_id"),
                    }
                )
        return clean_results
