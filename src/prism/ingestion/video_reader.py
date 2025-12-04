import cv2
import time
import numpy as np
from typing import Generator, Tuple
from src.prism.exceptions import VideoSourceError

class VideoReader:
    def __init__(self, source: str, target_size: Tuple[int, int] = (640, 640), fps_limit: int = 0):
        self.source = 0 if source == "0" else source
        self.target_size = target_size
        self.fps_limit = fps_limit

    def stream(self) -> Generator[np.ndarray, None, None]:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise VideoSourceError(f"Failed to open video source: {self.source}")
        
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # Resize
            frame = cv2.resize(frame, self.target_size)
            yield frame

            # FPS Limiter 
            if self.fps_limit > 0:
                elapsed = time.time() - start_time
                wait = (1.0 / self.fps_limit) - elapsed
                if wait > 0:
                    time.sleep(wait)
        
        cap.release()