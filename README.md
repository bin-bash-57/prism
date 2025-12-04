# Prism

Prism indexes video streams so you can search them semantically.

## The Idea

The idea of this repo is that video data is currently opaque. We have thousands of hours of CCTV footage, but if you want to find "a red truck", you have to watch it manually. Prism changes this by turning video into vectors. It ingests a stream, runs object detection (YOLOv8) to find interesting entities, embeds them into a vector space, and stores them in a vector database (Milvus).

In a bit more detail, here is what happens when you run the pipeline:

* **Stage 1: Ingestion.** A Producer script reads a video file (or RTSP stream) frame-by-frame. It pushes these frames into **Apache Kafka**. This decouples the heavy processing from the ingestion, allowing the system to handle backpressure.
* **Stage 2: Inference.** A Consumer worker pulls frames from Kafka and sends them to **NVIDIA Triton Inference Server**. The model (YOLOv8-Nano) detects objects and returns bounding boxes.
* **Stage 3: Embedding & Storage.** We take the detected object, flatten its features into a vector embedding, and push it into **Milvus**.
* **Stage 4: Search.** A FastAPI backend exposes a search endpoint. You upload an image (e.g., a photo of a car), and it performs a similarity search in Milvus to find the exact timestamps where that object appeared in the video.

## Hardware Hack Alert

This project was built on a **MacBook Air M2** (ARM64), which is notoriously hostile to enterprise inference stacks. The industry standard is NVIDIA Triton on Linux/AMD64.

To make this work locally, I had to engineer a hybrid architecture. Kafka and Milvus run natively on ARM64 for speed. However, Triton runs in a Docker container using **Rosetta 2 emulation** (linux/amd64). I also had to manually patch the YOLOv8 ONNX export to use `opset=12` because the latest export format (IR v10) broke the Triton backend.

It works, but it's a "heavy" stack running on a fanless laptop. It's provided here as a proof-of-concept that you can do serious MLOps engineering on consumer hardware if you are stubborn enough.

## Setup

The project uses `uv` for package management and `docker-compose` for infrastructure.

1. **Infrastructure:**

    ```bash
    make up
    ```

    *Wait ~60s for Triton to initialize.*

2. **Install Dependencies:**

    ```bash
    uv sync
    ```

3. **Run:**
    You need three terminals:

    ```bash
    # Terminal 1: The Search API
    uv run python -m src.prism.api.app

    # Terminal 2: The Ingestion (Feeds the video)
    VIDEO_PATH="traffic.mp4" uv run python -m src.prism.ingestion.producer

    # Terminal 3: The Processor (The heavy lifter)
    uv run python -m src.prism.inference.processor
    ```

4. **Interact:**
    Visit `http://localhost:8501` for the Streamlit UI.

I plan to implement DeepSORT for object tracking. Right now, the system detects objects per frame. Adding tracking would allow unique IDs for vehicles across frames, reducing duplicate search results.

## License

MIT
