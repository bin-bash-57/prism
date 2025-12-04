import json
import time
import os
import cv2
import base64
from confluent_kafka import Producer
from src.prism.ingestion.video_reader import VideoReader

# Configuration
KAFKA_BROKER = "localhost:9092"
TOPIC_NAME = "video_frames"


def numpy_to_bytes(arr):
    return base64.b64encode(cv2.imencode(".jpg", arr)[1]).decode("utf-8")


def delivery_report(err, msg):
    """Called once for each message produced to indicate delivery result."""
    if err is not None:
        print(f"Message delivery failed: {err}")


def main():
    print("Initializing Kafka Producer...", flush=True)
    try:
        producer = Producer({"bootstrap.servers": KAFKA_BROKER})
        print("Kafka Producer initialized.", flush=True)
    except Exception as e:
        print(f"Failed to init Producer: {e}")
        return

    video_path = os.getenv("VIDEO_PATH", "traffic.mp4")
    print(f"Opening video: {video_path}", flush=True)

    # Initialize Reader
    try:
        reader = VideoReader(source=video_path, fps_limit=10)
    except Exception as e:
        print(f"Failed to open video: {e}")
        return

    print("Starting Streaming Loop...", flush=True)
    frame_count = 0

    try:
        for frame in reader.stream():
            frame_count += 1

            # Prepare Payload
            message = {
                "frame_id": frame_count,
                "timestamp": time.time(),
                "image": numpy_to_bytes(frame),
                "shape": frame.shape,
            }
            producer.produce(
                TOPIC_NAME,
                json.dumps(message).encode("utf-8"),
                callback=delivery_report,
            )
            producer.poll(0)

            if frame_count % 10 == 0:
                print(f"Sent frame {frame_count} to '{TOPIC_NAME}'", flush=True)
                # Force send
                producer.flush(timeout=0.1)

    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    except Exception as e:
        print(f"Loop crashed: {e}", flush=True)
    finally:
        print("Flushing remaining messages...", flush=True)
        producer.flush()
        print("Done.")


if __name__ == "__main__":
    main()
