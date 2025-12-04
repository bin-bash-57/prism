import json
import base64
import numpy as np
import cv2
from confluent_kafka import Consumer, KafkaError
from src.prism.inference.triton_client import TritonClient
from src.prism.database.milvus_client import MilvusClient

# Config
KAFKA_BROKER = "localhost:9092"
TOPIC = "video_frames"
GROUP_ID = "prism_consumer_group_v2"


def decode_image(base64_string):
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def run_processor():
    print("Starting Processor (Confluent Version)...", flush=True)

    conf = {
        "bootstrap.servers": KAFKA_BROKER,
        "group.id": GROUP_ID,
        "auto.offset.reset": "earliest",
    }
    consumer = Consumer(conf)
    consumer.subscribe([TOPIC])
    try:
        triton = TritonClient()
        milvus = MilvusClient()
    except Exception as e:
        print(f"❌ Init Error: {e}")
        return

    print(f"Listening on {TOPIC}...", flush=True)

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"Kafka Error: {msg.error()}")
                    continue
            try:
                data = json.loads(msg.value().decode("utf-8"))
                frame_id = data["frame_id"]

                img = decode_image(data["image"])
                img_input = cv2.resize(img, (640, 640)).transpose((2, 0, 1))
                img_input = np.expand_dims(img_input, axis=0).astype(np.float32) / 255.0
                res = triton.infer("yolov8", img_input)
                if res is None:
                    continue
                vec = res.flatten()[:8400].tolist()
                milvus.insert(frame_id, data["timestamp"], vec)

                print(f"Processed Frame {frame_id}", flush=True)
            except Exception as e:
                print(f"Error: {e}")

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        consumer.close()


if __name__ == "__main__":
    run_processor()
