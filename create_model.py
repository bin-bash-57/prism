from ultralytics import YOLO
import os
import onnx


model_path = "model_repository/yolov8/1"
os.makedirs(model_path, exist_ok=True)

model = YOLO("yolov8n.pt")
success = model.export(format="onnx", dynamic=True, opset=12)

if os.path.exists("yolov8n.onnx"):
    os.rename("yolov8n.onnx", os.path.join(model_path, "model.onnx"))
else:
    print("Export finished but file not found. Check current directory.")