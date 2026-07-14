from ultralytics import YOLO

# Load a YOLO26n PyTorch model
model = YOLO("model/exp-seg-1.pt")

model.export(format="engine", half=True, imgsz=640)  # dla:0 or dla:1 corresponds to the DLA cores

# Load the exported TensorRT model
trt_model = YOLO("exp-seg-1.engine")