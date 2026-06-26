from ultralytics import YOLO
import cv2

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink drop=true sync=false"
    )

model = YOLO("exp-3.engine")

cap = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    raise RuntimeError("Cannot open CSI camera with GStreamer pipeline")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.25,
        verbose=False,
    )

    annotated = results[0].plot()
    cv2.imshow("YOLO CSI", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()