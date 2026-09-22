import cv2
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.detection import load_models, predict_rgb_array

camera = None
latest_frame = None
latest_detection = {"fire": 0, "violence": 0, "accident": 0, "event": "normal"}
last_detection_time = 0.0
model_cache = None
camera_lock = threading.Lock()
detection_lock = threading.Lock()


def _normal_detection():
    return {"fire": 0, "violence": 0, "accident": 0, "event": "normal"}


def start_camera():

    global camera, latest_frame, latest_detection, last_detection_time

    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
        latest_frame = None
        latest_detection = _normal_detection()
        last_detection_time = 0.0

    print("Camera opened:", camera.isOpened())

    return {"status": "Camera started"}


def stop_camera():

    global camera, latest_frame, latest_detection

    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
        latest_frame = None
        latest_detection = _normal_detection()

    return {"status": "Camera stopped"}


def generate_frames():
    global latest_frame

    while True:

        with camera_lock:
            if camera is None:
                break
            ret, frame = camera.read()
            if ret:
                latest_frame = frame.copy()

        if not ret:
            break

        ret, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame +
            b"\r\n"
        )


def get_detection():
    global latest_detection, last_detection_time, model_cache

    with camera_lock:
        if camera is None:
            return _normal_detection()
        frame = latest_frame.copy() if latest_frame is not None else None

    if frame is None or time.monotonic() - last_detection_time < 1.0:
        return latest_detection

    if not detection_lock.acquire(blocking=False):
        return latest_detection

    try:
        last_detection_time = time.monotonic()
        if model_cache is None:
            model_cache = load_models()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = predict_rgb_array(rgb_frame, model_cache)
        latest_detection = {
            name: result[name]["probability"]
            for name in ("fire", "violence", "accident")
        }
        latest_detection["event"] = result["event"]
    except Exception as error:
        print("Detection error:", error)
    finally:
        detection_lock.release()

    return latest_detection