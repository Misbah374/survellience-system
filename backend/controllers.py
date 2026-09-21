import cv2

camera = None


def start_camera():

    global camera

    if camera is None:
        camera = cv2.VideoCapture(0)

    print("Camera opened:", camera.isOpened())

    return {"status": "Camera started"}


def stop_camera():

    global camera

    if camera is not None:
        camera.release()
        camera = None

    return {"status": "Camera stopped"}


def generate_frames():

    while True:

        ret, frame = camera.read()

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