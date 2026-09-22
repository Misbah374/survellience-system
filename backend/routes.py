from flask import Blueprint, jsonify, send_from_directory, Response
from controllers import get_detection, start_camera, stop_camera, generate_frames

routes = Blueprint("routes", __name__)

@routes.route("/")
def home():
    return send_from_directory("../frontend", "index.html")

@routes.route("/style.css")
def style():
    return send_from_directory("../frontend", "style.css")

@routes.route("/script.js")
def script():
    return send_from_directory("../frontend", "script.js")

@routes.route("/start_camera")
def start():
    return start_camera()


@routes.route("/stop_camera")
def stop():
    return stop_camera()


@routes.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@routes.route("/detection")
def detection():
    return jsonify(get_detection())