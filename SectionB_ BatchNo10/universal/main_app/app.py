from flask import Flask, render_template, request, jsonify
import os
import subprocess
import requests
from image_video_model import predict_image, predict_video

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

audio_api_running = False

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/load-audio")
def load_audio():
    global audio_api_running

    if not audio_api_running:
        subprocess.Popen(
            ["C:/Users/marru/Downloads/universal/start_audio_api.bat"],
            shell=True,
            cwd=os.path.dirname(os.path.abspath(__file__)) + "/.."
        )
        audio_api_running = True

    return jsonify({"status": "Audio API started"})

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return render_template(
            "index.html",
            error="No file uploaded"
        )

    file = request.files["file"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    ext = path.lower()
    preview_file = f"uploads/{file.filename}"
    preview_media_type = None

    if ext.endswith((".wav", ".mp3")):
        preview_media_type = "audio"
    elif ext.endswith((".jpg", ".jpeg", ".png")):
        preview_media_type = "image"
    elif ext.endswith((".mp4", ".avi", ".mkv")):
        preview_media_type = "video"

    # ---------------- AUDIO ----------------
    if ext.endswith((".wav", ".mp3")):
        try:
            r = requests.post(
                AUDIO_API_URL,
                files={"audio": open(path, "rb")},
                timeout=30
            )
            result = r.json()

            return render_template(
                "index.html",
                result_type="Audio",
                label=result["label"],
                confidence=round(result["confidence"] * 100, 2),
                prob_fake=round(result["prob_fake"] * 100, 2),
                prob_real=round(result["prob_real"] * 100, 2),
                preview_file=preview_file,
                preview_media_type=preview_media_type,
                preview_name=file.filename
            )

        except Exception as e:
            return render_template(
                "index.html",
                error="Audio API not running",
                preview_file=preview_file,
                preview_media_type=preview_media_type,
                preview_name=file.filename
            )

    # ---------------- IMAGE ----------------
    elif ext.endswith((".jpg", ".jpeg", ".png")):
        label, conf = predict_image(path)

        return render_template(
            "index.html",
            result_type="Image",
            label=label,
            confidence=round(conf * 100, 2),
            preview_file=preview_file,
            preview_media_type=preview_media_type,
            preview_name=file.filename
        )

    # ---------------- VIDEO ----------------
    elif ext.endswith((".mp4", ".avi", ".mkv")):
        label, conf = predict_video(path)

        return render_template(
            "index.html",
            result_type="Video",
            label=label,
            confidence=round(conf * 100, 2),
            preview_file=preview_file,
            preview_media_type=preview_media_type,
            preview_name=file.filename
        )

    return render_template(
        "index.html",
        error="Unsupported file type",
        preview_file=preview_file,
        preview_media_type=preview_media_type,
        preview_name=file.filename
    )
AUDIO_API_URL = "http://127.0.0.1:5001/predict-audio"
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5002,
        debug=False,
        use_reloader=False,
        threaded=False
    )
