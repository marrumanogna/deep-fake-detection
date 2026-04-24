from flask import Flask, request, jsonify
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import tempfile
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "audio_model.h5")

MAX_LEN = 500

model = load_model(MODEL_PATH, compile=False)
print("✅ Audio model loaded from:", MODEL_PATH)

@app.route("/predict-audio", methods=["POST"])
def predict_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    file = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file.save(tmp.name)
        audio_path = tmp.name

    signal, sr = librosa.load(audio_path, sr=22050)

    mfcc = librosa.feature.mfcc(y=signal, sr=22050, n_mfcc=40)

    if mfcc.shape[1] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_LEN - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :MAX_LEN]

    X = mfcc[np.newaxis, ..., np.newaxis]

    pred = model.predict(X)[0][0]

    os.remove(audio_path)

    return jsonify({
        "label": "FAKE" if pred >= 0.5 else "REAL",
        "confidence": float(pred if pred >= 0.5 else 1 - pred),
        "prob_fake": float(pred),
        "prob_real": float(1 - pred)
    })

if __name__ == "__main__":
    app.run(port=5001)
