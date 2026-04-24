import cv2
import numpy as np
import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "deepfake2.keras")

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Image/Video model loaded from:", MODEL_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def crop_face(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(img, 1.1, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (224, 224))
    return face / 255.0

def predict_image(path):
    img = cv2.imread(path)
    face = crop_face(img)

    if face is None:
        return "NO FACE", 0.0

    X = np.expand_dims(face, axis=0)
    pred = np.argmax(model.predict(X))
    return ("FAKE" if pred == 1 else "REAL"), float(pred)

def predict_video(path):
    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        face = crop_face(frame)
        if face is None:
            continue

        X = np.expand_dims(face, axis=0)
        pred = np.argmax(model.predict(X))

        if pred == 1:
            cap.release()
            return "FAKE", 1.0

    cap.release()
    return "REAL", 1.0
