import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
import cv2

app = Flask(__name__, static_folder='static')

# Load trained model
MODEL_PATH = "model/crop_disease_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Load class labels
LABELS_PATH = "model/class_labels.txt"

if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"❌ Labels file not found: {LABELS_PATH}")

with open(LABELS_PATH, "r") as f:
    class_labels = [line.strip() for line in f.readlines()]  # Read as a list
    class_labels = {i: label for i, label in enumerate(class_labels)}  # Convert list to dictionary


# Preprocess image
def preprocess_image(image_path):
    img_size = 224  # Same size used in training
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"❌ Image at '{image_path}' could not be loaded.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Reshape to (1, 224, 224, 3)
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "❌ No file uploaded!", 400

        file = request.files["file"]
        if file.filename == "":
            return "❌ No file selected!", 400

        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)

        # Preprocess and predict
        image = preprocess_image(filepath)
        predictions = model.predict(image)

        if predictions.shape[1] != len(class_labels):
            return f"❌ Model output size {predictions.shape[1]} does not match expected {len(class_labels)} classes.", 500

        predicted_class = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions) * 100  # Get confidence score

        return f"✅ Predicted: {predicted_class} ({confidence:.2f}%)"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
