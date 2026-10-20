import json
import os
import uuid
from pathlib import Path

import numpy as np
import onnxruntime as ort
from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from PIL import Image, ImageStat, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename


# Paths are based on this file's location, so the app can find its files
# even if you start it from a different folder.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mobilenetv2-12.onnx"
LABELS_PATH = BASE_DIR / "models" / "imagenet_labels.json"
UPLOAD_FOLDER = BASE_DIR / "uploads"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_UPLOAD_SIZE = 15 * 1024 * 1024  # 15 MB

UPLOAD_FOLDER.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE

# Load the model and labels once when the app starts.
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not LABELS_PATH.is_file():
    raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"],
)
model_input = session.get_inputs()[0]

with LABELS_PATH.open(encoding="utf-8") as file:
    labels = json.load(file)

print("Model loaded.")
print("Model input name:", model_input.name)
print("Model input shape:", model_input.shape)
print("Labels loaded:", len(labels))


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def analyze_image(image_path):
    """Read original image details and prepare a tensor for MobileNetV2."""
    with Image.open(image_path) as original:
        width, height = original.size
        image_format = original.format or "Unknown"
        color_mode = original.mode
        rgb_image = original.convert("RGB")

        average = ImageStat.Stat(rgb_image).mean
        average_rgb = {
            "red": round(average[0], 1),
            "green": round(average[1], 1),
            "blue": round(average[2], 1),
        }

        resized = rgb_image.resize((224, 224))
        pixels = np.asarray(resized, dtype=np.float32) / 255.0

    # Preprocessing for the ONNX Model Zoo MobileNetV2 model.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    pixels = (pixels - mean) / std

    # Change image layout from height-width-channels to batch-channels-height-width.
    tensor = np.transpose(pixels, (2, 0, 1))[np.newaxis, ...]

    file_size = os.path.getsize(image_path)

    image_info = {
        "width": width,
        "height": height,
        "total_pixels": width * height,
        "file_size": file_size,
        "file_size_kb": round(file_size / 1024, 2),
        "format": image_format,
        "color_mode": color_mode,
        "average_rgb": average_rgb,
    }

    return image_info, tensor


def recognize_image(tensor):
    """Run the model and return the five highest-scoring labels."""
    output = session.run(None, {model_input.name: tensor})[0]
    scores = np.asarray(output, dtype=np.float32).reshape(-1)

    # Convert logits to probabilities unless the model already returned them.
    if np.all(scores >= 0) and np.isclose(scores.sum(), 1.0, atol=1e-3):
        probabilities = scores
    else:
        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / exp_scores.sum()

    top_indices = np.argsort(probabilities)[-5:][::-1]

    return [
        {
            "label": labels[index] if index < len(labels) else f"Class {index}",
            "confidence": round(float(probabilities[index]) * 100, 2),
        }
        for index in top_indices
    ]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", error=None)

    uploaded_file = request.files.get("image")

    if uploaded_file is None or uploaded_file.filename == "":
        return render_template("index.html", error="Choose an image first.")

    safe_name = secure_filename(uploaded_file.filename)

    if not allowed_file(safe_name):
        return render_template(
            "index.html",
            error="Choose a PNG, JPG, JPEG, or WebP image.",
        )

    extension = safe_name.rsplit(".", 1)[1].lower()
    saved_name = f"{uuid.uuid4().hex}.{extension}"
    saved_path = UPLOAD_FOLDER / saved_name
    uploaded_file.save(saved_path)

    try:
        # verify() checks the file contents; reopen it afterward to analyze it.
        with Image.open(saved_path) as image:
            image.verify()

        image_info, tensor = analyze_image(saved_path)
        predictions = recognize_image(tensor)

        return render_template(
            "index.html",
            error=None,
            image_url=url_for("uploaded_file", filename=saved_name),
            image_info=image_info,
            predictions=predictions,
        )

    except (UnidentifiedImageError, OSError):
        saved_path.unlink(missing_ok=True)
        return render_template(
            "index.html",
            error="That file could not be read as a valid image.",
        )

    except Exception as error:
        print("Image processing error:", error)
        return render_template(
            "index.html",
            error="The image could not be analyzed. Check the VS Code terminal.",
        )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(error):
    return render_template(
        "index.html",
        error="The image is too large. The maximum upload size is 15 MB.",
    ), 413


if __name__ == "__main__":
    # Debug mode is for local development only.
    app.run(debug=True)