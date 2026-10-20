import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageStat, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mobilenetv2-12.onnx"
LABELS_PATH = BASE_DIR / "models" / "imagenet_labels.json"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_UPLOAD_MB = 15

st.set_page_config(
    page_title="VisionAI Pro | Image Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --ink: #f4f2ff;
        --muted: #aaa8c3;
        --cyan: #54f5dc;
        --pink: #f879ed;
        --purple: #9b7bff;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 10% 2%, rgba(155, 123, 255, .25), transparent 34rem),
            radial-gradient(ellipse at 95% 22%, rgba(35, 236, 205, .16), transparent 33rem),
            linear-gradient(145deg, #090914, #0b1020 58%, #071a1b);
        color: var(--ink);
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    .hero {
        margin: .5rem 0 2rem;
    }
    .brand-row {
        display: flex;
        align-items: center;
        gap: 13px;
        margin-bottom: 2.8rem;
    }
    .brand-icon {
        display: grid;
        width: 48px;
        height: 48px;
        place-items: center;
        border: 1px solid rgba(84, 245, 220, .7);
        border-radius: 15px;
        background: linear-gradient(140deg, #9b7bff, #6c4cff 45%, #54f5dc);
        color: #090914;
        font-size: 23px;
        font-weight: 900;
        box-shadow: 0 0 30px rgba(84, 245, 220, .22);
    }
    .brand-eyebrow, .hero-kicker {
        margin: 0;
        color: #65eed8;
        font-size: .7rem;
        font-weight: 800;
        letter-spacing: .17em;
        text-transform: uppercase;
    }
    .brand-title {
        margin: .1rem 0 0;
        color: var(--ink);
        font-size: 1.25rem;
        font-weight: 800;
    }
    .brand-title span {
        color: var(--cyan);
    }
    .hero h1 {
        max-width: 780px;
        margin: .6rem 0 .8rem;
        background: linear-gradient(95deg, #fff 5%, #d6baff 52%, #54f5dc 100%);
        background-clip: text;
        color: transparent;
        font-size: clamp(2.6rem, 6vw, 4.6rem);
        font-weight: 850;
        letter-spacing: -.055em;
        line-height: 1.02;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-copy {
        max-width: 660px;
        margin: 0;
        color: #b9b8cd;
        font-size: 1.05rem;
        line-height: 1.65;
    }
    .section-kicker {
        margin: 0 0 .35rem;
        color: var(--cyan);
        font-size: .68rem;
        font-weight: 800;
        letter-spacing: .15em;
        text-transform: uppercase;
    }
    .section-title {
        margin: 0 0 1rem;
        color: var(--ink);
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -.03em;
    }
    [data-testid="stFileUploader"] section {
        border: 1px dashed rgba(84, 245, 220, .65);
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(102, 45, 160, .2), rgba(0, 198, 174, .1));
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #a3fff0;
        background: linear-gradient(135deg, rgba(133, 66, 210, .29), rgba(0, 220, 190, .18));
    }
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #aaa8c3;
    }
    .stButton > button[kind="primary"] {
        min-height: 3rem;
        border: 0;
        border-radius: 12px;
        background: linear-gradient(100deg, #f879ed, #a77cff 48%, #54f5dc);
        color: #100d1b;
        font-weight: 850;
        box-shadow: 0 8px 28px rgba(155, 123, 255, .2);
    }
    .stButton > button[kind="primary"]:hover {
        border: 0;
        filter: brightness(1.1);
        box-shadow: 0 10px 34px rgba(84, 245, 220, .24);
    }
    [data-testid="stMetric"] {
        min-height: 112px;
        padding: 1rem;
        border: 1px solid rgba(190, 179, 255, .12);
        border-radius: 14px;
        background: rgba(7, 12, 27, .5);
    }
    [data-testid="stMetricLabel"] {
        color: #aaa8c3;
    }
    [data-testid="stMetricValue"] {
        color: #f4f2ff;
    }
    [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #9b7bff, #f879ed, #54f5dc);
    }
    [data-testid="stAlert"] {
        border-radius: 12px;
    }
    .result-panel {
        height: 100%;
        padding: 1.25rem;
        border: 1px solid rgba(190, 179, 255, .15);
        border-radius: 18px;
        background: linear-gradient(145deg, rgba(24, 19, 45, .88), rgba(9, 25, 35, .9));
    }
    .prediction-row {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        padding: .75rem .8rem;
        border: 1px solid rgba(190, 179, 255, .08);
        border-radius: 11px;
        background: rgba(7, 12, 27, .46);
        color: #f4f2ff;
        font-size: .9rem;
    }
    .prediction-row.top {
        border-color: rgba(84, 245, 220, .25);
        background: linear-gradient(100deg, rgba(155, 123, 255, .13), rgba(84, 245, 220, .08));
    }
    .prediction-score {
        color: #54f5dc;
        font-weight: 800;
        white-space: nowrap;
    }
    .footnote {
        color: #aaa8c3;
        font-size: .8rem;
        line-height: 1.5;
    }
    footer {
        visibility: hidden;
    }
    @media (max-width: 700px) {
        .block-container {
            padding: 1.2rem 1rem 2rem;
        }
        .brand-row {
            margin-bottom: 2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model():
    """Load ONNX model and class labels once per app process."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not LABELS_PATH.is_file():
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

    with LABELS_PATH.open(encoding="utf-8") as file:
        class_labels = json.load(file)

    inference_session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )
    input_name = inference_session.get_inputs()[0].name
    return inference_session, input_name, class_labels


def analyze_image(image_bytes):
    """Read original image metadata and prepare the MobileNetV2 tensor."""
    with Image.open(BytesIO(image_bytes)) as check_image:
        check_image.verify()

    with Image.open(BytesIO(image_bytes)) as original:
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

        resized = rgb_image.resize((224, 224), Image.Resampling.LANCZOS)
        pixels = np.asarray(resized, dtype=np.float32) / 255.0

    # Preprocessing documented for this ONNX Model Zoo MobileNetV2 model.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    pixels = (pixels - mean) / std

    # Convert HWC to NCHW and add the batch dimension.
    tensor = np.transpose(pixels, (2, 0, 1))[np.newaxis, ...]

    image_info = {
        "width": width,
        "height": height,
        "total_pixels": width * height,
        "file_size": len(image_bytes),
        "file_size_kb": round(len(image_bytes) / 1024, 2),
        "format": image_format,
        "color_mode": color_mode,
        "average_rgb": average_rgb,
    }
    return image_info, tensor


def predict(inference_session, input_name, tensor, class_labels):
    """Run inference and return the five most likely categories."""
    raw_output = inference_session.run(None, {input_name: tensor})[0]
    scores = np.asarray(raw_output, dtype=np.float32).reshape(-1)

    if not np.all(np.isfinite(scores)):
        raise ValueError("The model returned invalid scores.")

    if np.all(scores >= 0) and np.isclose(scores.sum(), 1.0, atol=1e-3):
        probabilities = scores
    else:
        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / exp_scores.sum()

    top_indices = np.argsort(probabilities)[-5:][::-1]
    return [
        {
            "label": (
                class_labels[index]
                if index < len(class_labels)
                else f"Class {index}"
            ),
            "confidence": round(float(probabilities[index]) * 100, 2),
        }
        for index in top_indices
    ]


def show_results(image_bytes, image_info, predictions):
    st.markdown(
        '<p class="section-kicker">Analysis complete</p>'
        '<h2 class="section-title">Your results</h2>',
        unsafe_allow_html=True,
    )

    image_column, prediction_column = st.columns([1.05, 1])

    with image_column:
        with st.container(border=True):
            st.markdown("#### Uploaded image")
            st.image(image_bytes, use_container_width=True)

    with prediction_column:
        with st.container(border=True):
            st.markdown("#### Top predictions")

            for rank, prediction in enumerate(predictions):
                label_column, score_column = st.columns([5, 1])
                with label_column:
                    st.write(prediction["label"])
                with score_column:
                    st.write(f'{prediction["confidence"]:.2f}%')
                st.progress(min(prediction["confidence"] / 100, 1.0))

            st.caption("Scores are model predictions, not guarantees.")

    st.markdown("")
    st.markdown("#### Image details")
    rows = [
        [
            ("Dimensions", f'{image_info["width"]:,} × {image_info["height"]:,}'),
            ("Total pixels", f'{image_info["total_pixels"]:,}'),
            ("File size", f'{image_info["file_size_kb"]:.2f} KB'),
        ],
        [
            ("Format", image_info["format"]),
            ("Color mode", image_info["color_mode"]),
            (
                "Average RGB",
                "{red}, {green}, {blue}".format(**image_info["average_rgb"]),
            ),
        ],
    ]

    for row in rows:
        columns = st.columns(3)
        for column, (label, value) in zip(columns, row):
            with column:
                st.metric(label, value)

    st.caption(
        "Image dimensions and pixel count describe the original file. "
        "Classification uses a resized copy."
    )

st.markdown(
    """
    <div class="hero">
      <div class="brand-row">
        <div class="brand-icon">V</div>
        <div>
          <p class="brand-eyebrow">Local computer vision</p>
          <p class="brand-title">VisionAI <span>Pro</span></p>
        </div>
      </div>
      <p class="hero-kicker">Image intelligence system</p>
      <h1>See more in every image.</h1>
      <p class="hero-copy">
        Upload an image for MobileNetV2 predictions and technical image details.
        Analysis runs locally on the server without a paid AI API.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="section-kicker">Start an analysis</p>'
    '<h2 class="section-title">Choose an image</h2>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a PNG, JPG, JPEG, or WebP image",
    type=["png", "jpg", "jpeg", "webp"],
    max_upload_size=MAX_UPLOAD_MB,
    help=f"Maximum file size: {MAX_UPLOAD_MB} MB.",
)

if uploaded_file is None:
    st.caption("Supported formats: PNG, JPG, JPEG, and WebP. Maximum size: 15 MB.")
    st.session_state.pop("last_analysis", None)
else:
    image_bytes = uploaded_file.getvalue()
    signature = hashlib.sha256(image_bytes).hexdigest()
    extension = Path(uploaded_file.name).suffix.lower().lstrip(".")

    if extension not in ALLOWED_EXTENSIONS:
        st.error("Choose a PNG, JPG, JPEG, or WebP image.")
    elif len(image_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error("The image is too large. The maximum upload size is 15 MB.")
    else:
        preview_column, action_column = st.columns([1.2, 0.8])

        with preview_column:
            st.image(image_bytes, caption=uploaded_file.name, use_container_width=True)

        with action_column:
            st.markdown(
                '<div class="result-panel">'
                '<p class="section-kicker">Ready when you are</p>'
                '<p style="color:#c3c0d8;line-height:1.6">'
                'The image will be classified and its original dimensions, '
                'format, file size, and average color will be calculated.'
                '</p></div>',
                unsafe_allow_html=True,
            )

        if st.button("Analyze image", type="primary", use_container_width=True):
            st.session_state.pop("last_analysis", None)
            try:
                with st.spinner("Loading the model and analyzing your image..."):
                    image_info, tensor = analyze_image(image_bytes)
                    model_session, input_name, class_labels = load_model()
                    predictions = predict(
                        model_session,
                        input_name,
                        tensor,
                        class_labels,
                    )

                st.session_state["last_analysis"] = {
                    "signature": signature,
                    "image_info": image_info,
                    "predictions": predictions,
                }
                st.success("Image analysis complete.")

            except (UnidentifiedImageError, OSError, ValueError) as error:
                st.error(f"Could not analyze this image: {error}")
            except Exception as error:
                st.error(
                    "The model could not analyze the image. "
                    f"Check that the model and labels are present. Details: {error}"
                )

        previous_analysis = st.session_state.get("last_analysis")
        if (
            previous_analysis
            and previous_analysis.get("signature") == signature
        ):
            st.divider()
            show_results(
                image_bytes,
                previous_analysis["image_info"],
                previous_analysis["predictions"],
            )

st.markdown(
    '<p class="footnote" style="text-align:center;margin-top:3rem">'
    'VisionAI Pro · Local image intelligence · Predictions may be incorrect'
    '</p>',
    unsafe_allow_html=True,
)
