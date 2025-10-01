"""Streamlit user interface for the FER2013 emotion classifier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

try:  # Pillow >= 9.1
    _RESAMPLING = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    _RESAMPLING = Image.LANCZOS  # type: ignore[attr-defined]

EMOTION_LABELS: Tuple[str, ...] = (
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
)

MODEL_PATH = Path(__file__).resolve().parent / "fer2013_emotion_cnn.h5"


def _normalise_inbound_nodes(layer: Dict[str, Any]) -> None:
    """Translate legacy inbound node dictionaries into modern list format."""
    nodes = layer.get("inbound_nodes")
    if not isinstance(nodes, list) or not nodes:
        return
    if not isinstance(nodes[0], dict):
        return

    converted: list[list[list[Any]]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        args = node.get("args", [])
        kwargs = node.get("kwargs", {})
        inbound_group: list[list[Any]] = []
        for arg in args:
            if not isinstance(arg, dict):
                continue
            config = arg.get("config")
            history = None
            if isinstance(config, dict):
                history = config.get("keras_history")
            if history and len(history) == 3:
                inbound_group.append(
                    [history[0], int(history[1]), int(history[2]), kwargs or {}]
                )
        if inbound_group:
            converted.append(inbound_group)
    if converted:
        layer["inbound_nodes"] = converted
    else:
        layer["inbound_nodes"] = []


def _patch_keras_config_shapes(payload: Any) -> None:
    """Recursively replace legacy batch_shape entries and inbound formats."""
    if isinstance(payload, dict):
        if "batch_shape" in payload and "batch_input_shape" not in payload:
            payload["batch_input_shape"] = payload.pop("batch_shape")
        if {"class_name", "config"}.issubset(payload.keys()):
            _normalise_inbound_nodes(payload)
        for value in payload.values():
            _patch_keras_config_shapes(value)
    elif isinstance(payload, list):
        for item in payload:
            _patch_keras_config_shapes(item)


def _load_legacy_keras_model(model_path: Path) -> tf.keras.Model:
    """Load old Keras .h5 files that store batch_shape on input layers."""
    with h5py.File(model_path, "r") as h5_file:
        raw_config = h5_file.attrs.get("model_config")
        if raw_config is None:
            raise RuntimeError("Model file is missing a JSON-encoded configuration.")
        if isinstance(raw_config, bytes):
            raw_config = raw_config.decode("utf-8")
        if not isinstance(raw_config, str):
            raise RuntimeError("Unexpected model configuration format encountered.")
        config_obj = json.loads(raw_config)

    _patch_keras_config_shapes(config_obj)
    custom_objects = {"DTypePolicy": tf.keras.mixed_precision.Policy}
    reconstructed = tf.keras.models.model_from_json(
        json.dumps(config_obj), custom_objects=custom_objects
    )
    reconstructed.load_weights(model_path)
    return reconstructed


def _normalize_scores(raw_scores: np.ndarray) -> np.ndarray:
    """Convert raw logits to a probability distribution."""
    total = float(np.sum(raw_scores))
    if total <= 0:
        return np.full_like(raw_scores, 1.0 / raw_scores.size)
    return raw_scores / total


@st.cache_resource(show_spinner=False)
def load_model() -> tf.keras.Model:
    """Load and cache the trained CNN model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH.name}'. "
            "Place the trained model in the project root before running the app."
        )
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except (ValueError, TypeError) as exc:
        if "batch_shape" in str(exc):
            try:
                return _load_legacy_keras_model(MODEL_PATH)
            except Exception as legacy_exc:
                raise RuntimeError("Failed to load the legacy emotion recognition model.") from legacy_exc
        raise RuntimeError("Failed to load the emotion recognition model.") from exc
    except Exception as exc:  # pragma: no cover - TensorFlow raises many subclasses
        raise RuntimeError("Failed to load the emotion recognition model.") from exc


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert an arbitrary image into the shape expected by the model."""
    corrected = ImageOps.exif_transpose(image)
    rgb_image = ImageOps.fit(corrected.convert("RGB"), size=(48, 48), method=_RESAMPLING)
    array = np.asarray(rgb_image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)
    return array


def predict_emotion(image: Image.Image) -> Tuple[str, Dict[str, float]]:
    """Run the CNN on the supplied image and return the label and probabilities."""
    processed = preprocess_image(image)
    model = load_model()
    raw_scores = np.asarray(model.predict(processed, verbose=0)[0], dtype=np.float32)
    probabilities = _normalize_scores(raw_scores)
    scores = {
        label: float(probabilities[idx])
        for idx, label in enumerate(EMOTION_LABELS)
    }
    top_label = max(scores, key=scores.get)
    return top_label, scores


def render_probability_chart(probabilities: Dict[str, float]) -> None:
    """Display a sorted probability bar chart."""
    chart_data = (
        pd.DataFrame(
            {
                "Emotion": list(probabilities.keys()),
                "Confidence": [score * 100 for score in probabilities.values()],
            }
        )
        .sort_values("Confidence", ascending=False)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(chart_data)))
    bars = ax.bar(
        chart_data.index,
        chart_data["Confidence"],
        color=colors,
        edgecolor="white",
        linewidth=1.2,
    )
    ax.set_xticks(chart_data.index)
    ax.set_xticklabels(chart_data["Emotion"], rotation=30, ha="right")
    ax.set_ylabel("Confidence (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, min(100, chart_data["Confidence"].max() * 1.15))
    ax.set_title("Emotion Probability Distribution", fontsize=12, fontweight="bold", pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main() -> None:
    """Entry point for the Streamlit app."""
    st.set_page_config(
        page_title="Face Emotion Detector",
        page_icon=":smiley:",
        layout="centered",
    )

    st.markdown(
        """
        <style>
        .emotion-highlight {
            font-size: 1.1rem;
            font-weight: 600;
            padding: 0.5rem 0.75rem;
            border-radius: 0.5rem;
            background: linear-gradient(135deg, rgba(255,111,97,0.15), rgba(255,195,160,0.25));
            display: inline-block;
            margin-top: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Face Emotion Detector")
    st.caption(
        "Upload a cropped face photo or take a quick snapshot to predict the dominant emotion."
    )

    with st.sidebar:
        st.header("Tips for best results")
        st.write(
            "- Use a clear, front-facing image of a single person.\n"
            "- Ensure the face is well lit and occupies most of the frame.\n"
            "- Neutral backgrounds reduce noise for the model."
        )
        st.divider()
        st.header("About the model")
        st.write(
            "The CNN was trained on the FER-2013 dataset and reaches about **63%** accuracy on the validation split."
        )

    st.subheader("Load a face image")
    uploaded_file = st.file_uploader(
        "Upload a JPEG or PNG image",
        type=("png", "jpg", "jpeg"),
        accept_multiple_files=False,
    )
    st.write("or")
    camera_photo = st.camera_input("Take a quick photo")

    chosen_image: Image.Image | None = None
    image_source: str | None = None

    if camera_photo is not None:
        chosen_image = Image.open(camera_photo)
        image_source = "Camera snapshot"
    elif uploaded_file is not None:
        chosen_image = Image.open(uploaded_file)
        image_source = "Uploaded file"

    if chosen_image is None:
        st.info("Start by uploading an image or capturing a photo.")
        return

    st.subheader("Preview")
    st.image(chosen_image, caption=image_source, width="stretch")

    with st.spinner("Analysing emotion..."):
        try:
            predicted_label, probability_map = predict_emotion(chosen_image)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()
        except RuntimeError as exc:
            st.error(
                "Something went wrong while loading the model. "
                "Check the console logs for details."
            )
            st.stop()

    confidence_pct = probability_map[predicted_label] * 100
    st.markdown(
        f"<p class='emotion-highlight'>Predicted emotion: <strong>{predicted_label}</strong> "
        f"({confidence_pct:.1f}% confidence)</p>",
        unsafe_allow_html=True,
    )

    with st.expander("See full probability breakdown", expanded=True):
        render_probability_chart(probability_map)
        st.dataframe(
            pd.DataFrame(
                {
                    "Emotion": list(probability_map.keys()),
                    "Confidence (%)": [
                        round(score * 100, 2) for score in probability_map.values()
                    ],
                }
            ).sort_values("Confidence (%)", ascending=False),
            width="stretch",
            hide_index=True,
        )

    st.divider()
    st.subheader("Next steps")
    st.write(
        "Want to get this running on the web? Push the project to GitHub and deploy it in minutes with "
        "[Streamlit Community Cloud](https://streamlit.io/cloud)."
    )


if __name__ == "__main__":
    main()
