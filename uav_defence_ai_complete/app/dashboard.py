from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import cv2
import numpy as np
import streamlit as st

from src.assistant.assistant_agent import PerceptionAssistant
from src.assistant.tools import AssistantTools
from src.detection.rtdetr import load_rtdetr, predict
from src.knowledge.retriever import SimpleRetriever
from src.state.event_store import EventStore
from src.utils.visualization import draw_detections, save_image


st.set_page_config(page_title="UAV AI Perception Platform", layout="wide")
st.title("UAV AI Perception Platform")
st.caption("Research prototype: perception, tracking, context and operator support")

store = EventStore("outputs/state/events.db")
retriever = SimpleRetriever.from_folder("docs")
assistant = PerceptionAssistant(
    AssistantTools(store),
    retriever,
    provider=os.getenv("ASSISTANT_PROVIDER", "rule_based"),
    url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
    model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
)

st.subheader("Image detection")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
controls = st.columns([2, 1, 1])
with controls[0]:
    model_path = st.text_input("Detector weights", value="rtdetr-l.pt")
with controls[1]:
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
with controls[2]:
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)

if uploaded_image is not None:
    image_bytes = np.frombuffer(uploaded_image.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Could not decode that image. Please choose a valid image file.")
    elif st.button("Run detection", type="primary"):
        try:
            detector = load_rtdetr(model_path)
            detections = predict(detector, image, confidence, 640, device)
            annotated = draw_detections(
                image,
                [item.xyxy for item in detections],
                [item.class_name for item in detections],
                [item.confidence for item in detections],
            )
            output_path = save_image(annotated, "outputs/visualizations/dashboard_detection.jpg")
            result_columns = st.columns([1.4, 1])
            with result_columns[0]:
                st.image(annotated, channels="BGR", use_container_width=True)
            with result_columns[1]:
                st.metric("Objects detected", len(detections))
                st.caption(f"Annotated image saved to `{output_path}`")
                st.dataframe(
                    [
                        {
                            "class": item.class_name,
                            "confidence": round(item.confidence, 3),
                            "box (x1, y1, x2, y2)": tuple(round(value, 1) for value in item.xyxy),
                        }
                        for item in detections
                    ],
                    use_container_width=True,
                )
                encoded_ok, encoded_image = cv2.imencode(".jpg", annotated)
                if encoded_ok:
                    st.download_button(
                        "Download annotated image",
                        encoded_image.tobytes(),
                        file_name="detection.jpg",
                        mime="image/jpeg",
                    )
        except Exception as exc:
            st.error(f"Detection failed: {exc}")

st.divider()
left, right = st.columns([1, 1])
with left:
    st.subheader("Active tracks")
    tracks = store.active_tracks()
    st.metric("Tracks", len(tracks))
    if tracks:
        st.dataframe(tracks, use_container_width=True)
    else:
        st.info("No track records yet. Run the video pipeline first.")

with right:
    st.subheader("Recent events")
    events = store.recent_events(20)
    if events:
        st.dataframe(events, use_container_width=True)
    else:
        st.info("No events recorded yet.")

st.divider()
st.subheader("AI Perception Assistant")
question = st.text_input("Ask a question about the current perception state")
if st.button("Ask") and question:
    result = assistant.answer(question)
    st.write(result["answer"])
    if result["sources"]:
        st.caption("Approved document sources: " + ", ".join(result["sources"]))
