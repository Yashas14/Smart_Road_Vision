"""Live Detection page: upload an image/video or use a webcam snapshot."""

from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st

# Allow importing dashboard components when run by Streamlit.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from components.metrics_cards import render_metric_row, severity_badge
from components.sidebar import render_sidebar

st.set_page_config(page_title="Live Detection", page_icon="🎥", layout="wide")

API_BASE = st.session_state.get("api_base_url", "http://localhost:8000/api/v1")
config = render_sidebar()

st.title("🎥 Live Detection")
st.caption("Upload road imagery or capture from a webcam for instant analysis.")

tab_image, tab_webcam, tab_video = st.tabs(["🖼️ Image", "📷 Webcam", "🎬 Video"])


def _post_image(file_bytes: bytes, filename: str) -> dict | None:
    """Send an image to the detection API and return the JSON result."""
    try:
        resp = requests.post(
            f"{API_BASE}/detect/image",
            files={"file": (filename, file_bytes, "application/octet-stream")},
            data={"annotate": str(config.annotate), "persist": str(config.persist)},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Detection request failed: {exc}")
        return None


def _render_result(result: dict) -> None:
    """Render detection metrics, annotated image and a per-anomaly table."""
    render_metric_row(
        {
            "Anomalies": str(result["count"]),
            "Road Score": f"{result['road_condition_score']:.0f}/100",
            "Latency": f"{result['processing_time_ms']:.0f} ms",
            "Model": result["model_version"],
        }
    )
    st.divider()
    col_img, col_table = st.columns([3, 2])
    with col_img:
        if result.get("annotated_image_base64"):
            st.image(
                f"data:image/jpeg;base64,{result['annotated_image_base64']}",
                caption="Annotated detection",
                use_container_width=True,
            )
    with col_table:
        st.subheader("Detected Anomalies")
        if not result["detections"]:
            st.success("No anomalies detected — road segment looks healthy.")
        for det in result["detections"]:
            st.markdown(
                f"**{det['class_name']}** · conf {det['confidence']:.2f} · "
                f"{severity_badge(det['severity_level'])} · "
                f"depth {det.get('depth_mm') or '—'} mm",
                unsafe_allow_html=True,
            )


with tab_image:
    uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        st.image(uploaded, caption="Input image", width=400)
        if st.button("🔍 Detect anomalies", type="primary"):
            with st.spinner("Running YOLOv11 + SAM2 + MiDaS..."):
                result = _post_image(uploaded.getvalue(), uploaded.name)
            if result:
                _render_result(result)


with tab_webcam:
    st.write("Capture a frame from your webcam and analyse it.")
    snapshot = st.camera_input("Take a picture")
    if snapshot is not None and st.button("🔍 Analyse snapshot", type="primary"):
        with st.spinner("Analysing webcam frame..."):
            result = _post_image(snapshot.getvalue(), "webcam.jpg")
        if result:
            _render_result(result)
    st.info(
        "For continuous RTSP/webcam streaming, connect a client to the "
        "WebSocket endpoint `/api/v1/ws/stream`.",
        icon="📡",
    )


with tab_video:
    video = st.file_uploader("Upload a road video", type=["mp4", "avi", "mov"])
    if video is not None and st.button("🚀 Submit for processing", type="primary"):
        try:
            resp = requests.post(
                f"{API_BASE}/detect/video",
                files={"file": (video.name, video.getvalue(), "video/mp4")},
                timeout=60,
            )
            resp.raise_for_status()
            task = resp.json()
            st.success(f"Video submitted. Task ID: `{task['task_id']}`")
            st.caption(
                "Poll `/api/v1/detect/video/{task_id}` or refresh the Analytics "
                "page once processing completes."
            )
        except requests.RequestException as exc:
            st.error(f"Video submission failed: {exc}")
