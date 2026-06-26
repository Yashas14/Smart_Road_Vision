"""Sidebar configuration component for detection settings."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(slots=True)
class DetectionConfig:
    """User-selected detection configuration from the sidebar."""

    confidence: float
    iou: float
    model_choice: str
    annotate: bool
    persist: bool


def render_sidebar() -> DetectionConfig:
    """Render the configuration sidebar and return the chosen settings.

    Returns:
        A :class:`DetectionConfig` reflecting the current control values.
    """
    st.sidebar.header("⚙️ Detection Settings")

    model_choice = st.sidebar.selectbox(
        "Model",
        options=["YOLOv11", "YOLOv11 + SAM2", "YOLOv11 + SAM2 + MiDaS"],
        index=2,
        help="Choose which components run in the pipeline.",
    )
    confidence = st.sidebar.slider(
        "Confidence threshold", min_value=0.05, max_value=0.95, value=0.35, step=0.05
    )
    iou = st.sidebar.slider(
        "NMS IoU threshold", min_value=0.1, max_value=0.9, value=0.45, step=0.05
    )
    annotate = st.sidebar.checkbox("Draw annotations", value=True)
    persist = st.sidebar.checkbox("Save to database", value=True)

    st.sidebar.divider()
    st.sidebar.caption(f"API: {st.session_state.get('api_base_url', 'n/a')}")

    return DetectionConfig(
        confidence=confidence,
        iou=iou,
        model_choice=model_choice,
        annotate=annotate,
        persist=persist,
    )
