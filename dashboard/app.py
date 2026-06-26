"""SmartRoadVision — Streamlit monitoring dashboard (entry point).

Run with::

    streamlit run dashboard/app.py

The dashboard talks to the FastAPI backend over HTTP (configurable via the
``API_BASE_URL`` environment variable) and provides four pages: live detection,
analytics, a geospatial map and report generation.
"""

from __future__ import annotations

import os

import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="SmartRoadVision",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Make the API base URL available to all pages.
st.session_state.setdefault("api_base_url", API_BASE_URL)


def main() -> None:
    """Render the dashboard landing page."""
    st.title("🛣️ SmartRoadVision — Road Surface Anomaly Detection")
    st.caption("Version 2.0 · YOLOv11 + SAM2 + MiDaS · Real-time monitoring")

    st.markdown(
        """
        Welcome to the **SmartRoadVision** operations dashboard. Use the pages in
        the sidebar to:

        - **Live Detection** — upload images/video or use a webcam for real-time
          pothole, hump and crack detection with severity scoring.
        - **Analytics** — explore detection trends, severity distribution and the
          most severe locations.
        - **Map View** — visualise GPS-tagged anomalies on an interactive map.
        - **Reports** — generate and download automated maintenance PDF reports.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Detection model", "YOLOv11")
    col2.metric("Segmentation", "SAM2")
    col3.metric("Depth", "MiDaS v3.1")
    col4.metric("API", st.session_state["api_base_url"].replace("/api/v1", ""))

    st.divider()
    st.info(
        "Tip: Configure detection thresholds and the model in the sidebar of the "
        "**Live Detection** page.",
        icon="💡",
    )


if __name__ == "__main__":
    main()
