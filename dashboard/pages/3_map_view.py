"""Map View page: geospatial map of GPS-tagged anomalies."""

from __future__ import annotations

import folium
import requests
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(page_title="Map View", page_icon="🗺️", layout="wide")

API_BASE = st.session_state.get("api_base_url", "http://localhost:8000/api/v1")

st.title("🗺️ Geospatial Map")
st.caption("GPS-tagged anomalies, colour-coded by road condition.")

_SCORE_COLOR = [
    (80, "green"),
    (60, "orange"),
    (40, "red"),
    (0, "darkred"),
]


def _color_for_score(score: float) -> str:
    """Map a road condition score to a marker colour."""
    for threshold, color in _SCORE_COLOR:
        if score >= threshold:
            return color
    return "darkred"


@st.cache_data(ttl=30)
def _fetch_geotagged(limit: int = 500) -> list[dict]:
    """Fetch detections that carry GPS coordinates."""
    try:
        resp = requests.get(f"{API_BASE}/detect", params={"limit": limit}, timeout=30)
        resp.raise_for_status()
        return [
            r for r in resp.json()
            if r.get("latitude") is not None and r.get("longitude") is not None
        ]
    except requests.RequestException as exc:
        st.warning(f"Could not load detections: {exc}")
        return []


records = _fetch_geotagged()

if not records:
    st.info(
        "No GPS-tagged detections found. Upload images containing EXIF GPS data "
        "or provide latitude/longitude when detecting.",
        icon="📍",
    )
    st.stop()

center = [records[0]["latitude"], records[0]["longitude"]]
fmap = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
cluster = MarkerCluster().add_to(fmap)

for r in records:
    color = _color_for_score(r["road_condition_score"])
    popup = folium.Popup(
        html=(
            f"<b>Detection #{r['id']}</b><br>"
            f"Source: {r['source']}<br>"
            f"Anomalies: {r['anomaly_count']}<br>"
            f"Road score: {r['road_condition_score']:.0f}/100"
        ),
        max_width=250,
    )
    folium.Marker(
        location=[r["latitude"], r["longitude"]],
        popup=popup,
        icon=folium.Icon(color=color, icon="exclamation-triangle", prefix="fa"),
    ).add_to(cluster)

st.markdown(
    "**Legend:** 🟢 good (≥80) · 🟠 fair (≥60) · 🔴 poor (≥40) · ⚫ critical (<40)"
)
st_folium(fmap, use_container_width=True, height=600)
