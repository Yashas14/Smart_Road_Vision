"""Analytics page: detection trends, severity distribution and top locations."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

API_BASE = st.session_state.get("api_base_url", "http://localhost:8000/api/v1")

st.title("📊 Analytics")
st.caption("Detection trends, severity breakdown and severity hotspots.")


@st.cache_data(ttl=30)
def _fetch_detections(limit: int = 200) -> list[dict]:
    """Fetch recent detections from the API (cached briefly)."""
    try:
        resp = requests.get(f"{API_BASE}/detect", params={"limit": limit}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.warning(f"Could not load detections: {exc}")
        return []


records = _fetch_detections()

if not records:
    st.info("No detection data available yet. Run some detections first.", icon="📭")
    st.stop()

df = pd.DataFrame(records)
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

# --- Top KPI row ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total detections", f"{len(df):,}")
col2.metric("Total anomalies", f"{int(df['anomaly_count'].sum()):,}")
col3.metric("Avg road score", f"{df['road_condition_score'].mean():.0f}/100")
col4.metric("Worst road score", f"{df['road_condition_score'].min():.0f}/100")

st.divider()

# --- Time series: anomalies per hour ---
left, right = st.columns(2)
with left:
    st.subheader("Anomalies over time")
    ts = (
        df.dropna(subset=["created_at"])
        .set_index("created_at")["anomaly_count"]
        .resample("1h")
        .sum()
        .reset_index()
    )
    if ts.empty:
        st.caption("Not enough timestamped data.")
    else:
        fig = px.area(
            ts, x="created_at", y="anomaly_count", labels={"anomaly_count": "anomalies"}
        )
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320)
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Road condition score distribution")
    fig2 = px.histogram(df, x="road_condition_score", nbins=20)
    fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- Source breakdown donut ---
left2, right2 = st.columns(2)
with left2:
    st.subheader("Detections by source")
    src_counts = df["source"].value_counts().reset_index()
    src_counts.columns = ["source", "count"]
    fig3 = px.pie(src_counts, names="source", values="count", hole=0.55)
    fig3.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320)
    st.plotly_chart(fig3, use_container_width=True)

with right2:
    st.subheader("Top-10 most severe segments")
    worst = df.nsmallest(10, "road_condition_score")[
        ["id", "source", "anomaly_count", "road_condition_score", "created_at"]
    ]
    st.dataframe(worst, use_container_width=True, hide_index=True)
