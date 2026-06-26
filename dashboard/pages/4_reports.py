"""Reports page: generate, preview and download maintenance PDF reports."""

from __future__ import annotations

import base64
from datetime import datetime, time

import requests
import streamlit as st

st.set_page_config(page_title="Reports", page_icon="📄", layout="wide")

API_BASE = st.session_state.get("api_base_url", "http://localhost:8000/api/v1")

st.title("📄 Maintenance Reports")
st.caption("Generate and download automated road-condition PDF reports.")

with st.form("report_form"):
    title = st.text_input("Report title", value="Road Condition Report")
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("From date", value=None)
    with col2:
        date_to = st.date_input("To date", value=None)
    severity = st.selectbox(
        "Severity filter", options=["(any)", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    )
    include_gallery = st.checkbox("Include annotated image gallery", value=True)
    submitted = st.form_submit_button("📝 Generate report", type="primary")


def _iso(date_value: object) -> str | None:
    """Convert a Streamlit date input to an ISO timestamp string."""
    if not date_value:
        return None
    return datetime.combine(date_value, time.min).isoformat()


if submitted:
    payload = {
        "title": title,
        "date_from": _iso(date_from),
        "date_to": _iso(date_to),
        "severity": None if severity == "(any)" else severity,
        "include_gallery": include_gallery,
    }
    with st.spinner("Generating PDF report..."):
        try:
            resp = requests.post(
                f"{API_BASE}/reports/generate", json=payload, timeout=120
            )
            resp.raise_for_status()
            report = resp.json()
        except requests.RequestException as exc:
            st.error(f"Report generation failed: {exc}")
            report = None

    if report:
        st.success(f"Report #{report['report_id']} generated.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total anomalies", report["total_anomalies"])
        c2.metric("Avg road score", f"{report['avg_road_score']:.0f}/100")
        c3.metric("Status", report["status"])

        try:
            pdf_resp = requests.get(
                f"{API_BASE}/reports/{report['report_id']}/download", timeout=60
            )
            pdf_resp.raise_for_status()
            pdf_bytes = pdf_resp.content

            st.download_button(
                "⬇️ Download PDF",
                data=pdf_bytes,
                file_name=f"road_report_{report['report_id']}.pdf",
                mime="application/pdf",
            )

            b64 = base64.b64encode(pdf_bytes).decode("ascii")
            st.markdown(
                f'<iframe src="data:application/pdf;base64,{b64}" '
                f'width="100%" height="700px"></iframe>',
                unsafe_allow_html=True,
            )
        except requests.RequestException as exc:
            st.warning(f"Could not load PDF preview: {exc}")
