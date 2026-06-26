"""Custom HTML metric cards for the dashboard."""

from __future__ import annotations

import streamlit as st

_SEVERITY_COLORS = {
    "LOW": "#00a000",
    "MEDIUM": "#d9a300",
    "HIGH": "#ff8c00",
    "CRITICAL": "#e00000",
}


def metric_card(title: str, value: str, color: str = "#1f3a5f") -> str:
    """Return HTML for a single coloured metric card.

    Args:
        title: Card label.
        value: Card value text.
        color: Accent colour (hex).

    Returns:
        An HTML string for the card.
    """
    return f"""
    <div style="
        background: linear-gradient(135deg, {color} 0%, {color}cc 100%);
        border-radius: 12px; padding: 18px; color: white; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);">
        <div style="font-size: 13px; opacity: 0.85;">{title}</div>
        <div style="font-size: 30px; font-weight: 700; margin-top: 4px;">{value}</div>
    </div>
    """


def render_metric_row(metrics: dict[str, str]) -> None:
    """Render a row of metric cards.

    Args:
        metrics: Mapping of label -> value text.
    """
    cols = st.columns(len(metrics))
    for col, (title, value) in zip(cols, metrics.items(), strict=False):
        with col:
            st.markdown(metric_card(title, value), unsafe_allow_html=True)


def severity_badge(level: str) -> str:
    """Return an inline HTML badge for a severity level."""
    color = _SEVERITY_COLORS.get(level.upper(), "#555")
    return (
        f"<span style='background:{color};color:white;padding:2px 8px;"
        f"border-radius:6px;font-size:12px;'>{level.upper()}</span>"
    )
