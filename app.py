"""Streamlit dashboard for separate Online Retail II and Olist segmentations."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    ENV_PATH = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=ENV_PATH, override=True)
except ImportError:
    pass

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from configs import DATASETS
from llm_advisor import get_segment_advice
from utils import artifact_paths, artifacts_ready, clean_numeric_frame, load_artifacts, read_json


st.set_page_config(page_title="E-Commerce Customer Loyalty Analytics", layout="wide")

APPLE_PALETTE_LIGHT = ["#0071E3", "#34C759", "#FF9500", "#AF52DE", "#5856D6", "#5AC8FA", "#FF2D55"]
APPLE_PALETTE_DARK = ["#0A84FF", "#30D158", "#FF9F0A", "#BF5AF2", "#5E5CE6", "#64D2FF", "#FF375F"]
APPLE_PALETTE = APPLE_PALETTE_LIGHT


def get_apple_palette() -> list[str]:
    theme = st.session_state.get("appearance_mode_select", st.session_state.get("app_theme", "Light"))
    return APPLE_PALETTE_DARK if theme == "Dark" else APPLE_PALETTE_LIGHT


def get_theme_css(theme_mode: str = "Light") -> str:
    """Generate dynamic CSS variables and layout styles for Light, Dark, and System themes."""
    if theme_mode == "Dark":
        vars_css = """
        :root {
            --bg-primary: #0E1117;
            --bg-surface: #1E212A;
            --bg-surface-secondary: #262A34;
            --bg-header: rgba(14, 17, 23, 0.85);
            --text-primary: #F5F5F7;
            --text-secondary: #A1A1A6;
            --border-color: #2F3440;
            --border-subtle: #3D4452;
            --accent-blue: #0A84FF;
            --accent-blue-hover: #409CFF;
            --accent-blue-active: #0071E3;
            --accent-blue-light: rgba(10, 132, 255, 0.15);
            --card-shadow: 0 4px 16px rgba(0, 0, 0, 0.45);
            --input-bg: #16181D;
            --hover-bg: #2C313C;
        }
        """
    elif theme_mode == "System":
        vars_css = """
        :root {
            --bg-primary: #F5F5F7;
            --bg-surface: #FFFFFF;
            --bg-surface-secondary: #FAFAFC;
            --bg-header: rgba(245, 245, 247, 0.85);
            --text-primary: #1D1D1F;
            --text-secondary: #86868B;
            --border-color: #E5E5EA;
            --border-subtle: #D2D2D7;
            --accent-blue: #0071E3;
            --accent-blue-hover: #0077ED;
            --accent-blue-active: #0062C4;
            --accent-blue-light: rgba(0, 113, 227, 0.08);
            --card-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
            --input-bg: #FFFFFF;
            --hover-bg: #ECECEE;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --bg-primary: #0E1117;
                --bg-surface: #1E212A;
                --bg-surface-secondary: #262A34;
                --bg-header: rgba(14, 17, 23, 0.85);
                --text-primary: #F5F5F7;
                --text-secondary: #A1A1A6;
                --border-color: #2F3440;
                --border-subtle: #3D4452;
                --accent-blue: #0A84FF;
                --accent-blue-hover: #409CFF;
                --accent-blue-active: #0071E3;
                --accent-blue-light: rgba(10, 132, 255, 0.15);
                --card-shadow: 0 4px 16px rgba(0, 0, 0, 0.45);
                --input-bg: #16181D;
                --hover-bg: #2C313C;
            }
        }
        """
    else:  # Light default
        vars_css = """
        :root {
            --bg-primary: #F5F5F7;
            --bg-surface: #FFFFFF;
            --bg-surface-secondary: #FAFAFC;
            --bg-header: rgba(245, 245, 247, 0.85);
            --text-primary: #1D1D1F;
            --text-secondary: #86868B;
            --border-color: #E5E5EA;
            --border-subtle: #D2D2D7;
            --accent-blue: #0071E3;
            --accent-blue-hover: #0077ED;
            --accent-blue-active: #0062C4;
            --accent-blue-light: rgba(0, 113, 227, 0.08);
            --card-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
            --input-bg: #FFFFFF;
            --hover-bg: #ECECEE;
        }
        """

    return f"""
    <style>
    /* =====================================================================
       Apple Design System - Dynamic Light & Dark Theme
       ===================================================================== */
    {vars_css}

    /* 1. Global Box-Sizing & Viewport Reset */
    *, *::before, *::after {{
        box-sizing: border-box !important;
    }}

    html, body {{
        margin: 0 !important;
        padding: 0 !important;
        background-color: var(--bg-primary) !important;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Helvetica Neue", Helvetica, Arial, sans-serif !important;
        color: var(--text-primary) !important;
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
        overflow-x: hidden !important;
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: var(--bg-primary) !important;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Helvetica Neue", Helvetica, Arial, sans-serif !important;
        color: var(--text-primary) !important;
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
        overflow-x: hidden !important;
        width: 100% !important;
        max-width: 100vw !important;
    }}

    [data-testid="stAppViewBlockContainer"] {{
        overflow-x: hidden !important;
    }}

    .main, [data-testid="stMain"] {{
        background-color: var(--bg-primary) !important;
        overflow-x: hidden !important;
        width: 100% !important;
    }}

    [data-testid="stHeader"] {{
        background: var(--bg-header) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid var(--border-color) !important;
    }}

    /* Main Content Container - balanced padding & responsive bounds */
    .main .block-container,
    [data-testid="stMainBlockContainer"],
    [data-testid="block-container"] {{
        padding-top: 1.5rem !important;
        padding-bottom: 4rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1360px !important;
        width: 100% !important;
        margin-left: auto !important;
        margin-right: auto !important;
        box-sizing: border-box !important;
    }}

    div[data-testid="column"] {{
        min-width: 0 !important;
        box-sizing: border-box !important;
    }}

    [data-testid="InputInstructions"] {{
        display: none !important;
    }}

    div[data-testid="stVerticalBlock"] {{
        transition: opacity 0.15s ease-in-out;
    }}

    /* 2. Headings & Typography */
    h1, h2, h3, h4, h5, h6,
    [data-testid="stHeadingWithActionElements"] h1,
    [data-testid="stHeadingWithActionElements"] h2,
    [data-testid="stHeadingWithActionElements"] h3 {{
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", Helvetica, Arial, sans-serif !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.022em !important;
        font-weight: 600 !important;
    }}

    h1, [data-testid="stHeadingWithActionElements"] h1 {{
        font-size: 28px !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
        margin-bottom: 6px !important;
    }}

    h2, [data-testid="stHeadingWithActionElements"] h2 {{
        font-size: 22px !important;
        margin-top: 28px !important;
        margin-bottom: 14px !important;
        font-weight: 600 !important;
    }}

    h3, [data-testid="stHeadingWithActionElements"] h3 {{
        font-size: 19px !important;
        margin-top: 22px !important;
        margin-bottom: 12px !important;
        font-weight: 600 !important;
    }}

    h4 {{
        font-size: 15px !important;
        margin-top: 14px !important;
        margin-bottom: 8px !important;
    }}

    h5 {{
        font-size: 12px !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        margin-top: 18px !important;
        margin-bottom: 10px !important;
    }}

    p, li, label {{
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif !important;
        color: var(--text-primary);
        line-height: 1.5;
    }}

    span:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded):not(.material-symbols-sharp):not(.material-symbols-outlined):not([class*="material-symbols"]):not(.stIcon) {{
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}

    [data-testid="stIconMaterial"],
    .material-symbols-rounded,
    .material-symbols-sharp,
    .material-symbols-outlined,
    .stIcon,
    [class*="material-symbols"] {{
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons", sans-serif !important;
        font-weight: normal !important;
        font-style: normal !important;
        line-height: 1 !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        word-wrap: normal !important;
        white-space: nowrap !important;
        direction: ltr !important;
        -webkit-font-smoothing: antialiased !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        vertical-align: middle !important;
    }}

    .stCaption, [data-testid="stCaptionContainer"] {{
        color: var(--text-secondary) !important;
        font-size: 13px !important;
        line-height: 1.45 !important;
        letter-spacing: -0.005em !important;
    }}

    hr {{
        border: none !important;
        border-top: 1px solid var(--border-color) !important;
        margin: 30px 0 !important;
    }}

    /* 3. KPI Metric Cards */
    div[data-testid="stMetric"] {{
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 14px !important;
        padding: 14px 18px !important;
        box-shadow: var(--card-shadow) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        min-height: 88px !important;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        overflow: hidden !important;
    }}

    div[data-testid="stMetric"]:hover {{
        border-color: var(--border-subtle) !important;
        transform: translateY(-1px) !important;
    }}

    div[data-testid="stMetricLabel"] {{
        color: var(--text-secondary) !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        letter-spacing: -0.01em !important;
        margin-bottom: 4px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: var(--text-primary) !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }}

    /* 4. Chart Containers */
    div[data-testid="stPlotlyChart"] {{
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 14px !important;
        padding: 12px 14px !important;
        box-shadow: var(--card-shadow) !important;
        margin-bottom: 20px !important;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
    }}

    div[data-testid="stPlotlyChart"] > div,
    div[data-testid="stPlotlyChart"] .js-plotly-plot,
    div[data-testid="stPlotlyChart"] .plot-container {{
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }}

    div[data-testid="stPlotlyChart"] .main-svg {{
        max-width: 100% !important;
        border-radius: 10px !important;
    }}

    /* 5. DataTables & DataFrames */
    div[data-testid="stDataFrame"] {{
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 14px !important;
        overflow: hidden !important;
        box-shadow: var(--card-shadow) !important;
        margin-bottom: 20px !important;
        padding: 4px !important;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }}

    /* 6. Expanders / Accordions */
    div[data-testid="stExpander"] {{
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 14px !important;
        overflow: hidden !important;
        box-shadow: var(--card-shadow) !important;
        margin-bottom: 16px !important;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }}

    div[data-testid="stExpander"] summary {{
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        font-size: 14px !important;
        padding: 14px 18px !important;
        border-radius: 14px !important;
        transition: color 0.15s ease !important;
        display: flex !important;
        align-items: center !important;
        box-sizing: border-box !important;
    }}

    div[data-testid="stExpander"] summary:hover {{
        color: var(--accent-blue) !important;
        background-color: var(--bg-surface-secondary) !important;
    }}

    div[data-testid="stExpander"] summary [data-testid="stIconMaterial"],
    div[data-testid="stExpander"] summary .material-symbols-rounded {{
        margin-right: 8px !important;
        font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
        font-weight: normal !important;
        font-size: 20px !important;
        flex-shrink: 0 !important;
    }}

    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
        padding: 14px 20px 20px 20px !important;
        border-top: 1px solid var(--border-color) !important;
        box-sizing: border-box !important;
    }}

    /* 7. Buttons */
    div.stButton > button,
    div.stDownloadButton > button {{
        background-color: var(--accent-blue) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        letter-spacing: -0.01em !important;
        padding: 8px 18px !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 1px 2px rgba(0, 113, 227, 0.2) !important;
        box-sizing: border-box !important;
    }}

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {{
        background-color: var(--accent-blue-hover) !important;
        box-shadow: 0 4px 12px rgba(0, 113, 227, 0.3) !important;
        transform: translateY(-1px) !important;
    }}

    div.stButton > button:active,
    div.stDownloadButton > button:active {{
        background-color: var(--accent-blue-active) !important;
        transform: translateY(0px) !important;
    }}

    div.stButton > button *,
    div.stDownloadButton > button * {{
        color: #FFFFFF !important;
    }}

    button[kind="secondary"] {{
        background-color: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
    }}

    /* 8. Sidebar Styling */
    div[data-testid="stSidebar"] {{
        background-color: var(--bg-surface-secondary) !important;
        border-right: 1px solid var(--border-color) !important;
        min-width: 260px !important;
        max-width: 300px !important;
        box-sizing: border-box !important;
    }}

    div[data-testid="stSidebar"] [data-testid="stRadio"] label {{
        padding: 8px 14px !important;
        border-radius: 8px !important;
        transition: all 0.15s ease !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        color: var(--text-primary) !important;
        margin-bottom: 2px !important;
    }}

    div[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
        background-color: var(--hover-bg) !important;
    }}

    div[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
        background-color: var(--accent-blue-light) !important;
        color: var(--accent-blue) !important;
        font-weight: 600 !important;
    }}

    div[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"]:checked {{
        accent-color: var(--accent-blue) !important;
    }}

    /* 9. Form Inputs & Selectboxes */
    div[data-baseweb="select"] > div {{
        background-color: var(--input-bg) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        font-size: 13.5px !important;
        color: var(--text-primary) !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
        box-sizing: border-box !important;
    }}

    div[data-baseweb="select"] > div:hover {{
        border-color: var(--accent-blue) !important;
    }}

    div[data-baseweb="select"] > div:focus-within {{
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 3px var(--accent-blue-light) !important;
    }}

    div[data-baseweb="select"] * {{
        color: var(--text-primary) !important;
    }}

    /* 10. Clean Alerts */
    div[data-testid="stAlert"] {{
        border-radius: 12px !important;
        font-size: 13px !important;
        border: 1px solid transparent !important;
        padding: 12px 16px !important;
        margin-bottom: 16px !important;
        box-sizing: border-box !important;
    }}

    div[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {{
        background-color: rgba(52, 199, 89, 0.12) !important;
        border-color: rgba(52, 199, 89, 0.3) !important;
        color: var(--text-primary) !important;
    }}

    div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {{
        background-color: rgba(255, 149, 0, 0.12) !important;
        border-color: rgba(255, 149, 0, 0.3) !important;
        color: var(--text-primary) !important;
    }}

    div[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {{
        background-color: rgba(0, 113, 227, 0.10) !important;
        border-color: rgba(0, 113, 227, 0.25) !important;
        color: var(--text-primary) !important;
    }}

    div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {{
        background-color: rgba(255, 45, 85, 0.12) !important;
        border-color: rgba(255, 45, 85, 0.3) !important;
        color: var(--text-primary) !important;
    }}

    /* 11. Border Containers (Cards) */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 14px !important;
        padding: 18px 22px !important;
        box-shadow: var(--card-shadow) !important;
        margin-bottom: 16px !important;
        box-sizing: border-box !important;
    }}

    /* 12. File Uploader */
    div[data-testid="stFileUploader"] section {{
        background-color: var(--bg-surface) !important;
        border: 1.5px dashed var(--border-subtle) !important;
        border-radius: 14px !important;
        padding: 24px !important;
        transition: all 0.2s ease !important;
        box-sizing: border-box !important;
    }}
    div[data-testid="stFileUploader"] section:hover {{
        border-color: var(--accent-blue) !important;
        background-color: var(--accent-blue-light) !important;
    }}
    div[data-testid="stFileUploader"] section * {{
        color: var(--text-primary) !important;
    }}
    div[data-testid="stFileUploader"] [data-testid="stIconMaterial"],
    div[data-testid="stFileUploader"] .material-symbols-rounded {{
        font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
        font-size: 24px !important;
    }}
    </style>
    """


# Inject active theme stylesheet
current_theme = st.session_state.get("appearance_mode_select", st.session_state.get("app_theme", "Light"))
st.markdown(get_theme_css(current_theme), unsafe_allow_html=True)


def show_chart(fig) -> None:
    theme_mode = st.session_state.get("appearance_mode_select", st.session_state.get("app_theme", "Light"))
    is_dark = (theme_mode == "Dark")

    bg_color = "#1E212A" if is_dark else "#FFFFFF"
    text_color = "#F5F5F7" if is_dark else "#1D1D1F"
    subtext_color = "#A1A1A6" if is_dark else "#86868B"
    grid_color = "#2A2F3D" if is_dark else "#F5F5F7"
    line_color = "#2F3440" if is_dark else "#E5E5EA"

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        margin=dict(l=15, r=15, t=48, b=25),
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif",
            color=text_color,
            size=12,
        ),
        title_font=dict(
            family="-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Helvetica Neue', Helvetica, Arial, sans-serif",
            color=text_color,
            size=15,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=text_color, size=11),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1,
        linecolor=line_color,
        linewidth=1,
        tickfont=dict(color=subtext_color, size=11),
        title_font=dict(color=subtext_color, size=12),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1,
        linecolor=line_color,
        linewidth=1,
        tickfont=dict(color=subtext_color, size=11),
        title_font=dict(color=subtext_color, size=12),
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


@st.cache_data
def load_table(dataset: str, name: str) -> pd.DataFrame:
    path = artifact_paths(dataset)[name]
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_metadata(dataset: str) -> tuple[dict, dict]:
    paths = artifact_paths(dataset)
    return read_json(paths["metrics"], {}), read_json(paths["quality"], {})


@st.cache_resource
def model_bundle(dataset: str):
    return load_artifacts(dataset)


def training_notice(dataset: str) -> bool:
    if artifacts_ready(dataset):
        return False
    st.warning(f"No trained artifacts for {DATASETS[dataset].label}.")
    st.code(f"python run_all.py --dataset {dataset}", language="bash")
    return True


@st.cache_data
def get_overview_kpis(dataset: str) -> dict[str, Any]:
    assignments = load_table(dataset, "assignments")
    cust_cnt = len(assignments)
    repeat_cnt = int((assignments["Frequency"] > 1).sum()) if "Frequency" in assignments.columns else 0
    repeat_rate = float(repeat_cnt / cust_cnt) if cust_cnt > 0 else 0.0
    tot_mon = float(assignments["Monetary"].sum()) if "Monetary" in assignments.columns else 0.0
    return {
        "customers": cust_cnt,
        "total_monetary": tot_mon,
        "repeat_customers": repeat_cnt,
        "repeat_rate": repeat_rate,
    }


@st.cache_data
def get_segment_counts(dataset: str) -> pd.DataFrame:
    assignments = load_table(dataset, "assignments")
    if assignments.empty:
        return pd.DataFrame(columns=["Segment_Name", "Customers", "Percentage", "Label"])
    counts = assignments.groupby("Segment_Name").size().reset_index(name="Customers")
    total = counts["Customers"].sum()
    counts["Percentage"] = (counts["Customers"] / total) * 100
    counts["Label"] = counts.apply(lambda r: f"{r['Customers']:,} ({r['Percentage']:.1f}%)", axis=1)
    return counts.sort_values(by="Customers", ascending=False).reset_index(drop=True)


def overview(
    assignments: pd.DataFrame,
    metrics: dict,
    quality: dict,
    dataset: str | None = None,
    profiles: pd.DataFrame | None = None,
) -> None:
    st.subheader("Business Overview")

    # Currency and Dataset Context
    is_olist = (dataset == "olist")
    curr_sym = "R$ " if is_olist else "£"
    curr_code = "BRL" if is_olist else "GBP"
    curr_name = "Brazilian Real" if is_olist else "British Pound"

    if dataset:
        kpis = get_overview_kpis(dataset)
        cust_cnt = kpis["customers"]
        tot_mon = kpis["total_monetary"]
        rep_cnt = kpis["repeat_customers"]
        rep_rate = kpis["repeat_rate"]
        counts = get_segment_counts(dataset)
    else:
        cust_cnt = len(assignments)
        tot_mon = float(assignments["Monetary"].sum()) if "Monetary" in assignments.columns else 0.0
        rep_cnt = int((assignments["Frequency"] > 1).sum()) if "Frequency" in assignments.columns else 0
        rep_rate = float(rep_cnt / cust_cnt) if cust_cnt > 0 else 0.0
        counts = (
            assignments.groupby("Segment_Name").size().reset_index(name="Customers")
            if not assignments.empty and "Segment_Name" in assignments.columns
            else pd.DataFrame(columns=["Segment_Name", "Customers"])
        )
        if not counts.empty:
            total = counts["Customers"].sum()
            counts["Percentage"] = (counts["Customers"] / total) * 100
            counts["Label"] = counts.apply(lambda r: f"{r['Customers']:,} ({r['Percentage']:.1f}%)", axis=1)
            counts = counts.sort_values(by="Customers", ascending=False).reset_index(drop=True)

    # 1. KPI Cards with Currency & Explanatory Tooltips
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Total Customers",
        f"{cust_cnt:,}",
        help="Total distinct customers transacting within the analytical observation window.",
    )
    c2.metric(
        f"Total Monetary ({curr_code})",
        f"{curr_sym}{tot_mon:,.0f}",
        help=f"Aggregate net spend across non-cancelled transactions ({curr_name}, {curr_code}).",
    )
    c3.metric(
        "Repeat Customers",
        f"{rep_cnt:,} ({rep_rate:.1%})",
        help="Customers with 2 or more distinct orders in the dataset window.",
    )

    sel_k = metrics.get("selected_k", 3)
    sel_algo = metrics.get("selected_algorithm", "K-Means")
    sel_cand = metrics.get("selected_candidate", {})
    sel_sil = sel_cand.get("silhouette")
    sil_str = f" (Holdout Silhouette: {sel_sil:.4f})" if isinstance(sel_sil, (int, float)) else ""
    c4.metric(
        "Selected Model (K)",
        f"K = {sel_k} ({sel_algo})",
        help=f"Optimal clustering model selected by unsupervised validation criteria{sil_str}.",
    )

    # 2. Segment Distribution Chart (Responsive, no clipped labels, count + percentage)
    if not counts.empty:
        fig_dist = px.bar(
            counts,
            x="Segment_Name",
            y="Customers",
            color="Segment_Name",
            text="Label",
            title="Customer Distribution by Segment",
            labels={"Segment_Name": "Customer Segment", "Customers": "Customer Count"},
            color_discrete_sequence=get_apple_palette(),
            custom_data=["Percentage"],
        )
        fig_dist.update_traces(
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>Customer Count: %{y:,}<br>Population Share: %{customdata[0]:.1f}%<extra></extra>",
        )
        fig_dist.update_layout(
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=40),
            xaxis=dict(tickangle=0, tickfont=dict(size=12)),
        )
        show_chart(fig_dist)

    # 3. Actionable Customer Insights Section (Distinguishing Measured Facts vs Hypotheses)
    if profiles is None and dataset:
        profiles = load_table(dataset, "profiles")

    if profiles is not None and not profiles.empty and "Segment_Name" in profiles.columns:
        st.markdown("---")
        st.subheader("Actionable Customer Insights")
        st.caption("Empirical findings measured directly from active cluster distributions, paired with testable business hypotheses.")

        largest_row = profiles.loc[profiles["Customer_Count"].idxmax()]
        largest_name = str(largest_row["Segment_Name"])
        largest_count = int(largest_row["Customer_Count"])
        largest_share = (largest_count / cust_cnt) * 100 if cust_cnt else 0

        top_mon_row = profiles.loc[profiles["Monetary"].idxmax()]
        top_mon_name = str(top_mon_row["Segment_Name"])
        top_mon_avg = float(top_mon_row["Monetary"])
        top_mon_count = int(top_mon_row["Customer_Count"])
        top_mon_share_rev = ((top_mon_count * top_mon_avg) / tot_mon) * 100 if tot_mon else 0

        at_risk_row = profiles.loc[profiles["Recency_Days"].idxmax()]
        at_risk_name = str(at_risk_row["Segment_Name"])
        at_risk_recency = float(at_risk_row["Recency_Days"])
        at_risk_count = int(at_risk_row["Customer_Count"])
        at_risk_share = (at_risk_count / cust_cnt) * 100 if cust_cnt else 0

        ins_c1, ins_c2, ins_c3 = st.columns(3)
        with ins_c1:
            with st.container(border=True):
                st.markdown(f"**Largest Base:** {largest_name}")
                st.markdown(
                    f"<div style='font-size:22px; font-weight:700; color:var(--text-primary); margin:4px 0;'>"
                    f"{largest_count:,} <span style='font-size:14px; font-weight:500; color:var(--text-secondary);'>({largest_share:.1f}% share)</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"**Measured Fact:** Core customer volume driver with average recency of "
                    f"{largest_row['Recency_Days']:.0f} days and frequency of {largest_row['Frequency']:.1f} orders."
                )

        with ins_c2:
            with st.container(border=True):
                st.markdown(f"**Top Spend Cohort:** {top_mon_name}")
                st.markdown(
                    f"<div style='font-size:22px; font-weight:700; color:var(--accent-blue); margin:4px 0;'>"
                    f"{curr_sym}{top_mon_avg:,.0f} <span style='font-size:14px; font-weight:500; color:var(--text-secondary);'>avg spend</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"**Measured Fact:** Comprises {top_mon_count:,} accounts ({ (top_mon_count/cust_cnt)*100 if cust_cnt else 0:.1f}% of base), "
                    f"yet generates ~{top_mon_share_rev:.1f}% of total monetary spend."
                )

        with ins_c3:
            with st.container(border=True):
                st.markdown(f"**Retention Priority:** {at_risk_name}")
                st.markdown(
                    f"<div style='font-size:22px; font-weight:700; color:#FF9500; margin:4px 0;'>"
                    f"{at_risk_recency:.0f} days <span style='font-size:14px; font-weight:500; color:var(--text-secondary);'>avg inactivity</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"**Measured Fact:** {at_risk_count:,} accounts ({at_risk_share:.1f}% of base) have exceeded normal repurchase intervals "
                    f"and require targeted re-engagement before permanent churn."
                )

        # Strategic Action & Testable Hypotheses Container
        with st.container(border=True):
            st.markdown("#### Strategic Actions & Testable Hypotheses")
            col_rec1, col_rec2 = st.columns(2)
            with col_rec1:
                st.markdown("**Repeat Purchase Expansion Opportunity**")
                repeat_note = (
                    "Wholesale and recurring multi-line orders drive substantial repeat engagement."
                    if not is_olist
                    else "Cross-category marketplace discovery represents an underutilized lever for single-order purchasers."
                )
                st.write(
                    f"- **Empirical Baseline (Fact):** Active repeat customer rate is **{rep_rate:.1%}** ({rep_cnt:,} customers).\n"
                    f"- **Context:** {repeat_note}\n"
                    f"- **Testable Hypothesis:** Introducing a targeted post-purchase replenishment prompt 14 days after initial delivery will increase 60-day repeat conversion by an estimated 2-4 percentage points."
                )
            with col_rec2:
                st.markdown("**Segment-Specific Loyalty & Retention Plan**")
                st.write(
                    f"- **High-Value Protection:** Safeguard *{top_mon_name}* with VIP concierge perks, priority order fulfillment, and loyalty tier access.\n"
                    f"  *Hypothesis:* Value-add non-monetary perks will sustain retention without degrading price realization.\n"
                    f"- **At-Risk Win-Back:** Deploy automated win-back workflows triggered prior to {at_risk_recency:.0f} days of inactivity.\n"
                    f"  *Hypothesis:* Dynamic replenishment reminders based on past browsing outperform generic discount codes."
                )

    # 4. Data Quality and Cleaning Rules Expander
    with st.expander("Data quality and cleaning rules"):
        q_c1, q_c2, q_c3, q_c4 = st.columns(4)
        raw_r = quality.get("raw_rows")
        clean_r = quality.get("clean_rows")
        rem_r = quality.get("rows_removed")
        snap_d = quality.get("snapshot_date", "N/A")

        with q_c1:
            st.metric("Raw Transaction Rows", f"{raw_r:,}" if isinstance(raw_r, (int, float)) else str(raw_r or "N/A"))
        with q_c2:
            st.metric("Clean Analytic Rows", f"{clean_r:,}" if isinstance(clean_r, (int, float)) else str(clean_r or "N/A"))
        with q_c3:
            st.metric("Rows Filtered / Excluded", f"{rem_r:,}" if isinstance(rem_r, (int, float)) else str(rem_r or "N/A"))
        with q_c4:
            st.metric("Snapshot Date", str(snap_d))

        sec_metrics = []
        if quality.get("missing_customer_rows"):
            sec_metrics.append(("Missing Customer IDs", f"{quality['missing_customer_rows']:,}"))
        if quality.get("duplicate_rows_removed"):
            sec_metrics.append(("Duplicates Removed", f"{quality['duplicate_rows_removed']:,}"))
        if quality.get("positive_sales_rows"):
            sec_metrics.append(("Positive Sales Lines", f"{quality['positive_sales_rows']:,}"))
        if quality.get("return_or_cancel_rows"):
            sec_metrics.append(("Returns / Cancellations", f"{quality['return_or_cancel_rows']:,}"))
        if quality.get("delivered_orders"):
            sec_metrics.append(("Delivered Orders", f"{quality['delivered_orders']:,}"))
        if quality.get("excluded_non_delivered_orders"):
            sec_metrics.append(("Non-Delivered Excluded", f"{quality['excluded_non_delivered_orders']:,}"))

        if sec_metrics:
            for i in range(0, len(sec_metrics), 4):
                chunk = sec_metrics[i : i + 4]
                cols = st.columns(len(chunk))
                for col, (label, val_str) in zip(cols, chunk):
                    col.metric(label, val_str)

        rules_text = quality.get("rules", "Standard data cleaning rules applied.")
        st.caption(f"**Cleaning Rules:** {rules_text}")


def profiles_page(profiles: pd.DataFrame, dataset: str = "unknown") -> None:
    st.subheader("Customer Segment Profiles")

    # Currency and Dataset Context
    is_olist = (dataset == "olist")
    curr_sym = "R$ " if is_olist else "£"

    if not profiles.empty and "Segment_Name" in profiles.columns:
        total_customers = profiles["Customer_Count"].sum() if "Customer_Count" in profiles.columns else len(profiles)
        st.markdown("##### Segment Archetypes & Behavioral Baselines")

        card_cols = st.columns(len(profiles))
        for col, (_, seg_row) in zip(card_cols, profiles.iterrows()):
            with col:
                with st.container(border=True):
                    seg_name = seg_row["Segment_Name"]
                    cnt = int(seg_row.get("Customer_Count", 0))
                    pct = (cnt / total_customers) * 100 if total_customers else 0
                    rec = seg_row.get("Recency_Days", 0)
                    freq = seg_row.get("Frequency", 0)
                    mon = seg_row.get("Monetary", 0)

                    st.markdown(f"**{seg_name}**")
                    st.markdown(
                        f"<div style='font-size:20px; font-weight:700; color:var(--text-primary); margin:2px 0 6px 0;'>"
                        f"{cnt:,} <span style='font-size:13px; font-weight:500; color:var(--text-secondary);'>({pct:.1f}%)</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"• **Recency:** {rec:.0f} days\n"
                        f"• **Frequency:** {freq:.1f} orders\n"
                        f"• **Monetary:** {curr_sym}{mon:,.0f} avg spend"
                    )

    st.markdown("##### Detailed Feature Profile Table")
    display_profiles = profiles.copy()
    if "Monetary" in display_profiles.columns:
        display_profiles["Monetary"] = display_profiles["Monetary"].apply(
            lambda v: f"{curr_sym}{v:,.2f}" if isinstance(v, (int, float)) else v
        )
    if "Average_Order_Value" in display_profiles.columns:
        display_profiles["Average_Order_Value"] = display_profiles["Average_Order_Value"].apply(
            lambda v: f"{curr_sym}{v:,.2f}" if isinstance(v, (int, float)) else v
        )
    if "Customer_Count" in display_profiles.columns:
        display_profiles["Customer_Count"] = display_profiles["Customer_Count"].apply(
            lambda v: f"{int(v):,}" if isinstance(v, (int, float)) else v
        )
    if "Recency_Days" in display_profiles.columns:
        display_profiles["Recency_Days"] = display_profiles["Recency_Days"].apply(
            lambda v: f"{v:.1f}" if isinstance(v, (int, float)) else v
        )
    if "Frequency" in display_profiles.columns:
        display_profiles["Frequency"] = display_profiles["Frequency"].apply(
            lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else v
        )

    st.dataframe(display_profiles, use_container_width=True)
    numeric = [column for column in ["Recency_Days", "Frequency", "Monetary", "Average_Order_Value", "Tenure_Days", "Return_Rate", "Unique_Products", "Category_Diversity"] if column in profiles]
    if numeric:
        long = profiles.melt(id_vars=["Segment_Name"], value_vars=numeric, var_name="Feature", value_name="Average")
        show_chart(px.bar(long, x="Feature", y="Average", color="Segment_Name", barmode="group", title="Segment Feature Averages", color_discrete_sequence=get_apple_palette()))

    if not profiles.empty and "Segment_Name" in profiles.columns:
        with st.expander("AI Segment Recommendations (Aggregate Analysis)", expanded=False):
            st.caption("Inspect aggregate behavioral recommendations per customer segment.")
            seg = st.selectbox("Segment", profiles["Segment_Name"], key="profile_seg_rec_select")
            profile = profiles.loc[profiles["Segment_Name"] == seg].iloc[0].to_dict()
            rec_key = f"profile_rec_{dataset}_{seg}"
            if st.button("Generate Recommendations", key="profile_gen_rec_btn"):
                with st.spinner("Analyzing segment data..."):
                    st.session_state[rec_key] = get_segment_advice(profile, dataset=dataset, allow_mock_fallback=True)
            if rec_key in st.session_state:
                rec = st.session_state[rec_key]
                if rec.get("mode") in ("gemini_live", "gemini", "live"):
                    st.success("🟢 LIVE GEMINI RESPONSE")
                elif "fallback" in rec.get("mode", "") or rec.get("mode") == "mock":
                    st.info("ℹ️ AI insights are temporarily operating in offline mode. Recommendations are based on available dataset metrics.")
                elif rec.get("mode") == "blocked":
                    st.error("🛡️ Request blocked by Privacy Guard.")
                elif rec.get("mode") == "error":
                    st.error("⚠️ Unable to generate recommendations. Please try again.")

                seg_title = rec.get("segment", seg)
                st.markdown(f"#### {seg_title}")
                if rec.get("summary"):
                    st.write(rec["summary"])

                drivers = rec.get("key_drivers", [])
                if drivers and isinstance(drivers, list):
                    st.markdown("##### Key Behavioral Drivers")
                    driver_rows = []
                    for d in drivers:
                        if isinstance(d, dict):
                            feat = d.get("feature", "N/A")
                            direction = str(d.get("direction", "N/A")).replace("_", " ").title()
                            val = d.get("value_cited", "N/A")
                            if isinstance(val, float):
                                val_str = f"{val:,.2f}"
                            elif isinstance(val, int):
                                val_str = f"{val:,}"
                            else:
                                val_str = str(val)
                            driver_rows.append({"Feature": feat, "Benchmark Direction": direction, "Cited Value": val_str})
                    if driver_rows:
                        st.dataframe(pd.DataFrame(driver_rows), use_container_width=True, hide_index=True)

                col_act, col_risk = st.columns(2)
                with col_act:
                    actions = rec.get("actions", [])
                    if actions:
                        st.markdown("##### Recommended Actions")
                        for act in actions:
                            st.markdown(f"- {act}")

                with col_risk:
                    risks = rec.get("risks", [])
                    if risks:
                        st.markdown("##### Key Risks & Watch-outs")
                        for rsk in risks:
                            st.markdown(f"- {rsk}")



def comparison_page(comparison: pd.DataFrame, pca: pd.DataFrame, metrics: dict, dataset: str | None = None) -> None:
    st.subheader("Clustering Comparison & Model Selection")

    # 1. Selected Algorithm and Core Metrics
    sel_algo = metrics.get("selected_algorithm", "K-Means")
    sel_k = metrics.get("selected_k", metrics.get("selected_candidate", {}).get("k", 3))
    sel_cand = metrics.get("selected_candidate", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Selected Algorithm", str(sel_algo))
    with col2:
        st.metric("Optimal Clusters (K)", f"K = {sel_k}")
    with col3:
        sil_val = sel_cand.get("silhouette")
        st.metric("Silhouette Score", f"{sil_val:.4f}" if isinstance(sil_val, (int, float)) else "N/A")
    with col4:
        db_val = sel_cand.get("davies_bouldin")
        st.metric("Davies-Bouldin Index", f"{db_val:.4f}" if isinstance(db_val, (int, float)) else "N/A")

    # 2. Performance & Stability Metrics
    ch_val = sel_cand.get("calinski_harabasz")
    mean_ari = sel_cand.get("mean_ari")
    min_ari = sel_cand.get("min_ari")
    stab_runs = sel_cand.get("stability_runs")

    if any(v is not None for v in [ch_val, mean_ari, stab_runs]):
        st.markdown("##### Model Quality & Stability")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Calinski-Harabasz", f"{ch_val:.1f}" if isinstance(ch_val, (int, float)) else "N/A")
        with m_col2:
            st.metric("Mean Stability (ARI)", f"{mean_ari:.4f}" if isinstance(mean_ari, (int, float)) else "N/A")
        with m_col3:
            st.metric("Min Stability (ARI)", f"{min_ari:.4f}" if isinstance(min_ari, (int, float)) else "N/A")
        with m_col4:
            st.metric("Stability Runs", f"{int(stab_runs)}" if isinstance(stab_runs, (int, float)) else "N/A")

    if metrics.get("selection_rule"):
        st.caption(f"**Selection Rule:** {metrics['selection_rule']}")

    # 3. Candidate Evaluation Across Algorithms & K
    if not comparison.empty:
        st.markdown("##### Candidate Evaluation Across Algorithms & K")
        fig_comp = px.line(
            comparison,
            x="k",
            y="silhouette",
            color="method",
            markers=True,
            title="Silhouette Score Across Methods and Cluster Counts (K)",
            labels={"k": "Number of Clusters (K)", "silhouette": "Silhouette Score", "method": "Algorithm"},
            color_discrete_sequence=get_apple_palette(),
        )
        show_chart(fig_comp)

        display_comp = comparison.copy()
        for col in ["silhouette", "davies_bouldin", "calinski_harabasz", "mean_ari"]:
            if col in display_comp.columns:
                display_comp[col] = display_comp[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) and isinstance(x, (int, float)) else x
                )
        st.dataframe(display_comp, use_container_width=True)

    # 4. Validation Metrics Section (Holdout Partition)
    val_metrics = metrics.get("validation_metrics")
    if val_metrics and isinstance(val_metrics, dict):
        st.markdown("---")
        st.subheader("Holdout Partition Validation")
        st.caption(metrics.get("validation_policy", "Evaluated on independent 20% holdout partition."))

        v_col1, v_col2, v_col3, v_col4 = st.columns(4)
        with v_col1:
            st.metric("Validation Partition", str(val_metrics.get("method", "Holdout Set")))
        with v_col2:
            v_sil = val_metrics.get("silhouette")
            st.metric("Holdout Silhouette", f"{v_sil:.4f}" if isinstance(v_sil, (int, float)) else "N/A")
        with v_col3:
            v_db = val_metrics.get("davies_bouldin")
            st.metric("Holdout Davies-Bouldin", f"{v_db:.4f}" if isinstance(v_db, (int, float)) else "N/A")
        with v_col4:
            v_ch = val_metrics.get("calinski_harabasz")
            st.metric("Holdout Calinski-Harabasz", f"{v_ch:.1f}" if isinstance(v_ch, (int, float)) else "N/A")

    # 5. PCA Cluster View
    if not pca.empty and "PCA1" in pca.columns and "PCA2" in pca.columns:
        st.markdown("---")
        st.subheader("PCA Cluster View")
        st.caption("2D projection of customer behavioral features via Principal Component Analysis.")

        if dataset:
            pca_plot_data = get_sampled_pca(dataset)
        else:
            pca_plot_data = pca.sample(n=5000, random_state=42) if len(pca) > 5000 else pca

        if len(pca) > 5000:
            st.caption(f"Displaying a representative sample of 5,000 customers (out of {len(pca):,}) for responsive rendering.")

        fig_pca = px.scatter(
            pca_plot_data,
            x="PCA1",
            y="PCA2",
            color="Segment_Name" if "Segment_Name" in pca_plot_data.columns else "Cluster",
            hover_data=["Cluster"] if "Cluster" in pca_plot_data.columns else None,
            title="Customer Segments Projected in 2D PCA Space",
            opacity=0.75,
            labels={"PCA1": "First Principal Component", "PCA2": "Second Principal Component", "Segment_Name": "Customer Segment"},
            color_discrete_sequence=get_apple_palette(),
        )
        show_chart(fig_pca)


@st.cache_data
def get_sampled_pca(dataset: str, max_points: int = 5000) -> pd.DataFrame:
    pca = load_table(dataset, "pca")
    if len(pca) > max_points:
        return pca.sample(n=max_points, random_state=42)
    return pca


@st.cache_data
def get_population_averages(dataset: str, available_cols: tuple[str, ...]) -> dict[str, float]:
    assignments = load_table(dataset, "assignments")
    return {col: float(assignments[col].mean()) for col in available_cols if col in assignments}


def explain_page(dataset: str, assignments: pd.DataFrame, profiles: pd.DataFrame) -> None:
    st.subheader("Customer-Level Segment Explanation")
    if assignments.empty:
        st.info("No customer assignment records available for explanation.")
        return
    config = DATASETS[dataset]
    id_col = config.customer_id_column
    id_sample = assignments[id_col].head(5000).astype(str).tolist()

    # Currency formatting context
    is_olist = (dataset == "olist")
    curr_sym = "R$ " if is_olist else "£"
    
    col_sel, _ = st.columns([2, 1])
    with col_sel:
        selected_id = st.selectbox(
            "Select Customer Identifier",
            id_sample,
            help=f"Select a customer record to view personal metrics vs. segment baseline ({id_col}).",
        )

    matching = assignments.loc[assignments[id_col].astype(str) == str(selected_id)]
    if matching.empty:
        return
    customer = matching.iloc[0]
    profile = profiles.loc[profiles["Cluster"] == customer["Cluster"]].iloc[0]

    with st.container(border=True):
        st.markdown(f"### {customer['Segment_Name']}")
        rec = profile.get("Recommendation", "")
        if rec:
            st.write(rec)

    available = [column for column in ["Recency_Days", "Frequency", "Monetary", "Average_Order_Value", "Tenure_Days", "Return_Rate", "Unique_Products", "Category_Diversity", "Average_Review_Score"] if column in assignments]
    pop_means = get_population_averages(dataset, tuple(available))

    def _format_val(feat: str, val: Any) -> str:
        if not isinstance(val, (int, float)) or pd.isna(val):
            return str(val)
        if "Monetary" in feat or "Value" in feat:
            return f"{curr_sym}{val:,.2f}"
        if "Rate" in feat:
            return f"{val:.1%}" if val <= 1.0 else f"{val:.2f}"
        if "Score" in feat:
            return f"{val:.2f} / 5.0"
        if "Days" in feat:
            return f"{val:.1f} days"
        if "Frequency" in feat:
            return f"{val:.1f} orders"
        return f"{val:,.2f}" if isinstance(val, float) else f"{val:,}"

    detail = pd.DataFrame({
        "Behavioral Feature": available,
        "Customer Value": [_format_val(col, customer[col]) for col in available],
        "Population Average Benchmark": [_format_val(col, pop_means.get(col, customer[col])) for col in available],
    })
    st.dataframe(detail, use_container_width=True, hide_index=True)


@st.cache_data(show_spinner=False)
def process_batch_inference(dataset: str, file_bytes: bytes) -> tuple[pd.DataFrame, bytes]:
    """Run batch customer segmentation with caching to eliminate rerun overhead."""
    if not file_bytes or len(file_bytes.strip()) == 0:
        raise ValueError("The uploaded file is empty. Please provide a CSV file containing customer feature columns.")
    if len(file_bytes) > 50 * 1024 * 1024:
        raise ValueError("Uploaded file exceeds 50MB limit. For large populations, please split into smaller batches.")
    bundle = model_bundle(dataset)
    required = bundle["feature_columns"]
    try:
        frame = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV file: {exc}")
    if len(frame) == 0:
        raise ValueError("The uploaded CSV contains headers but no customer data rows.")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))
    cleaned = clean_numeric_frame(frame, required)[required]
    transformed = bundle["preprocessor"].transform(cleaned)
    predictor = bundle.get("predictor", bundle["model"])
    labels = predictor.predict(transformed)
    result = frame.copy()
    result["Cluster"] = labels
    result["Segment_Name"] = [bundle["segment_names"][int(label)] for label in labels]
    csv_bytes = result.to_csv(index=False).encode("utf-8")
    return result, csv_bytes


def batch_page(dataset: str) -> None:
    st.subheader("Batch Assignment")
    bundle = model_bundle(dataset)
    required = bundle["feature_columns"]

    with st.container(border=True):
        st.markdown("#### Batch Customer Scoring")
        req_pills = " ".join([f"`{col}`" for col in required])
        st.caption(f"Upload a CSV containing these engineered behavioral features: {req_pills}")
        uploaded = st.file_uploader("Batch customer CSV", type=["csv"], key=f"batch_uploader_{dataset}")

    if uploaded is None:
        st.caption("Tip: Upload an unlabelled customer features CSV from your data pipeline to generate cluster assignments and segment names.")
        return

    try:
        with st.spinner("Processing batch assignments..."):
            file_bytes = uploaded.getvalue()
            result, csv_bytes = process_batch_inference(dataset, file_bytes)

        total_rows = len(result)
        cluster_count = result["Cluster"].nunique()

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Records Scored", f"{total_rows:,}")
        with m2:
            st.metric("Segments Assigned", f"{cluster_count}")
        with m3:
            algo_name = bundle.get("selected_algorithm", "K-Means")
            st.metric("Clustering Model", str(algo_name))

        if total_rows > 1000:
            st.caption(f"Displaying a responsive preview of the first 1,000 rows (out of {total_rows:,}). The complete dataset is available via download.")
            st.dataframe(result.head(1000), use_container_width=True)
        else:
            st.dataframe(result, use_container_width=True)

        st.download_button(
            "Download assignments",
            csv_bytes,
            "batch_assignments.csv",
            "text/csv",
            key="batch_download_btn",
        )
    except Exception as exc:
        st.error(f"Batch assignment failed: {exc}")


def render_floating_chat(dataset: str, profiles: pd.DataFrame) -> None:
    """Render a modern floating chatbot launcher and compact window fixed at bottom-right."""
    st.markdown(
        """
        <style>
        /* Floating Launcher Button - fixed at bottom right */
        div[data-testid="stVerticalBlock"] > div.st-key-chat_launcher_container,
        div.st-key-chat_launcher_container {
            position: fixed !important;
            bottom: 24px !important;
            right: 24px !important;
            z-index: 999990 !important;
            width: auto !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        .st-key-chat_launcher_btn button {
            width: 56px !important;
            height: 56px !important;
            border-radius: 50% !important;
            background: var(--accent-blue) !important;
            color: #FFFFFF !important;
            font-size: 22px !important;
            font-weight: 500 !important;
            border: none !important;
            box-shadow: 0 4px 16px rgba(0, 113, 227, 0.35) !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            cursor: pointer !important;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
            padding: 0 !important;
            line-height: 1 !important;
        }

        .st-key-chat_launcher_btn button * {
            color: #FFFFFF !important;
        }

        .st-key-chat_launcher_btn button:hover {
            transform: scale(1.06) !important;
            background: var(--accent-blue-hover) !important;
            box-shadow: 0 6px 22px rgba(0, 113, 227, 0.45) !important;
        }

        /* Floating Compact Chat Window - anchored above launcher */
        div[data-testid="stVerticalBlock"] > div.st-key-chat_window_container,
        div.st-key-chat_window_container {
            position: fixed !important;
            bottom: 92px !important;
            right: 24px !important;
            width: 390px !important;
            max-width: calc(100vw - 48px) !important;
            height: 560px !important;
            max-height: calc(100vh - 120px) !important;
            background-color: var(--bg-surface) !important;
            color: var(--text-primary) !important;
            border-radius: 18px !important;
            box-shadow: var(--card-shadow) !important;
            border: 1px solid var(--border-color) !important;
            z-index: 999995 !important;
            padding: 14px 14px 12px 14px !important;
            display: flex !important;
            flex-direction: column !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }

        /* Direct vertical block child inside chat window */
        div.st-key-chat_window_container > div[data-testid="stVerticalBlock"] {
            height: 100% !important;
            max-height: 100% !important;
            display: flex !important;
            flex-direction: column !important;
            gap: 6px !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }

        /* Enforce theme-aware readable text inside the chat window */
        div.st-key-chat_window_container,
        div.st-key-chat_window_container * {
            color: var(--text-primary);
        }

        div.st-key-chat_window_container p,
        div.st-key-chat_window_container span,
        div.st-key-chat_window_container li,
        div.st-key-chat_window_container strong,
        div.st-key-chat_window_container em {
            color: var(--text-primary) !important;
        }

        div.st-key-chat_window_container h3,
        div.st-key-chat_window_container .stMarkdown strong {
            color: var(--text-primary) !important;
            font-size: 15px !important;
        }

        div.st-key-chat_window_container .stCaption,
        div.st-key-chat_window_container small,
        div.st-key-chat_window_container [data-testid="stCaptionContainer"] {
            color: var(--text-secondary) !important;
            font-size: 11.5px !important;
            font-weight: 500 !important;
        }

        /* Close and Clear buttons in chat header */
        .st-key-chat_close_btn button,
        .st-key-chat_clear_btn button {
            background: var(--bg-primary) !important;
            border: 1px solid var(--border-color) !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            color: var(--text-secondary) !important;
            padding: 3px 8px !important;
            line-height: 1 !important;
            cursor: pointer !important;
            border-radius: 8px !important;
            box-shadow: none !important;
            transform: none !important;
            box-sizing: border-box !important;
        }

        .st-key-chat_close_btn button *,
        .st-key-chat_clear_btn button * {
            color: var(--text-secondary) !important;
        }

        .st-key-chat_close_btn button:hover,
        .st-key-chat_clear_btn button:hover {
            color: var(--text-primary) !important;
            background-color: var(--hover-bg) !important;
            box-shadow: none !important;
            transform: none !important;
        }

        .st-key-chat_close_btn button:hover *,
        .st-key-chat_clear_btn button:hover * {
            color: var(--text-primary) !important;
        }

        /* Dropdown select box inside chat */
        div.st-key-chat_window_container [data-baseweb="select"] {
            background-color: var(--input-bg) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 10px !important;
            margin-bottom: 2px !important;
            box-sizing: border-box !important;
        }

        div.st-key-chat_window_container [data-baseweb="select"] * {
            color: var(--text-primary) !important;
            font-size: 13px !important;
        }

        /* Scrollable messages box */
        div.st-key-chat_scroll_box {
            flex: 1 1 auto !important;
            min-height: 140px !important;
            overflow-y: auto !important;
            background-color: var(--bg-surface-secondary) !important;
            border-radius: 12px !important;
            border: 1px solid var(--border-color) !important;
            padding: 10px 12px !important;
            margin-bottom: 6px !important;
            box-sizing: border-box !important;
        }

        /* Chat message bubbles */
        div.st-key-chat_scroll_box [data-testid="stChatMessage"] {
            background-color: var(--bg-surface) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            padding: 8px 12px !important;
            margin-bottom: 8px !important;
            box-shadow: var(--card-shadow) !important;
            display: flex !important;
            flex-direction: row !important;
            align-items: flex-start !important;
            gap: 10px !important;
            box-sizing: border-box !important;
        }

        /* Avatar styling inside chat message */
        div.st-key-chat_scroll_box [data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] {
            flex-shrink: 0 !important;
            width: 28px !important;
            height: 28px !important;
            font-size: 15px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            border-radius: 50% !important;
            line-height: 1 !important;
        }

        /* Content container inside chat message */
        div.st-key-chat_scroll_box [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
            flex: 1 1 auto !important;
            min-width: 0 !important;
            box-sizing: border-box !important;
        }

        div.st-key-chat_scroll_box [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] p {
            color: var(--text-primary) !important;
            font-size: 13px !important;
            line-height: 1.45 !important;
            margin-top: 0 !important;
            margin-bottom: 6px !important;
        }

        div.st-key-chat_scroll_box [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] p:last-child {
            margin-bottom: 0 !important;
        }

        div.st-key-chat_scroll_box [data-testid="stChatMessage"] [data-testid="stCaptionContainer"],
        div.st-key-chat_scroll_box [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] * {
            color: var(--text-secondary) !important;
            font-size: 11px !important;
            font-weight: 500 !important;
        }

        /* Pinned bottom form container */
        div.st-key-chat_window_container div[data-testid="stForm"],
        div.st-key-chat_window_container form,
        div.st-key-floating_chat_form,
        .st-key-floating_chat_form {
            border: none !important;
            padding: 0 !important;
            margin: 0 !important;
            margin-top: auto !important;
            flex-shrink: 0 !important;
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
        }

        /* Palette overrides: high-contrast Apple theme replacing legacy #FF4B4B and #0f172a */
        /* Horizontal layout for input row */
        div.st-key-chat_window_container div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            align-items: center !important;
            width: 100% !important;
            gap: 10px !important;
            margin: 0 !important;
            padding: 0 !important;
            box-sizing: border-box !important;
        }

        /* Column sizing inside the chat form */
        div.st-key-chat_window_container div[data-testid="stForm"] div[data-testid="stColumn"] {
            min-width: 0 !important;
            box-sizing: border-box !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        div.st-key-chat_window_container div[data-testid="stForm"] div[data-testid="stColumn"]:first-child {
            flex: 1 1 auto !important;
            min-width: 0 !important;
            width: auto !important;
        }

        div.st-key-chat_window_container div[data-testid="stForm"] div[data-testid="stColumn"]:last-child {
            flex: 0 0 48px !important;
            min-width: 48px !important;
            max-width: 52px !important;
            width: 48px !important;
        }

        div.st-key-chat_window_container div[data-testid="stForm"] div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] {
            gap: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            display: flex !important;
            justify-content: center !important;
        }

        /* Text Input container and wrappers */
        div.st-key-chat_window_container div[data-testid="stTextInput"],
        div.st-key-floating_chat_input_field {
            width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            box-sizing: border-box !important;
        }

        div.st-key-chat_window_container div[data-testid="stTextInputRootElement"],
        div.st-key-chat_window_container div[data-baseweb="base-input"],
        div.st-key-chat_window_container div[data-baseweb="input"] {
            width: 100% !important;
            height: 40px !important;
            min-height: 40px !important;
            border-radius: 10px !important;
            background-color: var(--input-bg) !important;
            box-sizing: border-box !important;
        }

        div.st-key-chat_window_container div[data-testid="stTextInputRootElement"] {
            border: 1px solid var(--border-subtle) !important;
            transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
        }

        div.st-key-chat_window_container div[data-testid="stTextInputRootElement"]:focus-within {
            border-color: var(--accent-blue) !important;
            box-shadow: 0 0 0 2px var(--accent-blue-light) !important;
        }

        div.st-key-chat_window_container input[type="text"] {
            width: 100% !important;
            height: 100% !important;
            min-height: 36px !important;
            background-color: transparent !important;
            color: var(--text-primary) !important;
            border: none !important;
            padding: 6px 12px !important;
            font-size: 13.5px !important;
            box-sizing: border-box !important;
            outline: none !important;
        }

        div.st-key-chat_window_container input[type="text"]::placeholder {
            color: var(--text-secondary) !important;
            opacity: 1 !important;
        }

        /* Send button wrapper */
        div.st-key-chat_window_container div.st-key-chat_send_btn,
        div.st-key-chat_window_container div[data-testid="stFormSubmitButton"] {
            width: 100% !important;
            min-width: 44px !important;
            margin: 0 !important;
            padding: 0 !important;
            box-sizing: border-box !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        /* Send button in floating chat window */
        div.st-key-chat_window_container div.st-key-chat_send_btn button,
        div.st-key-chat_window_container div[data-testid="stFormSubmitButton"] button,
        div.st-key-chat_window_container div.st-key-chat_send_btn button:disabled,
        div.st-key-chat_window_container div[data-testid="stFormSubmitButton"] button:disabled,
        div.st-key-chat_window_container div.st-key-chat_send_btn button[disabled],
        div.st-key-chat_window_container div[data-testid="stFormSubmitButton"] button[disabled] {
            background-color: var(--accent-blue) !important;
            background: var(--accent-blue) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 10px !important;
            font-size: 15px !important;
            font-weight: 600 !important;
            height: 40px !important;
            min-height: 40px !important;
            min-width: 44px !important;
            max-width: 52px !important;
            width: 100% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            cursor: pointer !important;
            transition: all 0.15s ease !important;
            padding: 0 8px !important;
            margin: 0 !important;
            box-sizing: border-box !important;
            flex-shrink: 0 !important;
            box-shadow: 0 1px 3px rgba(0, 113, 227, 0.25) !important;
        }

        div.st-key-chat_window_container div.st-key-chat_send_btn button:hover,
        div.st-key-chat_window_container div[data-testid="stFormSubmitButton"] button:hover {
            background-color: var(--accent-blue-hover) !important;
            background: var(--accent-blue-hover) !important;
            box-shadow: 0 3px 8px rgba(0, 113, 227, 0.35) !important;
        }

        div.st-key-chat_window_container div.st-key-chat_send_btn button:focus,
        div.st-key-chat_window_container div[data-testid="stFormSubmitButton"] button:focus,
        div.st-key-chat_window_container div.st-key-chat_send_btn button:active,
        div.st-key-chat_window_container div[data-testid="stFormSubmitButton"] button:active {
            background-color: var(--accent-blue-active) !important;
            background: var(--accent-blue-active) !important;
            color: #FFFFFF !important;
            box-shadow: 0 0 0 2px var(--accent-blue-light) !important;
            outline: none !important;
        }

        /* Ensure Send button arrow icon (➤) is pure white across all states */
        div.st-key-chat_window_container div.st-key-chat_send_btn button *,
        div.st-key-chat_window_container div[data-testid="stFormSubmitButton"] button * {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
            stroke: #FFFFFF !important;
            opacity: 1 !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    is_open = st.session_state.get("chat_open", False)
    history = st.session_state.setdefault("floating_chat_history", [])

    # If open, render the compact floating window anchored above the launcher
    if is_open:
        with st.container(key="chat_window_container"):
            # Header with title, clear history, and close button (fixed at top)
            col_head, col_clear, col_close = st.columns([0.72, 0.14, 0.14])
            with col_head:
                st.markdown("**Loyalty AI Assistant**")
                from llm_advisor import circuit_breaker
                last_asst_msg = next((m for m in reversed(history) if m.get("role") == "assistant"), None)
                cb_state = circuit_breaker.state
                rem_cd = circuit_breaker.get_remaining_cooldown()

                if last_asst_msg and last_asst_msg.get("mode") in ("gemini_live", "gemini", "api", "live"):
                    hdr_status = "🟢 GEMINI (LIVE)"
                    caption_sub = None
                elif not os.getenv("GEMINI_API_KEY"):
                    hdr_status = "⚪ Offline Mode (Dataset Metrics)"
                    caption_sub = "ℹ️ *Operating in offline mode using verified dataset metrics.*"
                elif cb_state == "OPEN" and rem_cd > 0:
                    err_label = circuit_breaker.last_error_type.replace("_", " ").title()
                    hdr_status = f"⚪ Offline Fallback ({err_label})"
                    caption_sub = f"ℹ️ *Cooldown active ({rem_cd:.0f}s remaining). Using verified dataset metrics.*"
                elif os.getenv("GEMINI_API_KEY"):
                    hdr_status = "🟢 Gemini Ready"
                    caption_sub = None
                else:
                    hdr_status = "⚪ Offline Mode (Dataset Metrics)"
                    caption_sub = "ℹ️ *Operating in offline mode using verified dataset metrics.*"

                st.caption(f"{DATASETS[dataset].label} • {hdr_status}")
                if caption_sub:
                    st.caption(caption_sub)
            with col_clear:
                if st.button("🗑️", key="chat_clear_btn", help="Clear chat history"):
                    st.session_state["floating_chat_history"] = []
                    st.rerun()
            with col_close:
                if st.button("✕", key="chat_close_btn", help="Close Assistant"):
                    st.session_state["chat_open"] = False
                    st.rerun()

            # Optional segment context filter
            seg_options = ["All Segments (Overview)"]
            if not profiles.empty and "Segment_Name" in profiles.columns:
                seg_options += list(profiles["Segment_Name"].unique())
            active_seg = st.selectbox(
                "Segment Context",
                seg_options,
                index=0,
                key="floating_chat_seg_select",
                label_visibility="collapsed",
            )

            # Scrollable message area with bounded height so input form is never clipped
            with st.container(height=280, key="chat_scroll_box"):
                if not history:
                    with st.chat_message("assistant"):
                        st.markdown(
                            "👋 **Welcome to Loyalty AI Assistant!**\n\n"
                            f"Ask me anything about customer behavior, segment patterns, or retention ideas for **{DATASETS[dataset].label}**.\n\n"
                            "*Note: Real-time queries use Google Gemini when GEMINI_API_KEY is configured, or deterministic MOCK simulation if offline. Customer IDs and PII are never accessed or accepted.*"
                        )
                for msg in history:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])
                        if msg.get("mode"):
                            if msg["mode"] in ("gemini_live", "gemini", "api", "live"):
                                caption_text = "Mode: **🟢 LIVE GEMINI RESPONSE**"
                                if msg.get("timing_ms") and isinstance(msg["timing_ms"], dict) and "total_ms" in msg["timing_ms"]:
                                    t_ms = msg["timing_ms"]["total_ms"]
                                    if t_ms > 0:
                                        caption_text += f" • {t_ms:.0f}ms"
                                if msg.get("cached"):
                                    caption_text += " • (Cached)"
                                st.caption(caption_text)
                            elif msg["mode"] == "conversational":
                                st.caption("💬 *Assistant Greeting • Instant Routing*")
                            elif msg["mode"] in ("mock", "mock_fallback"):
                                cd_note = f" • Cooldown ({msg.get('cooldown_remaining', 0):.0f}s)" if msg.get("circuit_open") else ""
                                cached_note = " • (Cached)" if msg.get("cached") else ""
                                st.caption(f"⚪ *Dataset Analytics Fallback • Verified Data Grounding{cd_note}{cached_note}*")
                            elif msg["mode"] == "blocked":
                                st.caption("🛡️ *Request blocked by Privacy Guard.*")
                            elif msg["mode"] == "error":
                                st.caption("⚠️ *Unable to generate response. Please try again.*")

            # Pinned bottom input form: full-width horizontal row with text input and fixed Send button
            with st.form(key="floating_chat_form", clear_on_submit=True, border=False):
                col_inp, col_snd = st.columns([0.82, 0.18], gap="small", vertical_alignment="center")
                with col_inp:
                    user_text = st.text_input(
                        "Chat input",
                        placeholder="Ask about loyalty, retention...",
                        label_visibility="collapsed",
                        key="floating_chat_input_field",
                    )
                with col_snd:
                    send_clicked = st.form_submit_button(
                        "➤",
                        key="chat_send_btn",
                        help="Send question",
                        width="stretch",
                    )

            if send_clicked and user_text.strip():
                clean_q = user_text.strip()
                history.append({"role": "user", "content": clean_q})

                if not profiles.empty:
                    seg_summaries = []
                    for _, row in profiles.iterrows():
                        seg_item = {}
                        for col in profiles.columns:
                            val = row[col]
                            if pd.notna(val):
                                if isinstance(val, (np.floating, float)):
                                    seg_item[col] = round(float(val), 2)
                                elif isinstance(val, (np.integer, int)):
                                    seg_item[col] = int(val)
                                else:
                                    seg_item[col] = str(val)
                        seg_summaries.append(seg_item)

                    if active_seg == "All Segments (Overview)":
                        has_cc = "Customer_Count" in profiles.columns
                        total_cust = int(profiles["Customer_Count"].sum()) if has_cc else len(profiles)
                        
                        target_profile = {
                            "Segment_Name": "All Segments (Overview)",
                            "Total_Customers": total_cust,
                            "Segment_Count": len(profiles),
                            "Segments_Summary": seg_summaries,
                        }
                        for num_col in ["Recency_Days", "Frequency", "Monetary", "Tenure_Days", "Average_Order_Value", "Average_Basket_Size", "Return_Rate"]:
                            if num_col in profiles.columns:
                                if has_cc and profiles["Customer_Count"].sum() > 0:
                                    weighted_avg = float((profiles[num_col] * profiles["Customer_Count"]).sum() / total_cust)
                                    target_profile[num_col] = round(weighted_avg, 2)
                                else:
                                    target_profile[num_col] = round(float(profiles[num_col].mean()), 2)
                    else:
                        matching = profiles.loc[profiles["Segment_Name"] == active_seg]
                        target_profile = matching.iloc[0].to_dict() if not matching.empty else profiles.iloc[0].to_dict()
                        target_profile["Segments_Summary"] = seg_summaries
                else:
                    target_profile = {}

                try:
                    with st.spinner("Consulting loyalty advisor..."):
                        answer = get_segment_advice(
                            target_profile,
                            dataset=dataset,
                            allow_mock_fallback=True,
                            question=clean_q,
                        )
                    if answer.get("actions"):
                        resp_text = answer["summary"] + "\n\n" + "\n".join(f"- {act}" for act in answer["actions"])
                    else:
                        resp_text = answer["summary"]
                    history.append({
                        "role": "assistant",
                        "content": resp_text,
                        "mode": answer["mode"],
                        "status_detail": answer.get("status_detail", ""),
                        "api_error": answer.get("api_error", ""),
                        "timing_ms": answer.get("timing_ms"),
                        "circuit_open": answer.get("circuit_open", False),
                        "cooldown_remaining": answer.get("cooldown_remaining", 0.0),
                        "cached": answer.get("cached", False),
                    })
                except ValueError as exc:
                    history.append({"role": "assistant", "content": f"⚠️ Privacy Guard: {exc}", "mode": "blocked"})
                st.rerun()

    # Fixed floating circular launcher at the bottom-right corner
    with st.container(key="chat_launcher_container"):
        icon = "✕" if is_open else "💬"
        help_text = "Close Loyalty AI Assistant" if is_open else "Chat with Loyalty AI Assistant"
        if st.button(icon, key="chat_launcher_btn", help=help_text):
            st.session_state["chat_open"] = not is_open
            st.rerun()



def download_page(dataset: str, assignments: pd.DataFrame, metrics: dict) -> None:
    st.subheader("Download Results")
    st.caption("Export separate machine-readable metrics and customer segment assignment tables.")

    paths = artifact_paths(dataset)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Segmentation Metrics Report")
            st.caption("Metadata, model selection rules, stability, and validation metrics in clean JSON format.")
            metrics_path = paths["metrics"]
            report_bytes = (
                metrics_path.read_bytes()
                if metrics_path.exists()
                else json.dumps(metrics, indent=2, default=str).encode("utf-8")
            )
            st.download_button(
                "Download customer_segmentation_report.json",
                report_bytes,
                "customer_segmentation_report.json",
                "application/json",
                key="download_report_json_btn",
            )

    with col2:
        with st.container(border=True):
            st.markdown("#### Customer Segment Assignments")
            st.caption("Clean tabular customer assignment dataset with headers and segment labels in pure CSV format.")
            assign_path = paths["assignments"]
            assign_bytes = (
                assign_path.read_bytes()
                if assign_path.exists()
                else assignments.to_csv(index=False).encode("utf-8")
            )
            st.download_button(
                "Download customer_segment_assignments.csv",
                assign_bytes,
                "customer_segment_assignments.csv",
                "text/csv",
                key="download_assignments_csv_btn",
            )


def main() -> None:
    st.markdown("""
    <div style="margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border-color); width: 100%; box-sizing: border-box;">
        <div style="font-size: 11px; font-weight: 600; color: var(--accent-blue); background: var(--accent-blue-light); padding: 4px 10px; border-radius: 20px; display: inline-block; margin-bottom: 8px; letter-spacing: 0.04em; text-transform: uppercase;">
            Customer Intelligence & Segmentation
        </div>
        <h1 style="font-size: 30px; font-weight: 700; color: var(--text-primary); letter-spacing: -0.025em; margin: 0 0 6px 0; padding: 0; line-height: 1.25; word-wrap: break-word;">
            E-Commerce Customer Loyalty Analytics
        </h1>
        <p style="font-size: 14.5px; font-weight: 400; color: var(--text-secondary); margin: 0; letter-spacing: -0.01em; line-height: 1.4;">
            Understand customers. Discover patterns. Make smarter decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""
        <div style="padding: 4px 0 16px 0; margin-bottom: 8px; border-bottom: 1px solid var(--border-color);">
            <div style="font-size: 15px; font-weight: 700; color: var(--text-primary); letter-spacing: -0.02em;">
                Loyalty Studio
            </div>
            <div style="font-size: 12px; color: var(--text-secondary);">
                Portfolio Demonstration
            </div>
        </div>
        """, unsafe_allow_html=True)

        theme_options = ["Light", "Dark", "System"]
        current_theme = st.session_state.get("app_theme", "Light")
        theme_idx = theme_options.index(current_theme) if current_theme in theme_options else 0
        selected_theme = st.selectbox(
            "Appearance",
            theme_options,
            index=theme_idx,
            key="appearance_mode_select",
            help="Switch between Apple Light, Dark, or System OS appearance.",
        )
        if selected_theme != st.session_state.get("app_theme"):
            st.session_state["app_theme"] = selected_theme
            st.rerun()

    dataset = st.sidebar.selectbox("Dataset", list(DATASETS), format_func=lambda key: DATASETS[key].label)
    page = st.sidebar.radio("View", ["Overview", "Profiles", "Model Comparison", "Customer Explanation", "Batch Assignment", "Download"])
    if training_notice(dataset):
        return

    # Lightweight profile table needed for floating chat and profile/explanation pages
    profiles = load_table(dataset, "profiles")

    # Lazy-load only the artifacts required for the active page
    if page == "Overview":
        assignments = load_table(dataset, "assignments")
        metrics, quality = load_metadata(dataset)
        overview(assignments, metrics, quality, dataset=dataset, profiles=profiles)
    elif page == "Profiles":
        profiles_page(profiles, dataset)
    elif page == "Model Comparison":
        comparison = load_table(dataset, "comparison")
        pca = load_table(dataset, "pca")
        metrics, _ = load_metadata(dataset)
        comparison_page(comparison, pca, metrics, dataset=dataset)
    elif page == "Customer Explanation":
        assignments = load_table(dataset, "assignments")
        explain_page(dataset, assignments, profiles)
    elif page == "Batch Assignment":
        batch_page(dataset)
    else:
        assignments = load_table(dataset, "assignments")
        metrics, _ = load_metadata(dataset)
        download_page(dataset, assignments, metrics)

    # Floating customer-support chatbot widget across all dashboard pages
    render_floating_chat(dataset, profiles)



if __name__ == "__main__":
    main()
