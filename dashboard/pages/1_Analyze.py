"""
FinSight AI — Analyze Page
Live progress tracking with agent-by-agent status indicators.
"""
from __future__ import annotations

import time

import httpx
import streamlit as st

st.set_page_config(page_title="FinSight AI — Analyze", page_icon="🔍", layout="wide")

API_BASE = "http://localhost:8000/api/v1"

st.markdown("# 🔍 Real-Time Analysis Tracker")
st.caption("Monitor your analysis in real time as each agent completes its work.")

# ---- Input ----
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    ticker = st.text_input("Ticker Symbol", placeholder="AAPL")
with col2:
    force = st.checkbox("Force refresh")
with col3:
    start = st.button("▶ Start Analysis", type="primary", use_container_width=True)

if start and ticker:
    try:
        resp = httpx.post(
            f"{API_BASE}/analyze",
            json={"ticker": ticker.upper(), "force_refresh": force},
            timeout=15.0,
        )
        if resp.status_code in (200, 202):
            data = resp.json()
            st.session_state["tracking_run_id"] = data["run_id"]
            st.session_state["tracking_ticker"] = ticker.upper()
            st.rerun()
        else:
            st.error(f"Failed to start analysis: {resp.text[:200]}")
    except httpx.ConnectError:
        st.error("❌ API offline. Start the FastAPI server first.")

# ---- Live tracking ----
if "tracking_run_id" in st.session_state:
    run_id = st.session_state["tracking_run_id"]
    ticker = st.session_state.get("tracking_ticker", "")

    st.divider()
    st.markdown(f"### 📡 Tracking: **{ticker}** — Run `{run_id[:8]}...`")

    # Agent stages
    stages = [
        ("🔧", "Orchestrator", "Resolving company info & CIK"),
        ("📥", "Ingestor", "Fetching SEC filings, market data, news"),
        ("💰", "Financial Agent", "Computing ratios & interpreting MD&A"),
        ("⚠️", "Risk Agent", "Extracting risk factors from 10-K/10-Q"),
        ("📰", "Sentiment Agent", "Classifying news sentiment"),
        ("🧠", "Synthesis Agent", "Generating final investment report"),
    ]

    try:
        resp = httpx.get(f"{API_BASE}/runs/{run_id}", timeout=5.0)
        if resp.status_code == 200:
            status = resp.json().get("status", "PENDING")
        else:
            status = "PENDING"
    except Exception:
        status = "PENDING"

    # Display status stages
    stage_cols = st.columns(len(stages))
    for i, (icon, name, desc) in enumerate(stages):
        with stage_cols[i]:
            if status == "COMPLETE":
                st.markdown(f"✅  \n**{name}**")
            elif status == "FAILED":
                st.markdown(f"❌  \n**{name}**")
            elif status == "RUNNING":
                # Show spinner for middle stages, check for first/last
                if i == 0:
                    st.markdown(f"✅  \n**{name}**")
                elif i == len(stages) - 1:
                    st.markdown(f"⏳  \n**{name}**")
                else:
                    st.markdown(f"🔄  \n**{name}**")
            else:
                st.markdown(f"⏸️  \n**{name}**")
            st.caption(desc)

    st.divider()

    # Status display
    if status == "COMPLETE":
        st.success("✅ Analysis Complete!")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📋 View Full Report", type="primary"):
                st.session_state["view_run_id"] = run_id
                st.switch_page("pages/2_Report_Viewer.py")
        with col_b:
            pdf_url = f"http://localhost:8000/api/v1/reports/{run_id}/pdf"
            st.markdown(f"[📥 Download PDF]({pdf_url})", unsafe_allow_html=True)

    elif status == "FAILED":
        st.error("❌ Analysis failed. Check API logs for details.")
        if st.button("🔄 Retry"):
            del st.session_state["tracking_run_id"]
            st.rerun()

    else:
        # Show progress
        with st.spinner(f"Analysis in progress (status: {status})..."):
            elapsed = st.empty()
            elapsed.info(f"⏳ Status: **{status}** — Auto-refreshing every 5 seconds...")

        time.sleep(5)
        st.rerun()

    # Clear button
    if st.button("✖ Clear tracking"):
        del st.session_state["tracking_run_id"]
        st.rerun()

else:
    st.info("Enter a ticker above and click **Start Analysis** to begin real-time tracking.")
