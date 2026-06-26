"""
FinSight AI — Streamlit Home Page
Main entry point with company search and recent analyses display.
"""
from __future__ import annotations

import time

import httpx
import streamlit as st

# ---- Page config ----
st.set_page_config(
    page_title="FinSight AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000/api/v1"

# ---- Custom CSS ----
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
  }

  .main-header h1 { font-size: 2.5rem; font-weight: 700; margin: 0; }
  .main-header p { font-size: 1rem; opacity: 0.8; margin: 0.5rem 0 0; }

  .signal-bullish { color: #00b894; font-weight: 700; }
  .signal-bearish { color: #d63031; font-weight: 700; }
  .signal-neutral { color: #636e72; font-weight: 700; }

  .run-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.5rem;
    border-left: 4px solid #0f3460;
  }

  .metric-card {
    background: white;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
<div class="main-header">
  <h1>📊 FinSight AI</h1>
  <p>Automated Multi-Agent Financial Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)

# ---- Sidebar navigation ----
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    st.page_link("Home.py", label="🏠 Home")
    st.page_link("pages/1_Analyze.py", label="🔍 Analyze")
    st.page_link("pages/2_Report_Viewer.py", label="📋 Report Viewer")
    st.page_link("pages/3_Alerts.py", label="🔔 Alerts")
    st.divider()
    st.markdown("### About")
    st.caption(
        "FinSight AI uses LangGraph agents to fetch SEC EDGAR filings, "
        "market data, and news — then synthesizes them into actionable "
        "investment intelligence."
    )

# ---- Main search ----
st.markdown("### 🔍 Analyze a Company")

col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    ticker_input = st.text_input(
        "Enter company ticker symbol",
        placeholder="e.g., AAPL, MSFT, GOOGL",
        label_visibility="collapsed",
    )
with col2:
    force_refresh = st.checkbox("Force refresh", value=False)
with col3:
    analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)

# ---- Launch analysis ----
if analyze_btn and ticker_input:
    ticker = ticker_input.upper().strip()
    with st.spinner(f"Launching analysis for **{ticker}**..."):
        try:
            resp = httpx.post(
                f"{API_BASE}/analyze",
                json={"ticker": ticker, "force_refresh": force_refresh},
                timeout=15.0,
            )
            if resp.status_code in (200, 202):
                data = resp.json()
                run_id = data.get("run_id", "")
                status = data.get("status", "PENDING")
                st.session_state["current_run_id"] = run_id
                st.session_state["current_ticker"] = ticker

                if status == "COMPLETE":
                    st.success(f"✅ Report already exists for {ticker}! View in **Report Viewer**.")
                else:
                    st.success(f"✅ Analysis launched! Run ID: `{run_id}`")
                    st.info("📡 Track progress in the **Analyze** page.")
            else:
                st.error(f"API error: {resp.status_code} — {resp.text[:200]}")
        except httpx.ConnectError:
            st.error("❌ Could not connect to API. Make sure `uvicorn app.main:app --port 8000` is running.")
        except Exception as exc:
            st.error(f"Error: {exc}")

# ---- Active run status ----
if "current_run_id" in st.session_state:
    run_id = st.session_state["current_run_id"]
    ticker = st.session_state.get("current_ticker", "")
    st.divider()
    st.markdown(f"### 📡 Active Run — `{ticker}` (`{run_id[:8]}...`)")
    try:
        resp = httpx.get(f"{API_BASE}/runs/{run_id}", timeout=5.0)
        if resp.status_code == 200:
            run_data = resp.json()
            status = run_data.get("status", "UNKNOWN")
            if status == "COMPLETE":
                st.success("✅ Analysis complete! Go to **Report Viewer** to see results.")
                if st.button("View Report →"):
                    st.session_state["view_run_id"] = run_id
                    st.switch_page("pages/2_Report_Viewer.py")
            elif status == "FAILED":
                st.error(f"❌ Analysis failed: {run_data.get('error_message', 'Unknown error')}")
            else:
                st.info(f"⏳ Status: **{status}** — Refresh page to update")
    except Exception:
        pass

# ---- Recent runs ----
st.divider()
st.markdown("### 📈 Recent Analyses")

try:
    resp = httpx.get(f"{API_BASE}/health", timeout=5.0)
    api_online = resp.status_code == 200
except Exception:
    api_online = False

if not api_online:
    st.warning("⚠️ API is offline. Start with: `uvicorn app.main:app --reload --port 8000`")
else:
    # Fetch completed runs from DB via API
    try:
        # We show some recent tickers if stored in session state
        if "recent_tickers" not in st.session_state:
            st.session_state["recent_tickers"] = []

        if ticker_input and analyze_btn:
            t = ticker_input.upper()
            if t not in st.session_state["recent_tickers"]:
                st.session_state["recent_tickers"].insert(0, t)
                st.session_state["recent_tickers"] = st.session_state["recent_tickers"][:10]

        if st.session_state["recent_tickers"]:
            for t in st.session_state["recent_tickers"]:
                try:
                    r_resp = httpx.get(f"{API_BASE}/reports/ticker/{t}", timeout=5.0)
                    if r_resp.status_code == 200:
                        reports = r_resp.json()
                        if reports:
                            latest = reports[0]
                            signal = latest.get("overall_signal", "NEUTRAL")
                            signal_color = "green" if signal == "BULLISH" else "red" if signal == "BEARISH" else "gray"
                            st.markdown(f"""
                            <div class="run-card">
                              <strong>{t}</strong> &nbsp;
                              <span style="color:{signal_color}; font-weight:600;">{signal}</span> &nbsp;
                              <span style="color:#636e72; font-size:0.85rem;">
                                Financial: {latest.get('financial_score', 'N/A'):.1f}/10 |
                                Risk: {latest.get('risk_score', 'N/A'):.1f}/10 |
                                {latest.get('created_at', '')[:10]}
                              </span>
                            </div>
                            """, unsafe_allow_html=True)
                except Exception:
                    pass
        else:
            st.caption("No recent analyses. Enter a ticker above to get started.")

    except Exception as exc:
        st.caption(f"Error loading recent analyses: {exc}")
