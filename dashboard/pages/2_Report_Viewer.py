"""
FinSight AI — Report Viewer Page
Full interactive report display with tabs for Financial, Risk, and Sentiment.
"""
from __future__ import annotations

import json

import httpx
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="FinSight AI — Report Viewer",
    page_icon="📋",
    layout="wide",
)

API_BASE = "http://localhost:8000/api/v1"

st.markdown("# 📋 Report Viewer")

# ---- Sidebar controls ----
with st.sidebar:
    st.markdown("### 🔎 Load Report")
    run_id_input = st.text_input("Run ID", value=st.session_state.get("view_run_id", ""))
    ticker_search = st.text_input("Or search by ticker", placeholder="AAPL")
    load_btn = st.button("📂 Load Report", type="primary", use_container_width=True)

# ---- Load report ----
report = None
run_id = run_id_input.strip()

if load_btn and run_id:
    try:
        resp = httpx.get(f"{API_BASE}/reports/{run_id}", timeout=10.0)
        if resp.status_code == 200:
            report = resp.json()
            st.session_state["loaded_report"] = report
            st.session_state["loaded_run_id"] = run_id
        elif resp.status_code == 202:
            st.warning("⏳ Analysis still running. Check back soon.")
        else:
            st.error(f"Report not found: {resp.text[:200]}")
    except httpx.ConnectError:
        st.error("❌ API offline.")

elif load_btn and ticker_search:
    try:
        resp = httpx.get(f"{API_BASE}/reports/ticker/{ticker_search.upper()}", timeout=10.0)
        if resp.status_code == 200:
            reports_list = resp.json()
            if reports_list:
                run_id = reports_list[0]["run_id"]
                resp2 = httpx.get(f"{API_BASE}/reports/{run_id}", timeout=10.0)
                if resp2.status_code == 200:
                    report = resp2.json()
                    st.session_state["loaded_report"] = report
                    st.session_state["loaded_run_id"] = run_id
    except Exception as exc:
        st.error(f"Error: {exc}")

# Use session state report if available
if report is None and "loaded_report" in st.session_state:
    report = st.session_state["loaded_report"]
    run_id = st.session_state.get("loaded_run_id", "")

# ---- Display report ----
if report:
    meta = report.get("metadata", {})
    overall = report.get("overall_signal", {})
    financial = report.get("financial_analysis", {})
    risk = report.get("risk_analysis", {})
    sentiment = report.get("sentiment_analysis", {})
    ticker = meta.get("ticker", "")

    # ---- Signal Banner ----
    signal = overall.get("signal", "NEUTRAL")
    confidence = overall.get("confidence", "LOW")
    signal_color = "#00b894" if signal == "BULLISH" else "#d63031" if signal == "BEARISH" else "#636e72"

    st.markdown(f"""
    <div style="
      background: linear-gradient(135deg, #1a1a2e, #0f3460);
      color: white; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
    ">
      <div style="display:flex; align-items:center; gap:1rem;">
        <div>
          <div style="font-size:2rem; font-weight:700;">{ticker}</div>
          <div style="opacity:0.8;">{meta.get('company_name', '')}</div>
        </div>
        <div style="
          background:{signal_color}; padding:0.5rem 1.5rem;
          border-radius:50px; font-weight:700; font-size:1.2rem;
        ">{signal}</div>
        <div style="opacity:0.7;">Confidence: {confidence}</div>
        <div style="margin-left:auto; opacity:0.6; font-size:0.85rem;">
          {meta.get('generated_at', '')[:10]}
        </div>
      </div>
      <div style="margin-top:0.8rem; opacity:0.85; font-style:italic;">
        {overall.get('rationale', '')}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Executive Summary ----
    st.markdown("### 📝 Executive Summary")
    st.info(report.get("executive_summary", ""))

    col_thesis, col_risks = st.columns(2)
    with col_thesis:
        st.markdown("**💡 Investment Thesis**")
        for point in report.get("investment_thesis", []):
            st.markdown(f"✅ {point}")
    with col_risks:
        st.markdown("**⚠️ Key Risks**")
        for risk_point in report.get("key_risks", []):
            st.markdown(f"🔴 {risk_point}")

    # ---- Top metrics row ----
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Financial Health", f"{financial.get('health_score', 'N/A'):.1f}/10")
    with m2:
        st.metric("Risk Score", f"{risk.get('overall_risk_score', 'N/A'):.1f}/10")
    with m3:
        score = sentiment.get("aggregate_score", 0)
        st.metric("Sentiment", f"{score:.2f}", delta=f"{sentiment.get('trend', '')}")
    with m4:
        st.metric("Articles Analyzed", sentiment.get("article_count", 0))

    # ---- Tabs ----
    tab_fin, tab_risk, tab_sent = st.tabs(["💰 Financial", "⚠️ Risk", "📰 Sentiment"])

    # ---- FINANCIAL TAB ----
    with tab_fin:
        st.markdown("#### Financial Ratios")
        ratios = financial.get("ratios", {})
        ratio_data = {
            "P/E Ratio": ratios.get("pe_ratio"),
            "P/B Ratio": ratios.get("pb_ratio"),
            "EV/EBITDA": ratios.get("ev_ebitda"),
            "Debt/Equity": ratios.get("debt_to_equity"),
            "Current Ratio": ratios.get("current_ratio"),
            "ROE (%)": ratios.get("roe"),
            "ROA (%)": ratios.get("roa"),
            "Gross Margin (%)": ratios.get("gross_margin"),
            "Operating Margin (%)": ratios.get("operating_margin"),
            "FCF TTM": ratios.get("free_cash_flow_ttm"),
            "Revenue CAGR 3Y (%)": ratios.get("revenue_cagr_3y"),
        }

        col_r1, col_r2 = st.columns(2)
        items = [(k, v) for k, v in ratio_data.items() if v is not None]
        for i, (k, v) in enumerate(items):
            with (col_r1 if i % 2 == 0 else col_r2):
                st.metric(k, f"{v:.2f}")

        # Price returns chart
        st.markdown("#### Price Performance")
        perf = financial.get("price_performance", {})
        labels = ["1M", "3M", "6M", "1Y", "YTD"]
        keys = ["return_1m", "return_3m", "return_6m", "return_1y", "return_ytd"]
        values = [perf.get(k) for k in keys]
        valid_labels = [l for l, v in zip(labels, values) if v is not None]
        valid_values = [v for v in values if v is not None]

        if valid_values:
            colors = ["#00b894" if v >= 0 else "#d63031" for v in valid_values]
            fig = go.Figure(
                go.Bar(
                    x=valid_labels, y=valid_values,
                    marker_color=colors,
                    text=[f"{v:.1f}%" for v in valid_values],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="Price Returns (%)", yaxis_title="Return (%)",
                height=300, plot_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Health score gauge
        health = financial.get("health_score", 5)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health,
            title={"text": "Financial Health Score"},
            gauge={
                "axis": {"range": [0, 10]},
                "bar": {"color": "#0f3460"},
                "steps": [
                    {"range": [0, 4], "color": "#d63031"},
                    {"range": [4, 7], "color": "#e17055"},
                    {"range": [7, 10], "color": "#00b894"},
                ],
            },
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("**Strengths:** " + " | ".join(financial.get("strengths", [])))
        st.markdown("**Weaknesses:** " + " | ".join(financial.get("weaknesses", [])))

    # ---- RISK TAB ----
    with tab_risk:
        st.markdown(f"#### Overall Risk Score: **{risk.get('overall_risk_score', 'N/A'):.1f}/10**")

        factors = risk.get("risk_factors", [])
        if factors:
            # Radar chart
            categories = list({rf["category"] for rf in factors})
            cat_scores = {}
            for rf in factors:
                cat = rf["category"]
                score = (rf.get("severity", 3) + rf.get("likelihood", 3)) / 2
                cat_scores[cat] = max(cat_scores.get(cat, 0), score)

            if len(cat_scores) >= 3:
                cats = list(cat_scores.keys())
                vals = [cat_scores[c] for c in cats]
                fig_radar = go.Figure(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=cats + [cats[0]],
                    fill="toself",
                    fillcolor="rgba(214,48,49,0.2)",
                    line_color="#d63031",
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(range=[0, 5])),
                    title="Risk Category Radar",
                    height=350,
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # Risk factors table
            st.markdown("#### Risk Factors")
            for rf in sorted(factors, key=lambda x: x.get("severity", 0), reverse=True):
                sev = rf.get("severity", 3)
                color = "red" if sev >= 4 else "orange" if sev == 3 else "green"
                st.markdown(
                    f"**:{color}[{rf['category']}]** (Severity: {sev}/5, Likelihood: {rf.get('likelihood',3)}/5)"
                    f" — {rf['description']}"
                )
        else:
            st.info("No risk factors extracted.")

    # ---- SENTIMENT TAB ----
    with tab_sent:
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            # Donut chart
            pos = sentiment.get("positive_count", 0)
            neg = sentiment.get("negative_count", 0)
            neu = sentiment.get("neutral_count", 0)
            if pos + neg + neu > 0:
                fig_donut = go.Figure(go.Pie(
                    labels=["Positive", "Neutral", "Negative"],
                    values=[pos, neu, neg],
                    hole=0.55,
                    marker_colors=["#00b894", "#636e72", "#d63031"],
                ))
                fig_donut.update_layout(title="Sentiment Distribution", height=300)
                st.plotly_chart(fig_donut, use_container_width=True)

        with col_s2:
            st.metric("Aggregate Score", f"{sentiment.get('aggregate_score', 0):.2f}")
            st.metric("Trend", sentiment.get("trend", "STABLE"))
            st.metric("Earnings Tone", sentiment.get("earnings_tone", "N/A"))

        # Top events
        st.markdown("#### Top Events")
        for event in sentiment.get("top_events", []):
            sent_color = "green" if event.get("sentiment") == "POSITIVE" else "red" if event.get("sentiment") == "NEGATIVE" else "gray"
            st.markdown(
                f"**:{sent_color}[{event.get('sentiment', '')}]** `{event.get('date', '')}` — "
                f"{event.get('headline', '')}  \n*{event.get('significance', '')}*"
            )

    # ---- Download PDF ----
    st.divider()
    run_id_for_pdf = st.session_state.get("loaded_run_id", "")
    if run_id_for_pdf:
        pdf_url = f"http://localhost:8000/api/v1/reports/{run_id_for_pdf}/pdf"
        st.markdown(f"### 📥 [Download PDF Report]({pdf_url})")

    st.markdown(f"*Conclusion: {report.get('conclusion', '')}*")

else:
    st.info("👈 Enter a Run ID or ticker in the sidebar to load a report.")
