"""
FinSight AI — Alerts Management Page
Display and manage alert configs and events.
"""
from __future__ import annotations

import time

import httpx
import streamlit as st

st.set_page_config(
    page_title="FinSight AI — Alerts",
    page_icon="🔔",
    layout="wide",
)

API_BASE = "http://localhost:8000/api/v1"

st.markdown("# 🔔 Alerts Management")
st.caption("Configure alerts for new SEC filings, sentiment spikes, and signal changes.")

# ---- Create new alert ----
with st.expander("➕ Create New Alert", expanded=False):
    with st.form("new_alert_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            alert_ticker = st.text_input("Ticker", placeholder="AAPL")
        with col2:
            alert_type = st.selectbox(
                "Alert Type",
                ["NEW_FILING", "SENTIMENT_SPIKE", "SIGNAL_CHANGE"],
            )
        with col3:
            threshold = st.number_input("Threshold", value=0.3, min_value=0.0, max_value=1.0, step=0.05)
        with col4:
            delivery = st.selectbox("Delivery", ["dashboard", "log"])

        submitted = st.form_submit_button("Create Alert", type="primary")
        if submitted and alert_ticker:
            try:
                resp = httpx.post(
                    f"{API_BASE}/alerts/configs",
                    json={
                        "ticker": alert_ticker.upper(),
                        "alert_type": alert_type,
                        "threshold": threshold,
                        "delivery_method": delivery,
                        "enabled": True,
                    },
                    timeout=5.0,
                )
                if resp.status_code == 201:
                    st.success(f"✅ Alert created for {alert_ticker.upper()} ({alert_type})")
                    st.rerun()
                else:
                    st.error(f"Failed: {resp.text[:200]}")
            except Exception as exc:
                st.error(f"Error: {exc}")

# ---- Alert Configs Table ----
st.divider()
st.markdown("### ⚙️ Active Alert Configurations")

try:
    resp = httpx.get(f"{API_BASE}/alerts/configs", timeout=5.0)
    if resp.status_code == 200:
        configs = resp.json()
        if configs:
            for config in configs:
                col1, col2, col3, col4, col5 = st.columns([2, 2, 1.5, 1.5, 1])
                with col1:
                    st.markdown(f"**{config['ticker']}**")
                with col2:
                    st.markdown(config["alert_type"])
                with col3:
                    status_txt = "🟢 Enabled" if config["enabled"] else "⚫ Disabled"
                    st.markdown(status_txt)
                with col4:
                    st.markdown(f"Threshold: {config.get('threshold') or 'N/A'}")
                with col5:
                    col_toggle, col_del = st.columns(2)
                    with col_toggle:
                        if st.button("⏸" if config["enabled"] else "▶", key=f"toggle_{config['id']}"):
                            httpx.patch(f"{API_BASE}/alerts/configs/{config['id']}/toggle", timeout=5.0)
                            st.rerun()
                    with col_del:
                        if st.button("🗑", key=f"del_{config['id']}"):
                            httpx.delete(f"{API_BASE}/alerts/configs/{config['id']}", timeout=5.0)
                            st.rerun()
        else:
            st.info("No alert configurations yet. Create one above.")
    else:
        st.warning("Could not load alert configs.")
except httpx.ConnectError:
    st.error("❌ API offline.")

# ---- Alert Events ----
st.divider()
st.markdown("### 📬 Alert Events")

filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    filter_ticker = st.text_input("Filter by ticker", placeholder="Leave empty for all")
with filter_col2:
    show_unack_only = st.checkbox("Show unacknowledged only", value=False)

try:
    params: dict = {}
    if filter_ticker:
        params["ticker"] = filter_ticker.upper()
    if show_unack_only:
        params["unacknowledged_only"] = "true"

    resp = httpx.get(f"{API_BASE}/alerts/events", params=params, timeout=5.0)
    if resp.status_code == 200:
        events = resp.json()
        if events:
            for event in events:
                col1, col2, col3, col4 = st.columns([1.5, 1.5, 5, 1])
                with col1:
                    st.markdown(f"**{event['ticker']}**")
                with col2:
                    st.markdown(event["triggered_at"][:10])
                with col3:
                    icon = "🔴" if not event["acknowledged"] else "✅"
                    st.markdown(f"{icon} {event['message']}")
                with col4:
                    if not event["acknowledged"]:
                        if st.button("Ack", key=f"ack_{event['id']}"):
                            httpx.patch(
                                f"{API_BASE}/alerts/events/{event['id']}/acknowledge",
                                json={"acknowledged": True},
                                timeout=5.0,
                            )
                            st.rerun()
        else:
            st.info("No alert events found.")
    else:
        st.warning("Could not load alert events.")
except httpx.ConnectError:
    st.error("❌ API offline.")

# ---- Auto-refresh ----
st.divider()
if st.checkbox("🔄 Auto-refresh every 60 seconds"):
    time.sleep(60)
    st.rerun()
