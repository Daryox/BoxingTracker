import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

ARTIFACTS = Path("artifacts")
STATE_FILE = ARTIFACTS / "state.json"

st.set_page_config(page_title="BoxingTracker Dashboard", layout="wide")
st.title("🥊 BoxingTracker Live Dashboard")

# Refresh every 500 ms
if st_autorefresh is not None:
    st_autorefresh(interval=300, key="refresh")
else:
    # fallback: rerun button
    st.caption("Install streamlit-autorefresh for live updates: pip install streamlit-autorefresh")
    if st.button("Refresh now"):
        st.rerun()

def load_state():
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

state = load_state()

if state is None:
    st.warning("Waiting for runtime data... Start the webcam pipeline to generate artifacts/state.json")
    st.stop()

# ---- Metrics
c1, c2, c3 = st.columns(3)
c1.metric("Total Punches", state.get("total_punches", 0))
c2.metric("Landed", state.get("landed", 0))
c3.metric("Missed", state.get("missed", 0))

# ---- Punch type distribution
st.subheader("Punch Type Distribution")
types = state.get("punch_type_counts", {})
if types:
    df_types = pd.DataFrame({"count": types}).T
    # better: turn dict into series
    df = pd.DataFrame.from_dict(types, orient="index", columns=["count"])
    st.bar_chart(df)

# ---- Heatmaps (per fighter)
st.subheader("Ring Heatmaps")
heatmaps = state.get("heatmaps", None)
if heatmaps:
    cols = st.columns(min(3, len(heatmaps)))
    for i, (fid, hm_list) in enumerate(sorted(heatmaps.items(), key=lambda kv: int(kv[0]))):
        hm = np.array(hm_list, dtype=np.float32)
        if hm.size == 0:
            continue
        # normalize for display
        mx = float(hm.max())
        if mx > 0:
            hm = hm / mx
        # upscale if small (e.g., 5x5)
        if hm.shape[0] <= 20:
            hm_vis = np.kron(hm, np.ones((40, 40), dtype=np.float32))
        else:
            hm_vis = hm
        st.image(hm_vis, clamp=True)

# ---- Recent events table
st.subheader("Recent Events")
events = state.get("last_events", [])
if events:
    st.dataframe(pd.DataFrame(events))