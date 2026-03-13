import json
import time
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd

ARTIFACTS = Path("artifacts")
STATE_FILE = ARTIFACTS / "state.json"

st.set_page_config(page_title="BoxingTracker Dashboard", layout="wide")

st.title("🥊 BoxingTracker Live Dashboard")

# Auto refresh every 500ms
st.experimental_rerun  # (just to show it's allowed, we’ll use polling instead)

def load_state():
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except:
        return None

state = load_state()

if state is None:
    st.warning("Waiting for runtime data...")
    st.stop()

# ----------------------------
# Punch stats
# ----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Punches", state.get("total_punches", 0))
col2.metric("Landed", state.get("landed", 0))
col3.metric("Missed", state.get("missed", 0))

# ----------------------------
# Punch type distribution
# ----------------------------
st.subheader("Punch Type Distribution")

types = state.get("punch_type_counts", {})
if types:
    df = pd.DataFrame.from_dict(types, orient="index", columns=["count"])
    st.bar_chart(df)

# ----------------------------
# Ring Heatmap
# ----------------------------
st.subheader("Ring Position Heatmap")

heatmap = state.get("heatmap")
if heatmap:
    hm = np.array(heatmap)
    hm_vis = np.kron(hm, np.ones((40,40)))
    st.image(hm_vis, clamp=True)