"""
dashboard.py — Streamlit live monitoring dashboard.

Reads artifacts/state.json (written by the webcam pipeline) and displays:
  - Aggregate punch counters (total / landed / missed)
  - Punch-type bar chart
  - Per-fighter ring heatmaps
  - Full event log from events.json (round markers + punch rows)

The page auto-refreshes every 300 ms via streamlit-autorefresh if installed,
otherwise a manual "Refresh" button is shown as a fallback.
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# Path to the JSON files written by the webcam pipeline.
ARTIFACTS   = Path("artifacts")
STATE_FILE  = ARTIFACTS / "state.json"
EVENTS_FILE = ARTIFACTS / "events.json"

st.set_page_config(page_title="BoxingTracker Dashboard", layout="wide")
st.title("BoxingTracker Live Dashboard")

# Auto-refresh every 300 ms so the dashboard tracks live changes.
if st_autorefresh is not None:
    st_autorefresh(interval=300, key="refresh")
else:
    st.caption("Install streamlit-autorefresh for live updates: pip install streamlit-autorefresh")
    if st.button("Refresh now"):
        st.rerun()


def load_state() -> dict | None:
    """Read and parse state.json; return None on any error."""
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_events() -> list:
    """Read and parse events.json; return empty list on any error."""
    if not EVENTS_FILE.exists():
        return []
    try:
        with open(EVENTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("events", [])
    except Exception:
        return []


def _fmt_ts(ts: float) -> str:
    """Format a Unix timestamp as HH:MM:SS."""
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


state = load_state()

if state is None:
    st.warning("Waiting for runtime data… Start the webcam pipeline to generate artifacts/state.json")
    st.stop()

# ── Aggregate counters ────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Total Punches", state.get("total_punches", 0))
c2.metric("Landed", state.get("landed", 0))
c3.metric("Missed", state.get("missed", 0))

# ── Punch-type distribution bar chart (landed vs missed) ─────────────────────
st.subheader("Punch Type Distribution")
punch_events = [e for e in load_events() if e.get("event_type", "punch") == "punch"]
if punch_events:
    from collections import defaultdict
    counts: dict = defaultdict(lambda: {"landed": 0, "missed": 0})
    for e in punch_events:
        bucket = "landed" if e.get("landed") else "missed"
        counts[e.get("punch_type", "unknown")][bucket] += 1
    df = pd.DataFrame(counts).T.fillna(0).astype(int)[["landed", "missed"]]
    st.bar_chart(df, color=["#2ecc71", "#e74c3c"])

# ── Per-fighter ring heatmaps (live, coloured) ───────────────────────────────
st.subheader("Ring Heatmaps (live)")
heatmaps = state.get("heatmaps")
if heatmaps:
    sorted_fighters = sorted(heatmaps.items(), key=lambda kv: int(kv[0]))
    cols = st.columns(min(3, len(sorted_fighters)))

    for i, (fid, hm_list) in enumerate(sorted_fighters):
        hm = np.array(hm_list, dtype=np.float32)
        if hm.size == 0:
            continue

        # Up-scale small grids (e.g. 5×5) so individual cells are visible.
        if hm.shape[0] <= 20:
            scale = 40
            hm = np.kron(hm, np.ones((scale, scale), dtype=np.float32))

        total_hits = int(hm.sum())

        fig, ax = plt.subplots(figsize=(3, 3))
        fig.patch.set_facecolor("#0e1117")   # match Streamlit dark background
        ax.set_facecolor("#0e1117")

        # "hot" colormap: black → red → yellow → white as intensity rises.
        im = ax.imshow(
            hm,
            cmap="hot",
            interpolation="nearest",
            aspect="equal",
            vmin=0,                          # always anchor at 0 so colours are comparable
            vmax=max(hm.max(), 1.0),         # avoid division-by-zero on empty maps
        )

        # Axis labels orient the viewer relative to the ring.
        ax.set_title(f"Fighter {fid}  ({total_hits} hits)", color="white", fontsize=9, pad=4)
        ax.set_xlabel("← left · right →", color="#aaaaaa", fontsize=7)
        ax.set_ylabel("near · far →", color="#aaaaaa", fontsize=7)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="white", labelsize=6)

        plt.tight_layout(pad=0.5)
        cols[i % len(cols)].pyplot(fig, use_container_width=True)
        plt.close(fig)   # free memory; important inside a fast refresh loop
else:
    st.caption("No heatmap data yet — move around the ring to populate.")

# ── Full event log ────────────────────────────────────────────────────────────
st.subheader("Event Log")
raw_events = load_events()

if not raw_events:
    st.caption("No events yet.")
else:
    rows = []
    for e in raw_events:
        ts     = _fmt_ts(e["ts"]) if "ts" in e else ""
        etype  = e.get("event_type", "punch")

        if etype == "round_start":
            rows.append({
                "time":    ts,
                "type":    "▶ Round start",
                "fighter": "",
                "arm":     "",
                "punch":   "",
                "landed":  "",
            })
        elif etype == "round_end":
            dur = e.get("duration_s")
            rows.append({
                "time":    ts,
                "type":    f"⏹ Round end ({dur:.0f}s)" if dur else "⏹ Round end",
                "fighter": "",
                "arm":     "",
                "punch":   "",
                "landed":  "",
            })
        else:
            rows.append({
                "time":    ts,
                "type":    "punch",
                "fighter": e.get("fighter_id", ""),
                "arm":     e.get("arm", ""),
                "punch":   e.get("punch_type", ""),
                "landed":  "✓" if e.get("landed") else "✗",
            })

    df = pd.DataFrame(rows, columns=["time", "type", "fighter", "arm", "punch", "landed"])
    st.dataframe(df, use_container_width=True, hide_index=True)
