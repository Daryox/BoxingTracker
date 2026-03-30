"""
dashboard.py — Dash live monitoring dashboard.

Reads artifacts/state.json and artifacts/events.json (written by the webcam
pipeline) and displays:
  - Aggregate punch counters (total / landed / missed)
  - Punch-type bar chart (landed vs missed, stacked)
  - Per-fighter ring heatmaps
  - Full event log (round markers + punch rows)

Each section updates independently every 300 ms via dcc.Interval — only
changed components are re-rendered, not the entire page.

Run with:
    python app/dash/dashboard.py
then open http://127.0.0.1:8050 in a browser.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from dash import Dash, Input, Output, dash_table, dcc, html
import plotly.graph_objects as go

ARTIFACTS   = Path("artifacts")
STATE_FILE  = ARTIFACTS / "state.json"
EVENTS_FILE = ARTIFACTS / "events.json"

REFRESH_MS = 300


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_state() -> dict:
    """Read and parse state.json; return empty dict on any error."""
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


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


# ── Layout ────────────────────────────────────────────────────────────────────

app = Dash(__name__)
app.title = "BoxingTracker Dashboard"

app.layout = html.Div(
    style={
        "backgroundColor": "#0e1117",
        "color": "white",
        "fontFamily": "sans-serif",
        "padding": "20px",
        "minHeight": "100vh",
    },
    children=[
        dcc.Interval(id="interval", interval=REFRESH_MS, n_intervals=0),

        html.H1("BoxingTracker Live Dashboard", style={"marginBottom": "24px"}),

        # ── Counters ──────────────────────────────────────────────────────────
        html.Div(id="counters", style={"display": "flex", "gap": "16px", "marginBottom": "32px"}),

        # ── Punch type chart ──────────────────────────────────────────────────
        html.H3("Punch Type Distribution", style={"marginBottom": "8px"}),
        dcc.Graph(id="punch-chart", config={"displayModeBar": False}),

        # ── Heatmaps ──────────────────────────────────────────────────────────
        html.H3("Ring Heatmaps (live)", style={"marginTop": "32px", "marginBottom": "8px"}),
        html.Div(id="heatmaps", style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # ── Event log ─────────────────────────────────────────────────────────
        html.H3("Event Log", style={"marginTop": "32px", "marginBottom": "8px"}),
        html.Div(id="event-log"),
    ],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _counter_card(label: str, value: int, colour: str) -> html.Div:
    """Styled metric card for a single counter."""
    return html.Div(
        style={
            "backgroundColor": "#1e2130",
            "borderRadius": "8px",
            "padding": "16px 24px",
            "minWidth": "140px",
            "borderTop": f"3px solid {colour}",
        },
        children=[
            html.Div(label, style={"fontSize": "13px", "color": "#aaaaaa"}),
            html.Div(str(value), style={"fontSize": "36px", "fontWeight": "bold", "color": colour}),
        ],
    )


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(Output("counters", "children"), Input("interval", "n_intervals"))
def update_counters(_):
    state = load_state()
    return [
        _counter_card("Total Punches", state.get("total_punches", 0), "#ffffff"),
        _counter_card("Landed",        state.get("landed",        0), "#2ecc71"),
        _counter_card("Missed",        state.get("missed",        0), "#e74c3c"),
    ]


@app.callback(Output("punch-chart", "figure"), Input("interval", "n_intervals"))
def update_punch_chart(_):
    punch_events = [e for e in load_events() if e.get("event_type", "punch") == "punch"]

    counts: dict = defaultdict(lambda: {"landed": 0, "missed": 0})
    for e in punch_events:
        bucket = "landed" if e.get("landed") else "missed"
        counts[e.get("punch_type", "unknown")][bucket] += 1

    punch_types = list(counts.keys())
    landed = [counts[pt]["landed"] for pt in punch_types]
    missed = [counts[pt]["missed"] for pt in punch_types]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Landed", x=punch_types, y=landed, marker_color="#2ecc71"))
    fig.add_trace(go.Bar(name="Missed", x=punch_types, y=missed, marker_color="#e74c3c"))
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font_color="white",
        margin={"t": 10, "b": 10, "l": 10, "r": 10},
        legend={"orientation": "h", "y": -0.2},
        xaxis={"gridcolor": "#2a2d3e"},
        yaxis={"gridcolor": "#2a2d3e", "title": "Punches"},
    )
    return fig


@app.callback(Output("heatmaps", "children"), Input("interval", "n_intervals"))
def update_heatmaps(_):
    state   = load_state()
    heatmaps = state.get("heatmaps")
    if not heatmaps:
        return html.Span(
            "No heatmap data yet — move around the ring to populate.",
            style={"color": "#aaaaaa"},
        )

    figures = []
    for fid, hm_list in sorted(heatmaps.items(), key=lambda kv: int(kv[0])):
        hm = np.array(hm_list, dtype=np.float32)
        if hm.size == 0:
            continue
        total_frames = int(hm.sum())

        fig = go.Figure(go.Heatmap(
            z=hm,
            colorscale="Hot",
            zmin=0,
            zmax=max(float(hm.max()), 1.0),
            showscale=True,
        ))
        fig.update_layout(
            title={
                "text": f"Fighter {fid}  ({total_frames} frames)",
                "font": {"color": "white", "size": 13},
            },
            paper_bgcolor="#1e2130",
            plot_bgcolor="#1e2130",
            font_color="white",
            margin={"t": 40, "b": 40, "l": 40, "r": 40},
            width=280,
            height=280,
            xaxis={"title": "← left · right →", "showticklabels": False, "showgrid": False},
            yaxis={
                "title": "near · far →",
                "showticklabels": False,
                "showgrid": False,
                "autorange": "reversed",
            },
        )
        figures.append(dcc.Graph(figure=fig, config={"displayModeBar": False}))

    return figures


@app.callback(Output("event-log", "children"), Input("interval", "n_intervals"))
def update_event_log(_):
    raw_events = load_events()
    if not raw_events:
        return html.Span("No events yet.", style={"color": "#aaaaaa"})

    rows = []
    for e in raw_events:
        ts    = _fmt_ts(e["ts"]) if "ts" in e else ""
        etype = e.get("event_type", "punch")

        if etype == "round_start":
            rows.append({"time": ts, "type": "▶ Round start", "fighter": "", "arm": "", "punch": "", "landed": ""})
        elif etype == "round_end":
            dur = e.get("duration_s")
            label = f"⏹ Round end ({dur:.0f}s)" if dur else "⏹ Round end"
            rows.append({"time": ts, "type": label, "fighter": "", "arm": "", "punch": "", "landed": ""})
        else:
            rows.append({
                "time":    ts,
                "type":    "punch",
                "fighter": str(e.get("fighter_id", "")),
                "arm":     e.get("arm", ""),
                "punch":   e.get("punch_type", ""),
                "landed":  "✓" if e.get("landed") else "✗",
            })

    return dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in ["time", "type", "fighter", "arm", "punch", "landed"]],
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#1e2130",
            "color": "white",
            "fontWeight": "bold",
            "border": "1px solid #2a2d3e",
        },
        style_cell={
            "backgroundColor": "#0e1117",
            "color": "white",
            "border": "1px solid #2a2d3e",
            "padding": "6px 12px",
            "textAlign": "left",
        },
        style_data_conditional=[
            {
                "if": {"filter_query": '{landed} = "✓"', "column_id": "landed"},
                "color": "#2ecc71",
                "fontWeight": "bold",
            },
            {
                "if": {"filter_query": '{landed} = "✗"', "column_id": "landed"},
                "color": "#e74c3c",
                "fontWeight": "bold",
            },
            {
                "if": {"filter_query": '{type} contains "Round"'},
                "backgroundColor": "#1e2130",
                "fontWeight": "bold",
                "color": "#f0c040",
            },
        ],
        page_size=50,
        sort_action="native",
    )


if __name__ == "__main__":
    app.run(debug=False)
