"""
dashboard.py — Dash live monitoring dashboard.

Reads artifacts/state.json and artifacts/events.json (written by the webcam
pipeline) and displays live stats with per-round tabs.

Run with:
    python app/dash/dashboard.py
then open http://127.0.0.1:8050 in a browser.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from dash import Dash, Input, Output, State, dash_table, dcc, html, ctx
import plotly.graph_objects as go

ARTIFACTS    = Path("artifacts")
STATE_FILE   = ARTIFACTS / "state.json"
EVENTS_FILE  = ARTIFACTS / "events.json"
COMMAND_FILE = ARTIFACTS / "command.json"

REFRESH_MS = 300


def _clear_artifacts() -> None:
    STATE_FILE.unlink(missing_ok=True)
    EVENTS_FILE.unlink(missing_ok=True)
    COMMAND_FILE.unlink(missing_ok=True)
    (ARTIFACTS / "ring_calibration.json").unlink(missing_ok=True)


_clear_artifacts()


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_events() -> list:
    if not EVENTS_FILE.exists():
        return []
    try:
        with open(EVENTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("events", [])
    except Exception:
        return []


def _infer_round_state(events: list) -> dict:
    """Derive current round number and active flag from the event stream."""
    round_num = 0
    active = False
    for e in events:
        etype = e.get("event_type")
        if etype == "round_start":
            round_num = e.get("round", round_num + 1)
            active = True
        elif etype == "round_end":
            active = False
    return {"round": round_num, "active": active}


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def _write_command(cmd: str) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(COMMAND_FILE, "w", encoding="utf-8") as f:
        json.dump({"cmd": cmd}, f)


# ── Layout ────────────────────────────────────────────────────────────────────

_BTN_BASE = {
    "border": "none", "borderRadius": "6px",
    "padding": "10px 24px", "fontSize": "15px",
    "fontWeight": "bold",
}
_BTN_START  = {**_BTN_BASE, "backgroundColor": "#2ecc71", "color": "black",  "cursor": "pointer"}
_BTN_END    = {**_BTN_BASE, "backgroundColor": "#e74c3c", "color": "white",  "cursor": "pointer"}
_BTN_NEXT   = {**_BTN_BASE, "backgroundColor": "#3498db", "color": "white",  "cursor": "pointer"}
_BTN_OFF    = {"opacity": "0.4", "cursor": "not-allowed"}

_TAB_STYLE          = {"color": "white",   "backgroundColor": "#1e2130"}
_TAB_SELECTED_STYLE = {"color": "#f0c040", "backgroundColor": "#0e1117", "borderTop": "2px solid #f0c040"}

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
        # Stores the last-known round state so button logic doesn't need to re-read events
        dcc.Store(id="round-store", data={"round": 0, "active": False}),

        html.H1("BoxingTracker Live Dashboard", style={"marginBottom": "24px"}),

        # ── Round controls ────────────────────────────────────────────────────
        html.Div(
            style={"display": "flex", "gap": "12px", "marginBottom": "24px", "alignItems": "center"},
            children=[
                html.Button("▶ Start Round", id="btn-round-start", n_clicks=0,
                            disabled=False, style=_BTN_START),
                html.Button("⏹ End Round",   id="btn-round-end",   n_clicks=0,
                            disabled=True,  style={**_BTN_END, **_BTN_OFF}),
                html.Button("⏭ Next Round",  id="btn-next-round",  n_clicks=0,
                            disabled=True,  style={**_BTN_NEXT, **_BTN_OFF}),
                html.Div(id="round-status",
                         style={"lineHeight": "40px", "color": "#aaaaaa", "fontSize": "15px"}),
            ],
        ),

        # ── Per-round tabs ────────────────────────────────────────────────────
        dcc.Tabs(
            id="round-tabs",
            value="overview",
            colors={"border": "#2a2d3e", "primary": "#f0c040", "background": "#1e2130"},
            children=[
                dcc.Tab(label="Overview", value="overview",
                        style=_TAB_STYLE, selected_style=_TAB_SELECTED_STYLE),
            ],
        ),

        html.Div(id="tab-content", style={"marginTop": "24px"}),
    ],
)


# ── Reusable UI helpers ───────────────────────────────────────────────────────

def _counter_card(label: str, value: int, colour: str) -> html.Div:
    return html.Div(
        style={
            "backgroundColor": "#1e2130", "borderRadius": "8px",
            "padding": "16px 24px", "minWidth": "140px",
            "borderTop": f"3px solid {colour}",
        },
        children=[
            html.Div(label, style={"fontSize": "13px", "color": "#aaaaaa"}),
            html.Div(str(value), style={"fontSize": "36px", "fontWeight": "bold", "color": colour}),
        ],
    )


def _punch_chart(punch_events: list) -> go.Figure:
    counts: dict = defaultdict(lambda: {"landed": 0, "missed": 0})
    for e in punch_events:
        bucket = "landed" if e.get("landed") else "missed"
        counts[e.get("punch_type", "unknown")][bucket] += 1

    types  = list(counts.keys())
    landed = [counts[t]["landed"] for t in types]
    missed = [counts[t]["missed"] for t in types]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Landed", x=types, y=landed, marker_color="#2ecc71"))
    fig.add_trace(go.Bar(name="Missed", x=types, y=missed, marker_color="#e74c3c"))
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
        margin={"t": 10, "b": 10, "l": 10, "r": 10},
        legend={"orientation": "h", "y": -0.2},
        xaxis={"gridcolor": "#2a2d3e"},
        yaxis={"gridcolor": "#2a2d3e", "title": "Punches"},
    )
    return fig


def _event_table(events: list):
    rows = []
    for e in events:
        ts    = _fmt_ts(e["ts"]) if "ts" in e else ""
        etype = e.get("event_type", "punch")
        if etype == "round_start":
            rows.append({"time": ts, "type": f"▶ Round {e.get('round', '')} start",
                         "fighter": "", "arm": "", "punch": "", "landed": ""})
        elif etype == "round_end":
            dur = e.get("duration_s")
            label = (f"⏹ Round {e.get('round', '')} end ({dur:.0f}s)"
                     if dur else f"⏹ Round {e.get('round', '')} end")
            rows.append({"time": ts, "type": label,
                         "fighter": "", "arm": "", "punch": "", "landed": ""})
        else:
            rows.append({
                "time":    ts,
                "type":    "punch",
                "fighter": str(e.get("fighter_id", "")),
                "arm":     e.get("arm", ""),
                "punch":   e.get("punch_type", ""),
                "landed":  "✓" if e.get("landed") else "✗",
            })

    if not rows:
        return html.Span("No events yet.", style={"color": "#aaaaaa"})

    return dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in ["time", "type", "fighter", "arm", "punch", "landed"]],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#1e2130", "color": "white",
                      "fontWeight": "bold", "border": "1px solid #2a2d3e"},
        style_cell={"backgroundColor": "#0e1117", "color": "white",
                    "border": "1px solid #2a2d3e", "padding": "6px 12px", "textAlign": "left"},
        style_data_conditional=[
            {"if": {"filter_query": '{landed} = "✓"', "column_id": "landed"},
             "color": "#2ecc71", "fontWeight": "bold"},
            {"if": {"filter_query": '{landed} = "✗"', "column_id": "landed"},
             "color": "#e74c3c", "fontWeight": "bold"},
            {"if": {"filter_query": '{type} contains "Round"'},
             "backgroundColor": "#1e2130", "fontWeight": "bold", "color": "#f0c040"},
        ],
        page_size=50,
        sort_action="native",
    )


def _heatmap_section(state: dict) -> list:
    heatmaps = state.get("heatmaps")
    if not heatmaps:
        return []
    figs = []
    for fid, hm_list in sorted(heatmaps.items(), key=lambda kv: int(kv[0])):
        hm = np.array(hm_list, dtype=np.float32)
        if hm.size == 0:
            continue
        fig = go.Figure(go.Heatmap(
            z=hm, colorscale="Hot", zmin=0, zmax=max(float(hm.max()), 1.0), showscale=True,
        ))
        fig.update_layout(
            title={"text": f"Fighter {fid}  ({int(hm.sum())} frames)",
                   "font": {"color": "white", "size": 13}},
            paper_bgcolor="#1e2130", plot_bgcolor="#1e2130", font_color="white",
            margin={"t": 40, "b": 40, "l": 40, "r": 40}, width=280, height=280,
            xaxis={"title": "← left · right →", "showticklabels": False, "showgrid": False},
            yaxis={"title": "near · far →", "showticklabels": False,
                   "showgrid": False, "autorange": "reversed"},
        )
        figs.append(dcc.Graph(figure=fig, config={"displayModeBar": False}))

    if not figs:
        return []
    return [
        html.H3("Ring Heatmaps", style={"marginTop": "32px", "marginBottom": "8px"}),
        html.Div(figs, style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
    ]


def _round_page(events: list, state: dict, round_num: int | None) -> html.Div:
    """Build the stats content for a given round (None = all-session overview)."""
    if round_num is None:
        punch_events = [e for e in events if e.get("event_type", "punch") == "punch"]
        total  = state.get("total_punches", len(punch_events))
        landed = state.get("landed",  sum(1 for e in punch_events if e.get("landed")))
        missed = state.get("missed",  total - landed)
        table_events = events
        extra = _heatmap_section(state)
    else:
        punch_events = [e for e in events
                        if e.get("event_type", "punch") == "punch" and e.get("round") == round_num]
        total  = len(punch_events)
        landed = sum(1 for e in punch_events if e.get("landed"))
        missed = total - landed
        table_events = [
            e for e in events
            if e.get("round") == round_num
            or (e.get("event_type") in ("round_start", "round_end") and e.get("round") == round_num)
        ]

        # Duration badge
        end_ev  = next((e for e in table_events if e.get("event_type") == "round_end"), None)
        dur_str = ""
        if end_ev and end_ev.get("duration_s"):
            dur = end_ev["duration_s"]
            dur_str = f"Duration: {int(dur // 60)}:{int(dur % 60):02d}"
        extra = ([html.Div(dur_str, style={"color": "#aaaaaa", "marginBottom": "16px"})]
                 if dur_str else [])

    return html.Div([
        *extra,
        html.Div(
            [_counter_card("Total Punches", total, "#ffffff"),
             _counter_card("Landed", landed, "#2ecc71"),
             _counter_card("Missed", missed, "#e74c3c")],
            style={"display": "flex", "gap": "16px", "marginBottom": "32px"},
        ),
        html.H3("Punch Type Distribution", style={"marginBottom": "8px"}),
        dcc.Graph(figure=_punch_chart(punch_events), config={"displayModeBar": False}),
        html.H3("Event Log", style={"marginTop": "32px", "marginBottom": "8px"}),
        _event_table(table_events),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("round-store",      "data"),
    Output("round-status",     "children"),
    Output("btn-round-start",  "disabled"),
    Output("btn-round-start",  "style"),
    Output("btn-round-end",    "disabled"),
    Output("btn-round-end",    "style"),
    Output("btn-next-round",   "disabled"),
    Output("btn-next-round",   "style"),
    Input("btn-round-start",   "n_clicks"),
    Input("btn-round-end",     "n_clicks"),
    Input("btn-next-round",    "n_clicks"),
    Input("interval",          "n_intervals"),
    State("round-store",       "data"),
    prevent_initial_call=True,
)
def handle_round_controls(start_clicks, end_clicks, next_clicks, _n, store):
    triggered = ctx.triggered_id
    if triggered == "btn-round-start":
        _write_command("round_start")
    elif triggered == "btn-round-end":
        _write_command("round_end")
    elif triggered == "btn-next-round":
        _write_command("round_next")

    # Always re-derive state from the event log so buttons stay in sync
    rs     = _infer_round_state(load_events())
    active = rs["active"]
    rnum   = rs["round"]
    store  = {"round": rnum, "active": active}

    if active:
        status = f"🔴 Round {rnum} in progress"
    elif rnum > 0:
        status = f"⏸ Round {rnum} ended"
    else:
        status = "No round started"

    start_style = _BTN_START if not active else {**_BTN_START, **_BTN_OFF}
    end_style   = _BTN_END   if active     else {**_BTN_END,   **_BTN_OFF}
    next_style  = _BTN_NEXT  if active     else {**_BTN_NEXT,  **_BTN_OFF}

    return (
        store, status,
        active,   start_style,   # Start: disabled while round active
        not active, end_style,   # End: disabled while no round active
        not active, next_style,  # Next: disabled while no round active
    )


@app.callback(
    Output("round-tabs", "children"),
    Input("interval", "n_intervals"),
)
def update_tabs(_n):
    rs        = _infer_round_state(load_events())
    max_round = rs["round"]

    tabs = [dcc.Tab(label="Overview", value="overview",
                    style=_TAB_STYLE, selected_style=_TAB_SELECTED_STYLE)]
    for r in range(1, max_round + 1):
        live  = rs["active"] and r == max_round
        label = f"Round {r} 🔴" if live else f"Round {r}"
        tabs.append(dcc.Tab(label=label, value=f"round-{r}",
                            style=_TAB_STYLE, selected_style=_TAB_SELECTED_STYLE))
    return tabs


@app.callback(
    Output("tab-content", "children"),
    Input("round-tabs",   "value"),
    Input("interval",     "n_intervals"),
)
def update_tab_content(tab_value, _n):
    events = load_events()
    state  = load_state()

    if not tab_value or tab_value == "overview":
        return _round_page(events, state, round_num=None)

    if tab_value.startswith("round-"):
        try:
            return _round_page(events, state, round_num=int(tab_value.split("-", 1)[1]))
        except (ValueError, IndexError):
            pass

    return html.Span("Select a tab above.", style={"color": "#aaaaaa"})


if __name__ == "__main__":
    app.run(debug=False)
