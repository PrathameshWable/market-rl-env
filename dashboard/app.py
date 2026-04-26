"""Streamlit dashboard for the Meta-Trader Stage 1 / M7B agent.

Run locally with::

    pip install -r requirements-dashboard.txt
    streamlit run dashboard/app.py

Designed to be recorded as a 2-3 minute video walkthrough. Every number on
screen is loaded from JSON files saved by ``training/evaluate.py`` /
``training/tom_probes.py`` so judges can verify the artifacts independently.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard import loaders, sim

# ---------------------------------------------------------------------------
# Page config + global CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Meta-Trader · Stage 1 ToM Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 4rem; max-width: 1500px; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
}
h1, h2, h3 { color: #f0f6fc !important; letter-spacing: -0.02em; }
.hero {
    background: linear-gradient(135deg, #1a3a5c 0%, #2d5a8b 50%, #1f4068 100%);
    padding: 1.6rem 2rem; border-radius: 16px; margin-bottom: 1.2rem;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
}
.hero h1 { margin: 0 0 0.4rem 0; font-size: 2.2rem; }
.hero p  { color: #c9d1d9; font-size: 1.05rem; margin: 0; }
.kpi-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2129 100%);
    padding: 1.2rem 1.4rem; border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 2px 12px rgba(0,0,0,0.25);
    height: 100%;
}
.kpi-label { color: #8b949e; font-size: 0.78rem; text-transform: uppercase;
    letter-spacing: 0.08em; margin: 0 0 0.45rem 0; font-weight: 600; }
.kpi-value { color: #f0f6fc; font-size: 1.8rem; font-weight: 700;
    margin: 0; line-height: 1.1; }
.kpi-delta-pos { color: #3fb950; font-size: 0.85rem; font-weight: 600; }
.kpi-delta-neg { color: #f85149; font-size: 0.85rem; font-weight: 600; }
.kpi-sub  { color: #6e7681; font-size: 0.78rem; margin-top: 0.3rem; }
.story-card {
    background: #161b22; padding: 1.2rem 1.4rem; border-radius: 12px;
    border-left: 3px solid #58a6ff; margin: 0.6rem 0;
}
.story-card h4 { color: #58a6ff; margin: 0 0 0.5rem 0; font-size: 0.95rem;
    text-transform: uppercase; letter-spacing: 0.06em; }
.story-card p { color: #c9d1d9; margin: 0; line-height: 1.55; }
.tag {
    display: inline-block; background: #1f6feb33; color: #58a6ff;
    padding: 0.15rem 0.55rem; border-radius: 6px; font-size: 0.75rem;
    font-weight: 600; margin-right: 0.4rem;
}
.tag-warn { background: #d2992233; color: #f0b842; }
.tag-good { background: #2ea04333; color: #3fb950; }
.tag-bad  { background: #cf222e33; color: #f85149; }
hr { border-color: #21262d !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — run selector
# ---------------------------------------------------------------------------

st.sidebar.markdown("# 📈 Meta-Trader")
st.sidebar.caption("Stage 1 ToM agent · Qwen2.5-3B + LoRA + GRPO")

# Re-scan training/runs/ on every interaction so newly finished runs
# (e.g. aux_direction, no_curriculum) appear in the dropdown without
# restarting Streamlit.
if st.sidebar.button("🔄 Refresh runs", width='stretch'):
    st.rerun()

runs = loaders.discover_runs()
run_names = [r.name for r in runs] or ["(no runs found)"]
default_idx = next((i for i, r in enumerate(runs) if r.name.startswith("stage1")), 0)

selected_name = st.sidebar.selectbox("Run", run_names, index=default_idx)
selected_run = next((r for r in runs if r.name == selected_name), None)

st.sidebar.markdown("---")
section = st.sidebar.radio(
    "Section",
    [
        "🏠 Overview",
        "🎬 Live Market Replay",
        "🧠 Theory of Mind",
        "📊 Training Curves",
        "🛠 Methodology",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "All numbers are loaded from `training/runs/<run>/*.json`. "
    "Use the **Live Market Replay** section to drive a video demo."
)


def _fmt_pct(x, signed: bool = True) -> str:
    if x is None:
        return "—"
    sign = "+" if signed and x >= 0 else ""
    return f"{sign}{x*100:.2f}%"


def _kpi(label: str, value: str, sub: str = "", delta: str = "", delta_kind: str = "pos") -> None:
    delta_html = ""
    if delta:
        cls = "kpi-delta-pos" if delta_kind == "pos" else "kpi-delta-neg"
        delta_html = f'<div class="{cls}">{delta}</div>'
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section: Overview
# ---------------------------------------------------------------------------

def render_overview(run):
    if run is None:
        st.warning("No run found under `training/runs/`. Train Stage 1 first.")
        return

    kpi = loaders.summary_kpis(run)

    st.markdown(
        f"""
        <div class="hero">
          <h1>From random trader → market-aware agent</h1>
          <p>Stage 1 of a Theory-of-Mind market-trading curriculum. Qwen2.5-3B
          fine-tuned with SFT + GRPO on a multi-bot order-book environment,
          evaluated against scripted baselines and probed for emergent ToM
          behaviour.&nbsp;&nbsp;<span class="tag">Run · {run.name}</span></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Headline KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        trained_pnl = kpi.get("trained_pnl_mean")
        random_pnl = kpi.get("random_pnl_mean")
        delta = ""
        if trained_pnl is not None and random_pnl is not None:
            improvement = (trained_pnl - random_pnl) * 100
            delta = f"▲ {improvement:+.2f} pts vs Random"
        _kpi(
            "Mean P&L (trained)",
            _fmt_pct(trained_pnl),
            sub=f"{kpi.get('trained_n_episodes', 0)} episodes · medium difficulty",
            delta=delta,
            delta_kind="pos" if (trained_pnl or 0) > (random_pnl or 0) else "neg",
        )
    with c2:
        _kpi(
            "Parse success rate",
            _fmt_pct(kpi.get("trained_parse_rate"), signed=False),
            sub="JSON-action format obeyed every turn",
        )
    with c3:
        prb1 = kpi.get("probe1_improvement")
        _kpi(
            "Probe 1 — price efficiency",
            f"{prb1:.2f}" if prb1 is not None else "—",
            sub="$/share error reduction (init → final)",
            delta=f"start: {kpi.get('probe1_initial_err'):.2f} → end: {kpi.get('probe1_final_err'):.2f}"
            if kpi.get("probe1_initial_err") is not None else "",
            delta_kind="pos",
        )
    with c4:
        prb2 = kpi.get("probe2_alignment")
        _kpi(
            "Probe 2 — signal alignment",
            _fmt_pct(prb2, signed=False) if prb2 is not None else "—",
            sub=f"trades that match private-signal direction (n={kpi.get('probe2_n_active', 0)})",
        )

    st.markdown(" ")

    # Story cards — what does this all mean?
    sa, sb = st.columns([1.4, 1])
    with sa:
        st.markdown("### What the agent learned")
        st.markdown(
            f"""
            <div class="story-card">
              <h4>Beat the random baseline</h4>
              <p>The trained policy averages <b>{_fmt_pct(kpi.get('trained_pnl_mean'))}</b>
              normalised P&amp;L vs <b>{_fmt_pct(kpi.get('random_pnl_mean'))}</b> for a random
              actor. Variance is high (σ = {kpi.get('trained_pnl_std', 0):.3f}) because the
              agent now <i>participates</i> 100% of turns instead of randomly skipping —
              the upside of taking real positions cuts both ways.</p>
            </div>
            <div class="story-card">
              <h4>Reads the order book</h4>
              <p>By the end of an episode, the agent's mid-price is on average
              <b>{kpi.get('probe1_final_err', 0):.2f} $/share</b> away from the true value,
              down from <b>{kpi.get('probe1_initial_err', 0):.2f}</b> at the start. That's a
              <b>{kpi.get('probe1_improvement', 0):.2f} $/share</b> improvement attributable
              to its trades — the order book is being moved <i>toward</i> truth.</p>
            </div>
            <div class="story-card">
              <h4>Uses its private signals</h4>
              <p>When the agent trades, <b>{_fmt_pct(kpi.get('probe2_alignment'), signed=False)}</b>
              of those trades are in the direction implied by its noisy private signals.
              That's not just memorisation — the LLM is conditioning its action on
              numeric features inside the prompt.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sb:
        st.markdown("### Where it still falls short")
        prb3 = kpi.get("probe3_accuracy")
        prb3_pct = _fmt_pct(prb3, signed=False) if prb3 is not None else "—"
        st.markdown(
            f"""
            <div class="story-card" style="border-left-color:#f0b842;">
              <h4>ToM probe 3 — value direction</h4>
              <p>Asked “Is the true value above or below 50?”, the agent answers correctly
              <b>{prb3_pct}</b> of the time
              ({kpi.get('probe3_n_correct', 0)}/{kpi.get('probe3_n_probes', 0)} probes).
              Inspection shows a <b>“say above” bias</b> — the model hallucinates the
              majority class instead of reading its signals.</p>
              <p style="margin-top:0.6rem;">M7B fixes this with an
              <span class="tag">aux-direction reward</span>
              that penalises trades opposed to the signal sum, plus a
              <span class="tag">curriculum ablation</span> to test whether
              easy → hard ordering matters.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(" ")

    # P&L comparison chart
    st.markdown("### P&L distribution by policy")
    evals = loaders.load_all_evals(run)
    rows: list[dict] = []
    colors = {
        "trained": "#3fb950",
        "informed": "#58a6ff",
        "random": "#f85149",
        "hold": "#8b949e",
    }
    for label, data in evals.items():
        eps = data.get("episodes") or []
        if eps:
            for e in eps:
                rows.append({"policy": label, "pnl": e["pnl_normalized"]})
        else:
            # No per-episode breakdown — fall back to mean as a single bar.
            rows.append({"policy": label, "pnl": data.get("mean_pnl_normalized", 0)})
    if rows:
        df = pd.DataFrame(rows)
        fig = go.Figure()
        # Order: best policy on the left + at the top of the legend.
        # "trained" gets legendrank=1 so it appears first regardless of
        # how Plotly renders trace order.
        for rank, label in enumerate(["trained", "informed", "random", "hold"], start=1):
            sub = df[df["policy"] == label]
            if sub.empty:
                continue
            fig.add_trace(go.Box(
                y=sub["pnl"] * 100,
                name=label,
                marker_color=colors.get(label, "#58a6ff"),
                boxmean=True, boxpoints="all", jitter=0.4, pointpos=0,
                legendrank=rank,
            ))
        fig.update_layout(
            template="plotly_dark",
            height=380,
            yaxis_title="Normalised P&L (% of initial cash)",
            margin=dict(l=20, r=20, t=10, b=10),
            showlegend=False,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#6e7681")
        st.plotly_chart(fig, width='stretch')

    st.caption(
        f"📁 Source: `training/runs/{run.name}/eval_*.json`. "
        f"Means: trained {_fmt_pct(kpi.get('trained_pnl_mean'))}, "
        f"random {_fmt_pct(kpi.get('random_pnl_mean'))}, "
        f"hold {_fmt_pct(kpi.get('hold_pnl_mean'))}."
    )


# ---------------------------------------------------------------------------
# Section: Live Market Replay
# ---------------------------------------------------------------------------

def render_replay():
    st.markdown("## 🎬 Live market replay")
    st.caption(
        "Run a single market session with a chosen *surrogate trainable agent* "
        "and watch the order book evolve tick-by-tick. Compare a Random policy to "
        "the Informed (oracle) ceiling on the same scenario to see how much the "
        "trained model has to gain. All scripted bots run on CPU in <1 second."
    )

    cfg_col, run_col = st.columns([2, 1])
    with cfg_col:
        c1, c2, c3 = st.columns(3)
        with c1:
            seed = st.number_input("Seed", value=100, min_value=0, max_value=999_999, step=1)
        with c2:
            difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
        with c3:
            surrogate_label = st.selectbox(
                "Surrogate agent",
                list(sim.SURROGATES.keys()),
                index=1,  # Informed by default — most visually interesting
            )
        surrogate = sim.SURROGATES[surrogate_label]

    with run_col:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        run_btn = st.button("▶ Run scenario", width='stretch', type="primary")

    if run_btn or "replay" not in st.session_state:
        with st.spinner("Simulating 50 turns..."):
            st.session_state.replay = sim.run_replay(
                seed=int(seed),
                difficulty=difficulty,
                surrogate=surrogate,
            )
        # Reset the turn slider on a new run. Drop the widget key so that
        # the next slider() call rebuilds it cleanly without Streamlit
        # complaining about default-vs-session-state.
        st.session_state.pop("frame_idx", None)

    replay = st.session_state.replay
    frames = replay["frames"]
    total = len(frames)
    # Default the slider to the last frame on first display, but only
    # touch session_state if the widget hasn't been created yet — touching
    # it after creation triggers Streamlit's "value set both ways" warning.
    if "frame_idx" not in st.session_state:
        st.session_state.frame_idx = total

    # Episode-level summary card
    final_pnl = replay["final_pnl"]
    agent_pnl = final_pnl.get("agent_1", 0.0)
    tv = replay["true_value"]

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        _kpi(
            "Surrogate", surrogate_label.split(" (")[0],
            sub=f"seed = {replay['seed']} · {replay['difficulty']}",
        )
    with s2:
        _kpi(
            "Final P&L (agent_1)",
            _fmt_pct(agent_pnl),
            sub=f"true value = ${tv:.2f}",
            delta="positive" if agent_pnl >= 0 else "negative",
            delta_kind="pos" if agent_pnl >= 0 else "neg",
        )
    with s3:
        n_trades = sum(len(f.new_trades) for f in frames)
        _kpi("Trades executed", f"{n_trades}", sub=f"across {total} turns")
    with s4:
        bot_pnls = {a: p for a, p in final_pnl.items() if a != "agent_1"}
        best = max(bot_pnls.items(), key=lambda kv: kv[1]) if bot_pnls else (None, None)
        worst = min(bot_pnls.items(), key=lambda kv: kv[1]) if bot_pnls else (None, None)
        _kpi(
            "Best / worst bot",
            f"{best[0] or '—'} / {worst[0] or '—'}",
            sub=f"{_fmt_pct(best[1])} / {_fmt_pct(worst[1])}" if best[0] else "—",
        )

    st.markdown(" ")

    # Frame slider with auto-play
    sl_col, btn_col = st.columns([4, 1])
    with sl_col:
        idx = st.slider("Turn", min_value=1, max_value=total, key="frame_idx")
    with btn_col:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        play_btn = st.button("▶ Auto-play", width="stretch")

    placeholder = st.empty()

    def render_frame(i: int):
        with placeholder.container():
            _render_replay_frame(replay, i)

    if play_btn:
        # During auto-play we drive the placeholder directly. Streamlit
        # widgets are read-only after creation, so we don't try to move
        # the slider — the playback panel below it animates instead.
        for k in range(1, total + 1):
            render_frame(k)
            time.sleep(0.18)
    else:
        render_frame(idx)


def _render_replay_frame(replay: dict, upto: int) -> None:
    frames = replay["frames"]
    tv = replay["true_value"]

    # Build time series up to ``upto``
    turns = [f.turn for f in frames[:upto]]
    mids = [f.mid for f in frames[:upto]]
    last_trades = [f.last_trade_price for f in frames[:upto]]

    pnl_by_agent: dict[str, list[float]] = {}
    for f in frames[:upto]:
        for a in f.agents:
            pnl_by_agent.setdefault(a.agent_id, []).append(a.pnl_marked)

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.6, 0.4],
        row_heights=[0.55, 0.45],
        specs=[
            [{"type": "scatter"}, {"type": "table"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
        subplot_titles=(
            "Mid price vs. true value",
            "Top-of-book (current turn)",
            "P&L by agent (cash + shares × mid)",
            "Trades this turn",
        ),
        horizontal_spacing=0.07, vertical_spacing=0.18,
    )

    fig.add_trace(
        go.Scatter(
            x=turns, y=mids, mode="lines+markers",
            line=dict(color="#58a6ff", width=2.4), marker=dict(size=5),
            name="mid",
        ),
        row=1, col=1,
    )
    if any(p is not None for p in last_trades):
        fig.add_trace(
            go.Scatter(
                x=turns, y=last_trades, mode="markers",
                marker=dict(color="#f0b842", size=7, symbol="triangle-down"),
                name="last trade",
            ),
            row=1, col=1,
        )
    # True-value reference line. We use ``add_shape`` instead of ``add_hline``
    # because Plotly 6 raises a PlotlyKeyError when ``add_hline`` is called on
    # a figure that also contains non-cartesian traces (the order-book Table).
    fig.add_shape(
        type="line", xref="x domain", yref="y",
        x0=0, x1=1, y0=tv, y1=tv,
        line=dict(color="#3fb950", width=2, dash="dash"),
    )
    fig.add_annotation(
        text=f"true value = ${tv:.2f}",
        xref="x domain", yref="y",
        x=0.98, y=tv, xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(color="#3fb950"),
    )

    # Order book table for current frame
    cur = frames[upto - 1]
    bids = cur.bids_top5 + [sim.BookLevel(price=0, quantity=0)] * (5 - len(cur.bids_top5))
    asks = cur.asks_top5 + [sim.BookLevel(price=0, quantity=0)] * (5 - len(cur.asks_top5))
    rows = []
    for i in range(5):
        rows.append([
            f"{asks[i].quantity}" if asks[i].quantity else "",
            f"${asks[i].price:.2f}" if asks[i].price else "",
            f"${bids[i].price:.2f}" if bids[i].price else "",
            f"{bids[i].quantity}" if bids[i].quantity else "",
        ])
    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Ask qty</b>", "<b>Ask</b>", "<b>Bid</b>", "<b>Bid qty</b>"],
                fill_color="#1c2129",
                font=dict(color="#f0f6fc", size=12),
                align="center", height=28,
            ),
            cells=dict(
                values=list(zip(*rows)),
                fill_color=[["#161b22"] * 5, ["#3a1f1f"] * 5, ["#1f3a25"] * 5, ["#161b22"] * 5],
                font=dict(color=["#f0f6fc", "#f85149", "#3fb950", "#f0f6fc"], size=12),
                align="center", height=24,
            ),
        ),
        row=1, col=2,
    )

    pnl_colors = {
        "agent_1": "#3fb950",
        "random_bot": "#f85149",
        "momentum_bot": "#a371f7",
        "mean_rev_bot": "#f0b842",
        "mm_bot": "#58a6ff",
        "informed_bot": "#ff7b72",
    }
    for agent, series in pnl_by_agent.items():
        fig.add_trace(
            go.Scatter(
                x=turns, y=series, mode="lines",
                name=agent,
                line=dict(color=pnl_colors.get(agent, "#8b949e"),
                          width=3 if agent == "agent_1" else 1.5,
                          dash="solid" if agent == "agent_1" else "dot"),
            ),
            row=2, col=1,
        )
    # Zero-PnL reference line (subplot row=2,col=1 -> x3/y3 in a 2x2 grid).
    fig.add_shape(
        type="line", xref="x3 domain", yref="y3",
        x0=0, x1=1, y0=0, y1=0,
        line=dict(color="#6e7681", dash="dot"),
    )

    # Trades this turn — bar chart of buy vs sell prices
    trades = cur.new_trades
    if trades:
        buy_prices = [t.price for t in trades if t.aggressor_side == "buy"]
        sell_prices = [t.price for t in trades if t.aggressor_side == "sell"]
        if buy_prices:
            fig.add_trace(go.Scatter(
                x=list(range(len(buy_prices))), y=buy_prices, mode="markers",
                marker=dict(color="#3fb950", size=14, symbol="triangle-up"),
                name="buy", showlegend=False,
            ), row=2, col=2)
        if sell_prices:
            fig.add_trace(go.Scatter(
                x=list(range(len(sell_prices))), y=sell_prices, mode="markers",
                marker=dict(color="#f85149", size=14, symbol="triangle-down"),
                name="sell", showlegend=False,
            ), row=2, col=2)
        # True-value reference on the trades subplot (row=2,col=2 -> x4/y4).
        fig.add_shape(
            type="line", xref="x4 domain", yref="y4",
            x0=0, x1=1, y0=tv, y1=tv,
            line=dict(color="#3fb950", dash="dash"),
        )
    else:
        fig.add_annotation(
            text="(no trades this turn)", xref="x4", yref="y4",
            x=0, y=0, showarrow=False,
            font=dict(color="#6e7681"), row=2, col=2,
        )

    fig.update_layout(
        template="plotly_dark", height=720,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True, legend=dict(
            orientation="h", x=0, y=-0.08,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(title="turn", row=1, col=1)
    fig.update_yaxes(title="$/share", row=1, col=1)
    fig.update_xaxes(title="turn", row=2, col=1)
    fig.update_yaxes(title="$ marked-to-mid", row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=2)
    fig.update_yaxes(title="trade price", row=2, col=2)

    st.plotly_chart(fig, width='stretch')

    # Action card for current turn
    act = cur.surrogate_action
    parts = [f"<span class='tag'>turn {cur.turn}</span>",
             f"<span class='tag'>{act.get('type', '?').upper()}</span>"]
    if "price" in act:
        parts.append(f"<span class='tag'>@ ${act['price']:.2f}</span>")
    if "quantity" in act:
        parts.append(f"<span class='tag'>qty {act['quantity']}</span>")
    if cur.new_trades:
        parts.append(f"<span class='tag tag-good'>{len(cur.new_trades)} trade(s) filled</span>")
    if cur.done:
        parts.append("<span class='tag tag-warn'>EPISODE END</span>")
    st.markdown(
        f"""
        <div class="story-card">
          <h4>Surrogate action this turn</h4>
          <div style="font-size:1.0rem;">{' '.join(parts)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section: Theory of Mind
# ---------------------------------------------------------------------------

def render_tom(run):
    if run is None:
        st.warning("No run found.")
        return

    probes = loaders.load_probes(run)
    direction = loaders.load_direction_inference(run)

    st.markdown("## 🧠 Theory-of-mind probes")
    st.caption(
        "Three probes that test whether the agent has internalised an *implicit "
        "model* of the unobservable true value, not just memorised reward patterns. "
        "All numbers come from `tom_probes_*.json` and `tom_direction_inference.json`."
    )

    # --- Probe 1: price efficiency ---------------------------------------
    st.markdown("### Probe 1 — Price efficiency")
    st.markdown(
        "*If the agent moves the order book toward the true value, the "
        "mean |mid − true_value| should fall over the episode.*"
    )

    fig = go.Figure()
    colors = {
        "trained": "#3fb950",
        "informed": "#58a6ff",
        "random": "#f85149",
        "hold": "#8b949e",
    }
    # Draw weakest baselines first so the "good" curves render on top.
    # legendrank pins the trained agent + informed oracle to the top
    # of the legend regardless of trace order.
    legend_priority = {"trained": 1, "informed": 2, "random": 3, "hold": 4}
    for label in ["hold", "random", "informed", "trained"]:
        block = probes.get(label, {})
        pe = block.get("price_efficiency") if isinstance(block, dict) else None
        if pe is None or "mean_error_by_turn" not in pe:
            continue
        ys = pe["mean_error_by_turn"]
        xs = list(range(len(ys)))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            name=f"{label} (Δ={pe.get('improvement', 0):.2f})",
            line=dict(color=colors.get(label, "#c9d1d9"),
                      width=3 if label == "trained" else 1.6,
                      dash="solid" if label in ("trained", "informed") else "dot"),
            legendrank=legend_priority.get(label, 99),
        ))
    fig.update_layout(
        template="plotly_dark", height=380,
        margin=dict(l=20, r=20, t=10, b=10),
        xaxis_title="turn",
        yaxis_title="mean |mid − true value|  ($/share)",
        legend=dict(orientation="h", y=-0.18, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, width='stretch')

    # --- Probe 2: signal alignment ---------------------------------------
    st.markdown("### Probe 2 — Signal alignment")
    st.markdown(
        "*If the agent uses its private signals, its *direction of trade* should "
        "match the sign of the noisy signal sum (when it's strong enough).*"
    )

    rows = []
    # Trained first (left-most bar); informed (oracle) second; random last.
    for label in ["trained", "informed", "random"]:
        block = probes.get(label, {})
        sa = block.get("signal_alignment") if isinstance(block, dict) else None
        if sa is None:
            continue
        rows.append({
            "policy": label,
            "alignment_rate": sa.get("alignment_rate", 0),
            "n_active": sa.get("n_active", 0),
            "n_aligned": sa.get("n_aligned", 0),
            "n_misaligned": sa.get("n_misaligned", 0),
        })
    if rows:
        df = pd.DataFrame(rows)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=df["policy"], y=df["alignment_rate"] * 100,
            text=[f"{v*100:.1f}%" for v in df["alignment_rate"]],
            textposition="outside",
            marker_color=[colors.get(p, "#58a6ff") for p in df["policy"]],
        ))
        fig2.add_hline(y=50, line_dash="dot", line_color="#6e7681",
                       annotation_text="random chance", annotation_position="bottom right")
        fig2.update_layout(
            template="plotly_dark", height=320, showlegend=False,
            margin=dict(l=20, r=20, t=20, b=10),
            yaxis_title="trades aligned with signal sign (%)",
            yaxis_range=[0, 110],
        )
        a, b = st.columns([2, 1])
        with a:
            st.plotly_chart(fig2, width='stretch')
        with b:
            st.dataframe(
                df.assign(alignment_rate=lambda d: d["alignment_rate"].map("{:.1%}".format)),
                width='stretch', hide_index=True,
            )

    # --- Probe 3: direction inference ------------------------------------
    st.markdown("### Probe 3 — Direction inference (zero-shot)")
    st.markdown(
        "*Mid-episode the agent is asked: 'Is the true value above or below 50?' "
        "If it has built an internal estimate, accuracy should beat 50%.*"
    )

    if direction:
        accuracy = direction.get("accuracy", 0.0)
        n_correct = direction.get("n_correct", 0)
        n_total = direction.get("n_probes", 0)
        chance = direction.get("chance_baseline", 0.5)

        c1, c2, c3 = st.columns(3)
        with c1:
            _kpi(
                "Accuracy",
                _fmt_pct(accuracy, signed=False),
                sub=f"{n_correct}/{n_total} probes",
                delta=f"vs chance {_fmt_pct(chance, signed=False)}",
                delta_kind="pos" if accuracy > chance else "neg",
            )
        with c2:
            probes_list = direction.get("probes", [])
            ans_above = sum(1 for p in probes_list if p.get("model_answer") == "above")
            _kpi(
                "Bias check — said 'above'",
                f"{ans_above}/{len(probes_list)}",
                sub="50% would be unbiased",
                delta=f"{ans_above / max(1, len(probes_list)) * 100:.0f}% 'above'",
                delta_kind="neg" if ans_above / max(1, len(probes_list)) > 0.6 else "pos",
            )
        with c3:
            tv_above = sum(1 for p in probes_list if p.get("ground_truth") == "above")
            _kpi(
                "Ground truth — was 'above'",
                f"{tv_above}/{len(probes_list)}",
                sub="balance of the probe set",
            )

        # Confusion matrix
        if probes_list:
            mat = {("above", "above"): 0, ("above", "below"): 0,
                   ("below", "above"): 0, ("below", "below"): 0}
            for p in probes_list:
                gt = p.get("ground_truth")
                ans = p.get("model_answer")
                if gt in ("above", "below") and ans in ("above", "below"):
                    mat[(gt, ans)] += 1
            z = [
                [mat[("above", "above")], mat[("above", "below")]],
                [mat[("below", "above")], mat[("below", "below")]],
            ]
            fig3 = go.Figure(go.Heatmap(
                z=z,
                x=["Said 'above'", "Said 'below'"],
                y=["TV above 50", "TV below 50"],
                text=[[str(v) for v in row] for row in z],
                texttemplate="%{text}",
                colorscale=[[0, "#161b22"], [0.5, "#1f6feb"], [1, "#58a6ff"]],
                showscale=False,
            ))
            fig3.update_layout(
                template="plotly_dark", height=320,
                margin=dict(l=20, r=20, t=30, b=10),
                title="Confusion matrix · ground truth vs model answer",
            )
            st.plotly_chart(fig3, width='stretch')

        st.info(
            "**Interpretation.** The Stage 1 model leans hard toward 'above' regardless "
            "of true value. M7B's auxiliary direction-alignment reward + curriculum "
            "ablation are designed to break exactly this bias."
        )
    else:
        st.warning("No direction-inference probe data found.")


# ---------------------------------------------------------------------------
# Section: Training Curves
# ---------------------------------------------------------------------------

def render_training(run):
    if run is None:
        st.warning("No run found.")
        return
    log = loaders.load_training_log(run)
    if not log:
        st.info("No `training_log.json` for this run.")
        return

    df = pd.DataFrame(log)
    st.markdown("## 📊 GRPO training dynamics")
    st.caption(
        f"From `training/runs/{run.name}/training_log.json` · "
        f"{len(df)} log records over {df['step'].max()} GRPO steps."
    )

    # KPIs over the run
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi("Total GRPO steps", f"{int(df['step'].max())}",
             sub=f"{len(df)} log entries")
    with c2:
        last_reward = df["reward"].iloc[-1] if "reward" in df else 0
        _kpi("Final reward", f"{last_reward:+.3f}", sub="last logging step")
    with c3:
        if "completion_length" in df:
            _kpi("Mean completion length", f"{df['completion_length'].mean():.1f} tok",
                 sub="across the run")
    with c4:
        if "kl" in df:
            _kpi("Final KL", f"{df['kl'].iloc[-1]:.3f}", sub="vs reference")

    # Reward + reward_std
    if "reward" in df:
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            "Reward (rolling)",
            "Loss",
            "KL divergence vs reference",
            "Completion length (tokens)",
        ))
        roll = df["reward"].rolling(window=5, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=df["step"], y=df["reward"], mode="lines",
                                 line=dict(color="#21262d", width=1),
                                 name="reward (raw)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["step"], y=roll, mode="lines",
                                 line=dict(color="#3fb950", width=3),
                                 name="reward (5-step MA)"), row=1, col=1)
        if "loss" in df:
            fig.add_trace(go.Scatter(x=df["step"], y=df["loss"], mode="lines",
                                     line=dict(color="#f85149", width=1.6),
                                     name="loss"), row=1, col=2)
        if "kl" in df:
            fig.add_trace(go.Scatter(x=df["step"], y=df["kl"], mode="lines",
                                     line=dict(color="#a371f7", width=1.6),
                                     name="kl"), row=2, col=1)
        if "completion_length" in df:
            fig.add_trace(go.Scatter(x=df["step"], y=df["completion_length"], mode="lines",
                                     line=dict(color="#58a6ff", width=1.6),
                                     name="completion_length"), row=2, col=2)
        fig.update_layout(
            template="plotly_dark", height=620,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False,
        )
        fig.add_hline(y=0, line=dict(color="#6e7681", dash="dot"), row=1, col=1)
        st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Section: Methodology
# ---------------------------------------------------------------------------

def render_methodology():
    st.markdown("## 🛠 Methodology")
    st.caption("How the agent is trained, prompted, and evaluated.")

    a, b = st.columns(2)
    with a:
        st.markdown(
            """
            ### Training pipeline
            1. **Base model** — Qwen2.5-3B-Instruct, 4-bit (Unsloth) + LoRA r=16.
            2. **SFT warm-start** — 1500 examples from scripted-bot rollouts; teaches
               the strict JSON action format.
            3. **GRPO (TRL)** — 200 group-relative-policy steps, 4 generations per
               prompt, oracle reward = sign(true − price) * (true − price), clipped to
               ±1, normalised by `INITIAL_CASH`.
            4. **Eval (M6)** — 10 unseen seeds vs Random/Hold, plus 3 ToM probes.
            5. **M7B ablations** — toggleable curriculum + auxiliary
               direction-alignment reward to address the "say above" bias on Probe 3.
            """
        )
    with b:
        st.markdown(
            """
            ### Action format (strict JSON)
            ```json
            {
              "action_type": "buy",
              "price": 49.85,
              "quantity": 5,
              "reasoning": "signal sum is +1.2; buying below mid"
            }
            ```
            Allowed `action_type`: `buy`, `sell`, `cancel`, `hold`.

            ### Observation (excerpt)
            - top-5 bids/asks
            - last 10 trades
            - 4 private signals (Gaussian noise around true value)
            - own position (cash, shares, open orders)
            """
        )

    st.markdown("### Reward composition")
    st.markdown(
        """
        ```
        reward = clip( (true_value - price) * sign  /  INITIAL_CASH , -1, +1)

        # M7B optional aux term:
        if signal_sum and side aligns with signal:
            reward += AUX_WEIGHT * min(|signal_sum| / AUX_REF, 1.0)
        ```

        Reward is sparse at the *end of episode* in M6; in M7B's GRPO loop it is
        applied per-rollout via the surrogate `oracle_reward` function.
        """
    )

    st.markdown(
        "📄 **Read the full plan** — `blog.md` and `training/README.md` document the "
        "milestones, results, and ablation setup."
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if section == "🏠 Overview":
    render_overview(selected_run)
elif section == "🎬 Live Market Replay":
    render_replay()
elif section == "🧠 Theory of Mind":
    render_tom(selected_run)
elif section == "📊 Training Curves":
    render_training(selected_run)
elif section == "🛠 Methodology":
    render_methodology()
