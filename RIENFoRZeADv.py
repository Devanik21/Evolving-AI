"""
RIENFoRZe.py  —  Project A.L.I.V.E. NEXUS  v3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE DEFINITIVE MISSION CONTROL INTERFACE

Full backend wiring:
  world.py         →  MazeEnvironment    (DFS / Prim / Wilson / Hybrid, fog, traps, portals)
  brain.py         →  AgentBrain         (D3QN + PER + N-Step + ICM + Curriculum + Adam)
  soul.py          →  SoulCore           (Russell VA + OCEAN + IntentNLP + RelationshipEngine)
  memory_palace.py →  MemoryPalace       (Working / Episodic / Semantic / PersistentStore)
  analytics.py     →  PerformanceDashboard (Convergence + Heatmap + CapabilityScore)

Co-Investigator: Xylia
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import json
import io
import zipfile
import os
import random
import math
from collections import deque
from typing import Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# BACKEND IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
try:
    from world         import MazeEnvironment
    from brain         import AgentBrain
    from soul          import SoulCore
    from memory_palace import MemoryPalace
    from analytics     import PerformanceDashboard
    _BACKENDS_OK  = True
    _BACKEND_ERR  = ""
except ImportError as _e:
    _BACKENDS_OK  = False
    _BACKEND_ERR  = str(_e)

try:
    import plotly.graph_objects as go
    import plotly.express       as px
    from   plotly.subplots      import make_subplots
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A.L.I.V.E. NEXUS",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧬",
)

# ─────────────────────────────────────────────────────────────────────────────
# FULL CYBERPUNK CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&display=swap');

/* ── GLOBAL ── */
.stApp {
    background: radial-gradient(ellipse at 20% 10%, #0d0d2e 0%, #080818 55%, #0a1a18 100%);
    color: #c9d1d9;
    font-family: 'Segoe UI', sans-serif;
}
* { box-sizing: border-box; }

/* ── TITLE SHIMMER ── */
.nexus-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00f5ff 0%, #a855f7 40%, #f97316 70%, #00f5ff 100%);
    background-size: 300% 100%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
}
@keyframes shimmer { to { background-position: -300% 0; } }

/* ── KPI CARD ── */
.kpi-card {
    background: rgba(0, 245, 255, 0.03);
    border: 1px solid rgba(0, 245, 255, 0.12);
    border-radius: 10px; padding: 10px 14px;
    text-align: center;
    transition: border-color .25s, box-shadow .25s;
}
.kpi-card:hover {
    border-color: rgba(0, 245, 255, 0.35);
    box-shadow: 0 0 14px rgba(0, 245, 255, 0.12);
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.55rem; font-weight: 700; color: #00f5ff;
    line-height: 1.2;
}
.kpi-label {
    font-size: .65rem; color: #8b949e;
    text-transform: uppercase; letter-spacing: .1em; margin-top: 3px;
}
.kpi-sub { font-size: .6rem; color: #58a6ff; margin-top: 3px; }

/* ── PANEL HEADER ── */
.panel-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: .75rem; color: #a855f7;
    text-transform: uppercase; letter-spacing: .15em;
    padding: 4px 0 6px;
    border-bottom: 1px solid rgba(168, 85, 247, .25);
    margin-bottom: 10px;
}

/* ── INNER THOUGHT BOX ── */
.thought-box {
    background: rgba(168, 85, 247, .07);
    border-left: 3px solid #a855f7;
    border-radius: 0 8px 8px 0;
    padding: 9px 14px;
    font-size: .83rem; font-style: italic;
    color: #d8b4fe; line-height: 1.5; margin: 6px 0;
}

/* ── CHAT ── */
.chat-scroll {
    height: 290px; overflow-y: auto;
    padding: 8px;
    border: 1px solid rgba(255,255,255,.06);
    border-radius: 8px;
    background: rgba(0,0,0,.18);
    scroll-behavior: smooth;
}
.chat-user {
    background: rgba(88, 166, 255, .10);
    border-right: 3px solid #58a6ff;
    border-radius: 10px 0 0 10px;
    padding: 8px 12px; margin: 5px 0;
    text-align: right; font-size: .85rem; color: #cdd9e5;
}
.chat-ai {
    background: rgba(168, 85, 247, .09);
    border-left: 3px solid #a855f7;
    border-radius: 0 10px 10px 0;
    padding: 8px 12px; margin: 5px 0;
    font-size: .85rem; color: #d8b4fe;
}
.chat-meta { font-size: .62rem; color: #6e7681; margin-top: 2px; }

/* ── BADGE ── */
.badge {
    display: inline-block;
    padding: 2px 8px; border-radius: 16px;
    font-size: .65rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .06em;
    font-family: 'JetBrains Mono', monospace;
    margin: 1px 2px;
}
.badge-c  { background: rgba(0,245,255,.12);  color: #00f5ff; border: 1px solid rgba(0,245,255,.3);  }
.badge-p  { background: rgba(168,85,247,.12); color: #a855f7; border: 1px solid rgba(168,85,247,.3); }
.badge-g  { background: rgba(34,197,94,.12);  color: #22c55e; border: 1px solid rgba(34,197,94,.3);  }
.badge-r  { background: rgba(239,68,68,.12);  color: #ef4444; border: 1px solid rgba(239,68,68,.3);  }
.badge-o  { background: rgba(249,115,22,.12); color: #f97316; border: 1px solid rgba(249,115,22,.3); }

/* ── CONVERGENCE BANNERS ── */
.conv-base {
    border-radius: 8px; padding: 10px 14px; margin: 6px 0;
    display: flex; align-items: center; gap: 10px; font-size: .82rem;
}
.conv-warm     { background: rgba(59,130,246,.10); border: 1px solid rgba(59,130,246,.25); }
.conv-rapid    { background: rgba(34,197,94,.10);  border: 1px solid rgba(34,197,94,.25);  }
.conv-fine     { background: rgba(234,179,8,.10);  border: 1px solid rgba(234,179,8,.25);  }
.conv-ok       { background: rgba(0,245,255,.10);  border: 1px solid rgba(0,245,255,.25);  }
.conv-plateau  { background: rgba(168,85,247,.10); border: 1px solid rgba(168,85,247,.25); }
.conv-regress  { background: rgba(239,68,68,.10);  border: 1px solid rgba(239,68,68,.25);  }

/* ── PROGRESS BAR ── */
.pb-bg   { background: rgba(255,255,255,.07); border-radius: 4px; height: 7px; overflow: hidden; }
.pb-fill { height: 100%; border-radius: 4px; transition: width .4s ease; }

/* ── MEMORY TRACE / FACT ── */
.mem-trace {
    background: rgba(168,85,247,.06);
    border: 1px solid rgba(168,85,247,.14);
    border-radius: 7px; padding: 7px 11px; margin: 3px 0;
    font-size: .78rem; color: #c9d1d9;
}
.fact-card {
    background: rgba(34,197,94,.05);
    border: 1px solid rgba(34,197,94,.14);
    border-radius: 7px; padding: 7px 11px; margin: 3px 0;
    font-size: .78rem;
}
.ep-success { border-left: 3px solid #22c55e; padding-left: 7px; }
.ep-fail    { border-left: 3px solid #ef4444; padding-left: 7px; }

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,245,255,.08), rgba(168,85,247,.08));
    border: 1px solid rgba(0,245,255,.25);
    color: #00f5ff;
    border-radius: 7px;
    font-family: 'JetBrains Mono', monospace;
    font-size: .76rem; font-weight: 600;
    letter-spacing: .04em;
    transition: all .2s;
}
.stButton > button:hover {
    border-color: rgba(0,245,255,.6);
    background: rgba(0,245,255,.16);
    box-shadow: 0 0 12px rgba(0,245,255,.2);
    transform: translateY(-1px);
}

/* ── METRICS ── */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,.025);
    border: 1px solid rgba(255,255,255,.055);
    border-radius: 8px; padding: 7px 10px;
}
div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace; color: #00f5ff;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,.02);
    border: 1px solid rgba(255,255,255,.055);
    border-radius: 8px; gap: 2px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: .72rem; letter-spacing: .05em; color: #8b949e;
    border-radius: 6px;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,245,255,.10) !important;
    color: #00f5ff !important;
}

/* ── CODE BLOCK (maze) ── */
.stCode, .stCodeBlock {
    background: #0a0a18 !important;
    border: 1px solid rgba(0,245,255,.12) !important;
    border-radius: 8px !important;
    font-size: .7rem !important;
    line-height: 1.1 !important;
}
pre code { font-size: .7rem !important; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: rgba(8,8,24,.97);
    border-right: 1px solid rgba(0,245,255,.08);
}

/* ── ARCH BOX ── */
.arch-box {
    background: rgba(0,0,0,.25);
    border: 1px solid rgba(0,245,255,.1);
    border-radius: 8px; padding: 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: .78rem; line-height: 2.0;
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# BACKEND GUARD
# ═════════════════════════════════════════════════════════════════════════════
if not _BACKENDS_OK:
    st.error(f"❌ Backend import failed: `{_BACKEND_ERR}`")
    st.info(
        "Ensure `world.py`, `brain.py`, `soul.py`, `memory_palace.py`, `analytics.py` "
        "are in the **same directory** as this file."
    )
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
SAVE_PATH    = "./alive_nexus_v3.json"
STATE_SIZE   = 17      # MazeEnvironment.STATE_SIZE
ACTION_SIZE  = 4

DEFAULT_CONFIG: Dict = {
    # — Simulation
    "sim_speed":          0.04,
    "steps_per_frame":    1,
    "autosave_interval":  100,
    # — Brain
    "gamma":              0.99,
    "epsilon_min":        0.04,
    "epsilon_decay":      0.997,
    "lr":                 0.001,
    "batch_size":         64,
    "buffer_size":        50_000,
    "n_steps":            3,
    "icm_beta":           0.05,
    "tau":                0.005,
    "h1":                 256,
    "h2":                 128,
    "h3":                 64,
    # — Display
    "show_astar":         False,
    "compact_maze":       False,
    "chart_points":       200,
    # — Curriculum override
    "override_curriculum": False,
    "manual_level":        1,
}


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZER
# ═════════════════════════════════════════════════════════════════════════════
def _init_session():
    ss  = st.session_state
    cfg = ss.get("config", dict(DEFAULT_CONFIG))
    ss.config = cfg

    brain_cfg = {k: cfg[k] for k in (
        "gamma","epsilon_min","epsilon_decay","lr","batch_size",
        "buffer_size","n_steps","icm_beta","tau","h1","h2","h3"
    )}

    ss.brain     = AgentBrain(STATE_SIZE, ACTION_SIZE, config=brain_cfg)
    ss.env       = MazeEnvironment()
    ss.soul      = SoulCore(name="Prince")
    ss.memory    = MemoryPalace(save_path=SAVE_PATH)
    ss.analytics = PerformanceDashboard()

    # Seed environment with curriculum level 1 config
    level_cfg        = ss.brain.curriculum.config
    ss.current_state = ss.env.reset(config=level_cfg)
    ss._prev_cells   = 1

    ss.memory.start_episode(
        maze_seed = ss.env.seed,
        maze_alg  = ss.env.algorithm,
        maze_h    = ss.env.maze_h,
        maze_w    = ss.env.maze_w,
        level     = ss.brain.curriculum.level,
        epsilon   = ss.brain.epsilon,
        max_steps = ss.env.max_steps,
    )

    ss.auto_mode       = False
    ss.global_step     = 0
    ss.episode_count   = 0
    ss.capability_score = 0.0
    ss.last_ep_reward  = 0.0
    ss.last_ep_success = False


if "brain" not in st.session_state:
    _init_session()


# ═════════════════════════════════════════════════════════════════════════════
# CORE SIMULATION LOOP
# ═════════════════════════════════════════════════════════════════════════════
def _handle_episode_done(info: Dict):
    ss      = st.session_state
    env_st  = ss.env.get_stats()
    cur_lvl = ss.brain.curriculum.level

    # 1 — Curriculum record
    ss.brain.curriculum.record(
        success   = bool(info.get("reached", False)),
        steps     = env_st["step_count"],
        max_steps = env_st["max_steps"],
        reward    = env_st["episode_reward"],
    )

    # 2 — Memory: close episode
    ss.memory.end_episode(
        total_reward  = env_st["episode_reward"],
        steps         = env_st["step_count"],
        success       = bool(info.get("reached", False)),
        cells_visited = env_st["cells_visited"],
        astar_optimal = env_st["astar_optimal"],
        fog           = env_st["fog"],
        traps         = env_st["traps"] > 0,
        td_error      = ss.brain.avg_td_error,
        epsilon       = ss.brain.epsilon,
    )

    # 3 — Analytics: record episode
    H, W = ss.env.maze.shape
    cap  = ss.analytics.record_episode(
        reward           = env_st["episode_reward"],
        steps            = env_st["step_count"],
        success          = bool(info.get("reached", False)),
        info             = {
            "optimality":   info.get("optimality",   0.0),
            "fog_coverage": info.get("fog_coverage", 1.0),
            "level":        cur_lvl,
        },
        curriculum_level = cur_lvl,
        h=H, w=W,
        maze             = ss.env.maze,
    )

    ss.capability_score = cap
    ss.last_ep_reward   = env_st["episode_reward"]
    ss.last_ep_success  = bool(info.get("reached", False))
    ss.episode_count   += 1

    # 4 — Auto-save
    if ss.episode_count > 0 and ss.episode_count % ss.config.get("autosave_interval", 100) == 0:
        _do_save()

    # 5 — Reset env with updated curriculum config
    new_cfg = ss.brain.curriculum.config
    if ss.config.get("override_curriculum"):
        new_cfg = dict(
            ss.brain.curriculum.LEVEL_CONFIGS.get(
                ss.config.get("manual_level", 1),
                ss.brain.curriculum.config
            )
        )
    ss.current_state = ss.env.reset(config=new_cfg)
    ss._prev_cells   = 1

    # 6 — Memory: open next episode
    ss.memory.start_episode(
        maze_seed = ss.env.seed,
        maze_alg  = ss.env.algorithm,
        maze_h    = ss.env.maze_h,
        maze_w    = ss.env.maze_w,
        level     = ss.brain.curriculum.level,
        epsilon   = ss.brain.epsilon,
        max_steps = ss.env.max_steps,
    )


def process_step():
    """One full environment ↔ brain ↔ soul ↔ memory ↔ analytics tick."""
    ss = st.session_state
    if ss.current_state is None:
        return

    state  = ss.current_state
    action = ss.brain.act(state)

    next_state, reward, done, info = ss.env.step(action)
    loss, td_err                   = ss.brain.step(state, action, reward, next_state, done)

    # ── Memory transition ────────────────────────────────────────
    ss.memory.record_transition(state, action, reward, next_state, done)

    # ── Analytics step ───────────────────────────────────────────
    ss.analytics.record_step(
        agent_r  = ss.env.agent_r,
        agent_c  = ss.env.agent_c,
        loss     = loss,
        td_error = td_err,
        epsilon  = ss.brain.epsilon,
    )

    # ── Soul update ──────────────────────────────────────────────
    trap_nearby = bool(
        ss.env.traps and any(
            abs(t.r - ss.env.agent_r) + abs(t.c - ss.env.agent_c) <= 3
            for t in ss.env.traps
        )
    )
    new_cells = len(ss.env.cells_visited)
    is_new    = new_cells > ss._prev_cells
    ss._prev_cells = new_cells

    ss.soul.update_from_rl(
        stats    = {
            "epsilon":      ss.brain.epsilon,
            "avg_loss":     ss.brain.avg_loss,
            "avg_td_error": ss.brain.avg_td_error,
            "train_step":   ss.brain.train_step,
            "avg_reward":   ss.brain.avg_reward,
            "curriculum":   ss.brain.curriculum.get_stats(),
        },
        env_info = {
            "reward":        reward,
            "reached":       info.get("reached",   False),
            "timeout":       info.get("timeout",   False),
            "trap_hit":      info.get("trap_hit",  False),
            "trap_nearby":   trap_nearby,
            "portal_used":   False,
            "is_new_cell":   is_new,
            "success_count": ss.env.success_count,
            "success_rate":  ss.env.success_count / max(ss.env.total_episodes, 1),
            "cells_visited": new_cells,
            "maze_size":     f"{ss.env.maze_h}×{ss.env.maze_w}",
        },
    )

    ss.current_state = next_state
    ss.global_step  += 1

    if done:
        _handle_episode_done(info)


def reset_all():
    """Hard-reset: destroy all backends and re-initialize."""
    keys = ["brain","env","soul","memory","analytics",
            "current_state","global_step","episode_count",
            "capability_score","last_ep_reward","last_ep_success","_prev_cells"]
    for k in keys:
        st.session_state.pop(k, None)
    st.session_state.auto_mode = False
    _init_session()


# ═════════════════════════════════════════════════════════════════════════════
# PERSISTENCE
# ═════════════════════════════════════════════════════════════════════════════
def _do_save() -> bool:
    ss = st.session_state
    try:
        return ss.memory.save_all(
            brain_weights  = ss.brain.get_weights(),
            analytics_data = ss.analytics.tracker.session_summary(),
            soul_status    = ss.soul.get_status(),
        )
    except Exception as exc:
        st.toast(f"⚠️ Save error: {exc}", icon="⚠️")
        return False


def _export_zip() -> Optional[bytes]:
    ss = st.session_state
    try:
        payload = {
            "version":          "3.0",
            "saved_at":         time.time(),
            "config":           ss.config,
            "brain":            ss.brain.get_weights(),
            "soul":             ss.soul.get_status(),
            "analytics_summary": ss.analytics.tracker.session_summary(),
            "memory_status":    ss.memory.get_full_status(),
            "global_step":      ss.global_step,
            "episode_count":    ss.episode_count,
            "capability_score": ss.capability_score,
        }
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("alive_nexus_v3.json",
                       json.dumps(payload, indent=2, default=_np_default))
        return buf.getvalue()
    except Exception as exc:
        st.toast(f"⚠️ Export error: {exc}", icon="⚠️")
        return None


def _np_default(obj):
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, deque):       return list(obj)
    return str(obj)


def _load_zip(uploaded_file) -> bool:
    ss = st.session_state
    try:
        with zipfile.ZipFile(uploaded_file, "r") as z:
            with z.open("alive_nexus_v3.json") as f:
                data = json.load(f)

        if "brain" in data:
            ss.brain.set_weights(data["brain"])
            ss.brain.target_net.copy_from(ss.brain.online_net)

        if "soul" in data:
            p = data["soul"]
            for trait in ("O","C","E","A","N"):
                if trait in p:
                    setattr(ss.soul.personality, trait, float(p[trait]))
            if "relationship" in p:
                ss.soul.relationship.score = float(p["relationship"])

        ss.global_step      = data.get("global_step",      0)
        ss.episode_count    = data.get("episode_count",    0)
        ss.capability_score = data.get("capability_score", 0.0)
        if "config" in data:
            ss.config.update(data["config"])
        return True
    except Exception as exc:
        st.error(f"❌ Load failed: {exc}")
        return False


# ═════════════════════════════════════════════════════════════════════════════
# HTML / CSS UTILITIES
# ═════════════════════════════════════════════════════════════════════════════
def _kpi(value: str, label: str, sub: str = "") -> str:
    return (f'<div class="kpi-card">'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div>'
            + (f'<div class="kpi-sub">{sub}</div>' if sub else "")
            + "</div>")


def _pb(frac: float, color: str = "#00f5ff") -> str:
    p = max(0.0, min(1.0, frac)) * 100
    return (f'<div class="pb-bg">'
            f'<div class="pb-fill" style="width:{p:.1f}%;background:{color};"></div>'
            f'</div>')


def _conv_css(state: str) -> str:
    return {"warming_up":"conv-warm","rapid_learning":"conv-rapid",
            "fine_tuning":"conv-fine","converged":"conv-ok",
            "plateau":"conv-plateau","regressing":"conv-regress"}.get(state, "conv-warm")


# ═════════════════════════════════════════════════════════════════════════════
# PLOTLY CHART HELPERS
# ═════════════════════════════════════════════════════════════════════════════
_L = dict(  # shared layout base
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(8,8,24,.85)",
    font          = dict(color="#8b949e", family="JetBrains Mono, monospace", size=10),
    margin        = dict(l=38, r=16, t=34, b=32),
    xaxis         = dict(gridcolor="rgba(255,255,255,.04)", zeroline=False),
    yaxis         = dict(gridcolor="rgba(255,255,255,.04)", zeroline=False),
)


def _line_chart(x, traces: list, title: str = "", height: int = 220) -> "go.Figure":
    fig = go.Figure()
    for t in traces:
        fig.add_trace(go.Scatter(
            x=x, y=t["y"], name=t["name"],
            mode="lines",
            line=dict(color=t.get("color","#00f5ff"), width=t.get("w",1.5)),
            fill=t.get("fill"), fillcolor=t.get("fillcolor"),
        ))
    lay = dict(_L)
    lay["title"]  = dict(text=title, font=dict(size=10, color="#8b949e"))
    lay["height"] = height
    fig.update_layout(**lay)
    return fig


def _emotion_plot(v: float, a: float) -> "go.Figure":
    fig = go.Figure()
    zones = [
        ( 0.65,  0.55, .32, "Excited",  "rgba(249,115,22,.09)"),
        ( 0.65, -0.50, .28, "Serene",   "rgba(34,197,94,.09)"),
        (-0.60,  0.60, .30, "Alarmed",  "rgba(239,68,68,.07)"),
        (-0.60, -0.40, .28, "Depressed","rgba(88,166,255,.09)"),
        ( 0.60,  0.10, .24, "Happy",    "rgba(0,245,255,.07)"),
        ( 0.00,  0.00, .18, "Neutral",  "rgba(168,85,247,.06)"),
    ]
    for cx, cy, r, lbl, col in zones:
        fig.add_shape(type="circle",
            x0=cx-r, y0=cy-r, x1=cx+r, y1=cy+r,
            fillcolor=col, line_width=0)
        fig.add_annotation(x=cx, y=cy, text=lbl,
            font=dict(size=8, color="rgba(180,180,180,.45)"), showarrow=False)
    for val in [-1, 0, 1]:
        fig.add_hline(y=val, line_color="rgba(255,255,255,.05)", line_width=.6)
        fig.add_vline(x=val, line_color="rgba(255,255,255,.05)", line_width=.6)
    fig.add_trace(go.Scatter(x=[v], y=[a], mode="markers",
        marker=dict(size=30, color="rgba(168,85,247,.18)",
                    line=dict(color="rgba(168,85,247,.4)", width=1)),
        showlegend=False))
    fig.add_trace(go.Scatter(x=[v], y=[a], mode="markers",
        marker=dict(size=14, color="#a855f7",
                    line=dict(color="#d8b4fe", width=2)),
        showlegend=False))
    lay = dict(_L)
    lay.update(
        title=dict(text="Emotion Circumplex (Russell)", font=dict(size=10, color="#8b949e")),
        xaxis=dict(**_L["xaxis"], title="Valence →", range=[-1.2, 1.2]),
        yaxis=dict(**_L["yaxis"], title="Arousal ↑",  range=[-1.2, 1.2]),
        height=300, width=300,
    )
    fig.update_layout(**lay)
    return fig


def _ocean_radar(O: float, C: float, E: float, A: float, N: float) -> "go.Figure":
    cats = ["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]
    vals = [O, C, E, A, N, O]
    cats_c = cats + [cats[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats_c, fill="toself",
        fillcolor="rgba(168,85,247,.14)",
        line=dict(color="#a855f7", width=2),
        marker=dict(size=5, color="#a855f7"),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(8,8,24,.7)",
            radialaxis=dict(visible=True, range=[0,1],
                gridcolor="rgba(255,255,255,.06)",
                tickfont=dict(size=7, color="#6e7681")),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,.06)",
                tickfont=dict(size=8, color="#c9d1d9")),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=30, r=30, t=34, b=30),
        height=280,
        title=dict(text="Personality (OCEAN)", font=dict(size=10, color="#8b949e")),
    )
    return fig


def _heatmap_plot(data: np.ndarray, title: str = "Heatmap") -> "go.Figure":
    fig = go.Figure(go.Heatmap(
        z=data[::-1],
        colorscale=[
            [0.0, "rgba(8,8,24,1)"],   [0.15, "rgba(59,130,246,.6)"],
            [0.45, "rgba(168,85,247,.85)"], [0.75, "rgba(239,68,68,.9)"],
            [1.0, "rgba(255,230,0,1)"],
        ],
        showscale=False,
        hovertemplate="(%{x},%{y}): %{z:.2f}<extra></extra>",
    ))
    lay = dict(_L)
    lay.update(
        title=dict(text=title, font=dict(size=10, color="#8b949e")),
        height=240, xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    fig.update_layout(**lay)
    return fig


def _curriculum_bar(level: int, scores: list, promote: float, demote: float) -> "go.Figure":
    x = list(range(1, len(scores)+1))
    colors = ["#22c55e" if s >= promote else ("#ef4444" if s <= demote else "#58a6ff")
              for s in scores]
    fig = go.Figure(go.Bar(x=x, y=scores, marker_color=colors))
    fig.add_hline(y=promote, line_color="#22c55e", line_dash="dot",
                  annotation_text="Promote", annotation_font_size=8)
    fig.add_hline(y=demote,  line_color="#ef4444", line_dash="dot",
                  annotation_text="Demote",  annotation_font_size=8)
    lay = dict(_L)
    lay.update(
        title=dict(text=f"Curriculum Window — Level {level}", font=dict(size=10, color="#8b949e")),
        height=200,
    )
    fig.update_layout(**lay)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
def _sidebar():
    ss  = st.session_state
    cfg = ss.config

    with st.sidebar:
        st.markdown(
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:.9rem;'
            'font-weight:700;color:#00f5ff;letter-spacing:.06em;padding:6px 0 10px;">🧬 A.L.I.V.E.</div>',
            unsafe_allow_html=True,
        )

        # ── Run control ─────────────────────────────────────────
        st.markdown('<div class="panel-header">⚡ SIMULATION</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        if r1.button("▶ RUN",   use_container_width=True): ss.auto_mode = True
        if r2.button("⏸ PAUSE", use_container_width=True): ss.auto_mode = False
        r3, r4 = st.columns(2)
        if r3.button("⏭ STEP",  use_container_width=True):
            for _ in range(cfg.get("steps_per_frame",1)):
                process_step()
            st.rerun()
        if r4.button("🔄 RESET", use_container_width=True):
            reset_all(); st.rerun()

        cfg["sim_speed"]       = st.slider("Delay (s)", 0.0, 0.5,
                                            cfg.get("sim_speed", 0.04), 0.01)
        cfg["steps_per_frame"] = st.select_slider("Steps / frame",
                                                    [1,2,4,8,16,32],
                                                    cfg.get("steps_per_frame",1))

        # ── Environment ─────────────────────────────────────────
        st.markdown('<div class="panel-header">🌐 ENVIRONMENT</div>', unsafe_allow_html=True)
        cfg["override_curriculum"] = st.toggle("Override Curriculum",
                                                cfg.get("override_curriculum", False))
        if cfg["override_curriculum"]:
            cfg["manual_level"] = st.slider("Force Level", 1, 10,
                                             cfg.get("manual_level", 1))
        cfg["show_astar"]   = st.toggle("Show A* Overlay",  cfg.get("show_astar",   False))
        cfg["compact_maze"] = st.toggle("Compact Maze View", cfg.get("compact_maze", False))

        # ── Brain hyperparams ────────────────────────────────────
        st.markdown('<div class="panel-header">🧠 BRAIN HYPERPARAMS</div>', unsafe_allow_html=True)
        with st.expander("Tune Hyperparameters", expanded=False):
            cfg["gamma"]         = st.slider("γ Discount",   0.80, 0.999,
                                              cfg.get("gamma",0.99),  format="%.3f")
            cfg["epsilon_decay"] = st.slider("ε Decay",      0.990, 0.9999,
                                              cfg.get("epsilon_decay",0.997), format="%.4f")
            cfg["epsilon_min"]   = st.slider("ε Min",        0.01, 0.15,
                                              cfg.get("epsilon_min",0.04), 0.005)
            cfg["lr"]            = st.slider("Learning Rate",1e-4, 5e-3,
                                              cfg.get("lr",1e-3), format="%.4f")
            cfg["tau"]           = st.slider("τ (soft update)", 0.001, 0.05,
                                              cfg.get("tau",0.005), 0.001)
            cfg["icm_beta"]      = st.slider("ICM β",       0.0, 0.2,
                                              cfg.get("icm_beta",0.05), 0.01)
            cfg["n_steps"]       = st.select_slider("N-Step", [1,2,3,5,8],
                                                     cfg.get("n_steps",3))
            cfg["batch_size"]    = st.select_slider("Batch",  [32,64,128,256],
                                                     cfg.get("batch_size",64))
        with st.expander("Architecture (→ Reset)", expanded=False):
            st.info("Changing these requires a Hard Reset.")
            cfg["h1"] = st.select_slider("H1", [64,128,256,512], cfg.get("h1",256))
            cfg["h2"] = st.select_slider("H2", [32,64,128,256],  cfg.get("h2",128))
            cfg["h3"] = st.select_slider("H3", [16,32,64,128],   cfg.get("h3",64))
            cfg["buffer_size"] = st.select_slider(
                "Buffer", [10_000,25_000,50_000,100_000], cfg.get("buffer_size",50_000))

        # ── Persistence ──────────────────────────────────────────
        st.markdown('<div class="panel-header">💾 PERSISTENCE</div>', unsafe_allow_html=True)
        if st.button("💾 Save Checkpoint", use_container_width=True):
            ok = _do_save()
            st.toast("✅ Saved!" if ok else "❌ Save failed")

        zb = _export_zip()
        if zb:
            st.download_button("⬇️ Export ZIP", data=zb,
                file_name=f"ALIVE_ep{ss.episode_count}.zip",
                mime="application/zip", use_container_width=True)

        up = st.file_uploader("📂 Load ZIP", type="zip",
                               label_visibility="collapsed")
        if up and st.button("Restore Checkpoint", use_container_width=True):
            if _load_zip(up):
                st.toast("✅ Restored!"); st.rerun()

        # ── Soul ────────────────────────────────────────────────
        st.markdown('<div class="panel-header">👤 SOUL CONFIG</div>', unsafe_allow_html=True)
        new_name = st.text_input("Your Name", value=ss.soul.user_name,
                                  label_visibility="collapsed")
        if new_name != ss.soul.user_name:
            ss.soul.user_name = new_name

        cfg["chart_points"]     = st.slider("Chart History",  50, 500,
                                             cfg.get("chart_points",200), 25)
        cfg["autosave_interval"]= st.slider("Autosave (eps)", 25, 500,
                                             cfg.get("autosave_interval",100), 25)

        # ── Status mini-strip ────────────────────────────────────
        br_st  = ss.brain.get_stats()
        soul_s = ss.soul.get_status()
        st.markdown("---")
        st.markdown(
            f'<div style="font-size:.7rem;line-height:1.9;">'
            f'<b style="color:#00f5ff;">ε</b> {br_st["epsilon"]:.4f} &nbsp; '
            f'<b style="color:#22c55e;">Lvl</b> {br_st["curriculum"]["level"]} &nbsp; '
            f'<b style="color:#a855f7;">Mood</b> {soul_s["mood_emoji"]} {soul_s["mood"]}<br>'
            f'<b style="color:#f97316;">Steps</b> {ss.global_step:,} &nbsp; '
            f'<b style="color:#58a6ff;">Eps</b> {ss.episode_count}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — MISSION CONTROL
# ═════════════════════════════════════════════════════════════════════════════
def _tab_mission():
    ss  = st.session_state
    cfg = ss.config

    left, right = st.columns([3, 2], gap="large")

    # ── LEFT: Maze ───────────────────────────────────────────────
    with left:
        rd     = ss.env.get_render_data()
        env_st = ss.env.get_stats()

        st.markdown('<div class="panel-header">🗺️ LIVE ENVIRONMENT</div>',
                    unsafe_allow_html=True)

        # Stat row
        ms1, ms2, ms3, ms4, ms5 = st.columns(5)
        ms1.metric("Episode",   ss.episode_count)
        ms2.metric("Step",      f'{rd["step_count"]}/{rd["max_steps"]}')
        ms3.metric("Level",     ss.brain.curriculum.level)
        ms4.metric("Visited",   f'{env_st["cells_visited"]}/{env_st["total_cells"]}')
        ms5.metric("A* Opt.",   rd["astar_optimal"])

        # Feature badges
        badges = f'<span class="badge badge-c">{rd["algorithm"].upper()}</span>'
        if rd["use_fog"]:             badges += ' <span class="badge badge-p">FOG</span>'
        if len(rd["traps"]) > 0:      badges += f' <span class="badge badge-r">{len(rd["traps"])}×TRAP</span>'
        if len(rd["portals_a"]) > 0:  badges += f' <span class="badge badge-o">{len(rd["portals_a"])}×PORTAL</span>'
        st.markdown(badges, unsafe_allow_html=True)
        st.markdown("")

        # A* path overlay
        astar_path = ss.env.get_astar_path() if cfg.get("show_astar") else []

        grid_str, legend = ss.env.render_ascii()

        if astar_path:
            rows = grid_str.split("\n")
            for (r, c) in astar_path[1:-1]:
                if r < len(rows):
                    row_chars = list(rows[r])
                    idx = c * 2
                    if idx + 1 < len(row_chars):
                        if row_chars[idx] == " " and row_chars[idx+1] == " ":
                            row_chars[idx] = "·"; row_chars[idx+1] = "·"
                    rows[r] = "".join(row_chars)
            grid_str = "\n".join(rows)

        if cfg.get("compact_maze"):
            lines    = grid_str.split("\n")
            grid_str = "\n".join(lines[::2])

        st.code(grid_str, language=None)
        st.caption(f"📌 {legend}")

    # ── RIGHT: Soul + Chat + Controls ────────────────────────────
    with right:
        soul_s = ss.soul.get_status()
        live   = ss.analytics.get_live_stats()

        st.markdown('<div class="panel-header">🧠 CONSCIOUSNESS STREAM</div>',
                    unsafe_allow_html=True)

        # Mood row
        rel_frac = soul_s["relationship"] / 100
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
            f'  <span style="font-size:2rem;">{soul_s["mood_emoji"]}</span>'
            f'  <div>'
            f'    <div style="font-size:.95rem;font-weight:700;color:#d8b4fe;">{soul_s["mood"]}</div>'
            f'    <div style="font-size:.65rem;color:#6e7681;">'
            f'      V={soul_s["valence"]:+.2f} A={soul_s["arousal"]:+.2f} ⚡{soul_s["intensity"]:.2f}'
            f'    </div>'
            f'  </div>'
            f'  <div style="margin-left:auto;text-align:right;">'
            f'    <div style="font-size:.65rem;color:#6e7681;">{soul_s["stage"]}</div>'
            f'    <div style="font-size:.82rem;font-weight:700;color:#58a6ff;">❤ {soul_s["relationship"]}</div>'
            f'  </div>'
            f'</div>'
            f'{_pb(rel_frac, "#a855f7")}'
            f'<div style="font-size:.62rem;color:#6e7681;margin-top:3px;">{soul_s["stage_desc"]}</div>',
            unsafe_allow_html=True,
        )

        # Inner thought
        st.markdown(
            f'<div class="thought-box">💭 {soul_s["thought"]}</div>',
            unsafe_allow_html=True,
        )

        # Convergence banner
        conv_state = live.get("convergence", "warming_up")
        conv_icon  = live.get("convergence_icon", "🔥")
        conv_desc  = live.get("convergence_desc", "")
        st.markdown(
            f'<div class="conv-base {_conv_css(conv_state)}">'
            f'  <span style="font-size:1.2rem;">{conv_icon}</span>'
            f'  <div>'
            f'    <div style="font-weight:700;font-size:.82rem;color:#c9d1d9;">'
            f'      {conv_state.replace("_"," ").title()}</div>'
            f'    <div style="font-size:.68rem;color:#6e7681;">{conv_desc}</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Capability bar
        cap = ss.capability_score or 0.0
        cap_trend = live.get("capability_trend", "→")
        st.markdown(
            f'<div style="margin:6px 0;">'
            f'  <div style="display:flex;justify-content:space-between;'
            f'              font-size:.68rem;color:#6e7681;margin-bottom:3px;">'
            f'    <span>Capability Score {cap_trend}</span>'
            f'    <span style="color:#00f5ff;font-weight:700;">{cap:.1f} / 100</span>'
            f'  </div>'
            f'  {_pb(cap/100, "#00f5ff")}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Chat
        st.markdown("")
        st.markdown('<div class="panel-header">💬 COMM LINK</div>', unsafe_allow_html=True)
        chat_html = '<div class="chat-scroll">'
        for msg in ss.soul.get_chat_history()[-16:]:
            if msg["role"] == "user":
                chat_html += (f'<div class="chat-user"><b>{ss.soul.user_name}</b>: {msg["text"]}'
                              f'<div class="chat-meta">intent: {msg.get("intent","?")}</div></div>')
            else:
                chat_html += (f'<div class="chat-ai"><b>A.L.I.V.E</b>: {msg["text"]}'
                              f'<div class="chat-meta">feeling: {msg.get("emotion","?")}</div></div>')
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)

        ui = st.chat_input("Talk to A.L.I.V.E...", key="mc_chat")
        if ui:
            ss.soul.chat(ui); st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS LAB
# ═════════════════════════════════════════════════════════════════════════════
def _tab_analytics():
    ss  = st.session_state
    cfg = ss.config
    live = ss.analytics.get_live_stats()
    cd   = ss.analytics.get_chart_data(cfg.get("chart_points",200))

    # ── Top KPIs ─────────────────────────────────────────────────
    kc = st.columns(7)
    kpis = [
        (f'{live["total_episodes"]}',         "Episodes"),
        (f'{live["success_rate"]:.1f}%',       "Success"),
        (f'{live["avg_reward"]:+.2f}',         "Avg Reward"),
        (f'{live["avg_steps"]:.0f}',           "Avg Steps"),
        (f'{live["capability"]:.1f}',          "Capability"),
        (f'{live["reward_trend"]:+.4f}',       "Trend"),
        (f'{ss.brain.get_stats()["epsilon"]:.4f}', "Epsilon"),
    ]
    for col, (val, lbl) in zip(kc, kpis):
        col.markdown(_kpi(val, lbl), unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 1: Reward + Capability ──────────────────────────────
    if _PLOTLY and cd.get("rewards"):
        r1, r2 = st.columns(2)
        n = len(cd["rewards"])
        with r1:
            fig = _line_chart(
                x=list(range(n)),
                traces=[
                    {"y": cd["rewards"],     "name": "Reward",  "color": "rgba(0,245,255,.35)", "w": 1.0},
                    {"y": cd["ema_rewards"], "name": "EMA(10)", "color": "#00f5ff", "w": 2.2},
                ],
                title="Episode Reward + EMA-10",
                height=230,
            )
            st.plotly_chart(fig, use_container_width=True, key="cht_reward")

        cap_hist = list(ss.analytics.capability.history)
        opt      = cd.get("optimality", [])
        with r2:
            n2 = max(len(cap_hist), len(opt))
            fig2 = _line_chart(
                x=list(range(n2)),
                traces=[
                    {"y": cap_hist, "name": "Capability",    "color": "#a855f7", "w": 2.0},
                    {"y": opt,      "name": "Path Efficiency","color": "#22c55e", "w": 1.4},
                ],
                title="Capability Score & Path Efficiency",
                height=230,
            )
            st.plotly_chart(fig2, use_container_width=True, key="cht_cap")

    # ── Row 2: Loss / TD / Epsilon ──────────────────────────────
    if _PLOTLY and cd.get("losses"):
        losses   = cd["losses"]
        td_errs  = cd["td_errors"]
        epsilons = cd.get("epsilons", [])
        r3, r4, r5 = st.columns(3)

        with r3:
            fig3 = _line_chart(list(range(len(losses))),
                [{"y": losses,  "name": "Loss",     "color": "#f97316", "w": 1.1}],
                "Training Loss", 200)
            st.plotly_chart(fig3, use_container_width=True, key="cht_loss")

        with r4:
            fig4 = _line_chart(list(range(len(td_errs))),
                [{"y": td_errs, "name": "TD Error", "color": "#ef4444", "w": 1.1}],
                "TD Error", 200)
            st.plotly_chart(fig4, use_container_width=True, key="cht_td")

        with r5:
            fig5 = _line_chart(list(range(len(epsilons))),
                [{"y": epsilons,"name": "ε",        "color": "#58a6ff", "w": 1.5}],
                "Exploration Rate ε", 200)
            st.plotly_chart(fig5, use_container_width=True, key="cht_eps")

    # ── Row 3: Heatmap + Success Rate ───────────────────────────
    r6, r7 = st.columns([3, 2])

    with r6:
        st.markdown('<div class="panel-header">🗺️ EXPLORATION HEATMAP</div>',
                    unsafe_allow_html=True)
        H, W    = ss.env.maze.shape
        hm      = ss.analytics.get_heatmap(H, W)
        cov     = float((hm > 0).sum()) / max(int((ss.env.maze == 0).sum()), 1)
        if _PLOTLY:
            figh = _heatmap_plot(hm, f"Visit Frequency — {H}×{W}  ({cov*100:.1f}% explored)")
            st.plotly_chart(figh, use_container_width=True, key="cht_hm")

    with r7:
        st.markdown('<div class="panel-header">📈 ROLLING SUCCESS</div>',
                    unsafe_allow_html=True)
        succ = cd.get("successes", [])
        if succ and _PLOTLY:
            win = 20
            rs  = [sum(succ[max(0,i-win):i+1]) / min(i+1,win) for i in range(len(succ))]
            fig_s = _line_chart(list(range(len(rs))),
                [{"y": rs, "name": "Success",
                  "color": "#22c55e", "w": 2.0,
                  "fill": "tozeroy", "fillcolor": "rgba(34,197,94,.08)"}],
                f"Rolling Success Rate (w={win})", 240)
            fig_s.update_layout(yaxis=dict(range=[0,1]))
            st.plotly_chart(fig_s, use_container_width=True, key="cht_succ")

        # Training snapshot
        br_st = ss.brain.get_stats()
        st.markdown('<div class="panel-header">🔧 TRAINING SNAPSHOT</div>',
                    unsafe_allow_html=True)
        sa, sb = st.columns(2)
        sa.metric("Train Steps",  f'{br_st["train_step"]:,}')
        sb.metric("Avg Loss",     f'{br_st["avg_loss"]:.5f}')
        sa.metric("Avg TD-Err",   f'{br_st["avg_td_error"]:.4f}')
        sb.metric("LR",           f'{br_st["lr"]:.6f}')
        sa.metric("Buffer",       f'{br_st["memory_size"]:,}')
        sb.metric("ICM States",   br_st["unique_states"])
        sa.metric("LR Reductions",ss.brain.lr_sched.reductions)
        sb.metric("Total Eps",    br_st["total_episodes"])

    # ── Session report ──────────────────────────────────────────
    with st.expander("📋 Full Session Report", expanded=False):
        st.code(ss.analytics.get_session_report(), language=None)
    with st.expander("📤 Export JSON Preview", expanded=False):
        st.code(ss.analytics.export_json()[:3000] + "\n...[truncated]", language="json")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — SOUL MATRIX
# ═════════════════════════════════════════════════════════════════════════════
def _tab_soul():
    ss     = st.session_state
    soul_s = ss.soul.get_status()
    rl_ctx = ss.soul._rl_ctx

    sc1, sc2, sc3 = st.columns([1, 1, 2])

    # ── Emotion circumplex ───────────────────────────────────────
    with sc1:
        if _PLOTLY:
            st.plotly_chart(
                _emotion_plot(soul_s["valence"], soul_s["arousal"]),
                use_container_width=True, key="soul_em",
            )
        st.markdown(
            f'<div style="text-align:center;">'
            f'  <div style="font-size:2.4rem;">{soul_s["mood_emoji"]}</div>'
            f'  <div style="font-size:.95rem;font-weight:700;color:#d8b4fe;">{soul_s["mood"]}</div>'
            f'  <div style="font-size:.68rem;color:#6e7681;">Intensity: {soul_s["intensity"]:.3f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── OCEAN radar ─────────────────────────────────────────────
    with sc2:
        if _PLOTLY:
            st.plotly_chart(
                _ocean_radar(soul_s["O"], soul_s["C"],
                             soul_s["E"], soul_s["A"], soul_s["N"]),
                use_container_width=True, key="soul_oc",
            )
        st.markdown(
            f'<div style="text-align:center;font-size:.76rem;color:#6e7681;">'
            f'Personality: <b style="color:#d8b4fe;">{soul_s["personality"]}</b></div>',
            unsafe_allow_html=True,
        )
        # OCEAN trait bars
        trait_colors = {"O":"#00f5ff","C":"#22c55e","E":"#f97316","A":"#a855f7","N":"#ef4444"}
        trait_labels = {"O":"Openness","C":"Conscientiousness","E":"Extraversion",
                        "A":"Agreeableness","N":"Neuroticism"}
        for t, lbl in trait_labels.items():
            v = soul_s[t]
            c = trait_colors[t]
            st.markdown(
                f'<div style="margin-top:4px;">'
                f'  <div style="display:flex;justify-content:space-between;'
                f'              font-size:.65rem;color:#6e7681;">'
                f'    <span>{lbl}</span><span style="color:{c};">{v:.2f}</span>'
                f'  </div>'
                f'  {_pb(v, c)}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Relationship + Memories ──────────────────────────────────
    with sc3:
        st.markdown('<div class="panel-header">❤ RELATIONSHIP</div>', unsafe_allow_html=True)
        rel = soul_s["relationship"] / 100
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:4px;">'
            f'  <b style="color:#d8b4fe;">{soul_s["stage"]}</b>'
            f'  <b style="color:#a855f7;">{soul_s["relationship"]} / 100</b>'
            f'</div>'
            f'{_pb(rel, "#a855f7")}'
            f'<div style="font-size:.68rem;color:#6e7681;margin-top:3px;">{soul_s["stage_desc"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown(
            f'<div style="font-size:.72rem;color:#6e7681;">'
            f'  💬 {soul_s["turns"]} turns | 🧠 {soul_s["memories_stored"]} memories'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown('<div class="panel-header">💭 STRONGEST MEMORIES</div>',
                    unsafe_allow_html=True)
        for i, mem in enumerate(soul_s.get("strongest_memories",[])[:5], 1):
            st.markdown(f'<div class="mem-trace">#{i} {mem}</div>', unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="panel-header">📊 RL CONTEXT</div>', unsafe_allow_html=True)
        kv_pairs = [
            ("ε",       f'{rl_ctx.get("epsilon",1.0):.4f}',   "#00f5ff"),
            ("Loss",    f'{rl_ctx.get("avg_loss",0):.5f}',    "#f97316"),
            ("TD Err",  f'{rl_ctx.get("td_error",0):.4f}',    "#ef4444"),
            ("Step",    f'{rl_ctx.get("train_step",0):,}',    "#58a6ff"),
            ("Wins",    f'{rl_ctx.get("wins",0)}',             "#22c55e"),
            ("Avg R",   f'{rl_ctx.get("avg_reward",0):+.3f}', "#00f5ff"),
            ("Level",   f'{rl_ctx.get("curriculum_level",1)}', "#f97316"),
            ("Maze",    f'{rl_ctx.get("maze_size","?")}',      "#c9d1d9"),
        ]
        cols_kv = st.columns(4)
        for i, (k, v, c) in enumerate(kv_pairs):
            cols_kv[i % 4].markdown(
                f'<div style="text-align:center;padding:4px;">'
                f'  <div style="font-size:.6rem;color:#6e7681;text-transform:uppercase;">{k}</div>'
                f'  <div style="font-size:.88rem;font-family:\'JetBrains Mono\',monospace;'
                f'             font-weight:700;color:{c};">{v}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Full chat ────────────────────────────────────────────────
    st.markdown('<div class="panel-header">💬 FULL DIALOGUE INTERFACE</div>',
                unsafe_allow_html=True)
    ch_left, ch_right = st.columns([3, 1])

    with ch_left:
        chat_html = '<div class="chat-scroll">'
        for msg in ss.soul.get_chat_history()[-50:]:
            if msg["role"] == "user":
                chat_html += (f'<div class="chat-user"><b>{ss.soul.user_name}</b>: {msg["text"]}'
                              f'<div class="chat-meta">intent: {msg.get("intent","?")}</div></div>')
            else:
                chat_html += (f'<div class="chat-ai"><b>A.L.I.V.E</b>: {msg["text"]}'
                              f'<div class="chat-meta">state: {msg.get("emotion","?")} '
                              f'V={soul_s["valence"]:+.2f} A={soul_s["arousal"]:+.2f}</div></div>')
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)
        ui = st.chat_input("Communicate...", key="soul_chat_full")
        if ui:
            ss.soul.chat(ui); st.rerun()

    with ch_right:
        st.markdown('<div class="panel-header">🎭 RESPONSE STYLE</div>',
                    unsafe_allow_html=True)
        style = ss.soul.personality.response_style()
        style_items = [
            ("Verbose",    style.get("verbose",    False), "#58a6ff"),
            ("Poetic",     style.get("poetic",     False), "#a855f7"),
            ("Analytical", style.get("analytical", False), "#00f5ff"),
            ("Warm",       style.get("warm",       False), "#f97316"),
            ("Dramatic",   style.get("dramatic",   False), "#ef4444"),
        ]
        for name, active, color in style_items:
            ico = "●" if active else "○"
            col = color if active else "#3a3a4a"
            st.markdown(
                f'<div style="font-size:.76rem;color:{col};margin:3px 0;">'
                f'  {ico} {name}</div>',
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — MEMORY PALACE
# ═════════════════════════════════════════════════════════════════════════════
def _tab_memory():
    ss     = st.session_state
    mem_st = ss.memory.get_full_status()
    ep_st  = mem_st.get("episodic_stats", {})

    mc1, mc2 = st.columns([3, 2])

    with mc1:
        st.markdown('<div class="panel-header">📜 EPISODIC MEMORY</div>',
                    unsafe_allow_html=True)

        if ep_st:
            mm1, mm2, mm3, mm4 = st.columns(4)
            mm1.metric("Stored",        ep_st.get("total_stored",  0))
            mm2.metric("Success Rate",  f'{ep_st.get("success_rate",0)*100:.1f}%')
            mm3.metric("Best Level",    ep_st.get("max_level_reached", 1))
            mm4.metric("Avg Efficiency",f'{ep_st.get("avg_efficiency",0)*100:.1f}%')

        recent = mem_st.get("episodic_recent", [])
        if recent:
            rows = []
            for ep in recent:
                rows.append({
                    "#":       ep.get("episode_id", "?"),
                    "Lvl":     ep.get("curriculum_level", 1),
                    "Alg":     ep.get("maze_alg", "?")[:4].upper(),
                    "Reward":  f'{ep.get("total_reward",0):.2f}',
                    "Steps":   ep.get("total_steps", 0),
                    "✓":       "✅" if ep.get("success") else "❌",
                    "Eff%":    f'{ep.get("efficiency",0)*100:.0f}%',
                    "Fog":     "🌫" if ep.get("fog_used") else "—",
                    "Tags":    (ep.get("tags",[])[:1]+[""])[0][:10],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True, height=268)
        else:
            st.info("No episodes yet. Start the simulation!")

        # Landmarks
        st.markdown('<div class="panel-header">🌟 LANDMARK EPISODES</div>',
                    unsafe_allow_html=True)
        for lm in mem_st.get("landmark_episodes",[])[:5]:
            cls  = "ep-success" if lm.get("success") else "ep-fail"
            icon = "✅" if lm.get("success") else "❌"
            st.markdown(
                f'<div class="mem-trace {cls}">'
                f' {icon} L{lm.get("curriculum_level","?")} | '
                f'{lm.get("maze_alg","?")} | '
                f'R={lm.get("total_reward",0):.2f} | '
                f'Eff={lm.get("efficiency",0)*100:.0f}%'
                f'</div>',
                unsafe_allow_html=True,
            )

    with mc2:
        # Semantic knowledge
        st.markdown('<div class="panel-header">📖 SEMANTIC KNOWLEDGE</div>',
                    unsafe_allow_html=True)
        sem_sum = mem_st.get("semantic_summary", {})
        if sem_sum:
            total = sem_sum.get("total_facts", 0)
            hi    = sem_sum.get("high_confidence_count", 0)
            st.markdown(
                f'<div style="font-size:.75rem;color:#6e7681;margin-bottom:6px;">'
                f'  {total} facts | '
                f'  <span style="color:#22c55e;">{hi} high-confidence</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        for fact in mem_st.get("semantic_facts",[])[:10]:
            conf = float(fact.get("confidence", 0))
            cc   = "#22c55e" if conf > 0.7 else ("#f97316" if conf > 0.4 else "#ef4444")
            st.markdown(
                f'<div class="fact-card">'
                f'  <b style="color:#00f5ff;">{fact.get("key","?")}</b>: '
                f'{str(fact.get("value","?"))[:55]}'
                f'  <div style="font-size:.62rem;color:{cc};">'
                f'    conf={conf:.2f} | src={fact.get("source","?")}'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Insights
        st.markdown("")
        st.markdown('<div class="panel-header">💡 AGENT INSIGHTS</div>',
                    unsafe_allow_html=True)
        for ins in mem_st.get("insights",[]):
            st.markdown(f'<div class="mem-trace">{ins}</div>', unsafe_allow_html=True)

        # Persistence info
        st.markdown("")
        st.markdown('<div class="panel-header">💾 STORE STATUS</div>',
                    unsafe_allow_html=True)
        save_path = mem_st.get("save_path", SAVE_PATH)
        size_kb   = mem_st.get("save_size_kb", 0.0)
        loaded    = mem_st.get("loaded_from_disk", False)
        total_ep  = mem_st.get("total_episodes", 0)
        st.markdown(
            f'<div style="font-size:.75rem;line-height:1.9;">'
            f'  <div><span style="color:#6e7681;">Path: </span>'
            f'       <code style="font-size:.68rem;">{save_path}</code></div>'
            f'  <div><span style="color:#6e7681;">Size: </span>'
            f'       <b style="color:#00f5ff;">{size_kb:.1f} KB</b></div>'
            f'  <div><span style="color:#6e7681;">From disk: </span>'
            f'       <b style="color:{"#22c55e" if loaded else "#ef4444"};">'
            f'       {"Yes ✅" if loaded else "No"}</b></div>'
            f'  <div><span style="color:#6e7681;">Episodes: </span>'
            f'       <b style="color:#a855f7;">{total_ep}</b></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("💾 Save Now", use_container_width=True, key="mem_save"):
            ok = _do_save()
            st.toast("✅ Saved!" if ok else "❌ Failed", icon="💾")
        if ss.memory.store.exists():
            if st.button("📂 Load from Disk", use_container_width=True, key="mem_load"):
                data = ss.memory.load_all()
                if data and "brain_weights" in data:
                    ss.brain.set_weights(data["brain_weights"])
                    ss.brain.target_net.copy_from(ss.brain.online_net)
                    st.toast("✅ Loaded from disk!", icon="📂")
                    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — BRAIN AUTOPSY
# ═════════════════════════════════════════════════════════════════════════════
def _tab_brain():
    ss    = st.session_state
    br_st = ss.brain.get_stats()
    cur   = br_st.get("curriculum", {})
    cfg   = ss.config

    # ── Curriculum ───────────────────────────────────────────────
    st.markdown('<div class="panel-header">🎓 CURRICULUM LEARNING</div>',
                unsafe_allow_html=True)
    cc = st.columns(6)
    cc[0].metric("Level",       cur.get("level",1))
    cc[1].metric("Max Level",   cur.get("max_level",10))
    cc[2].metric("Avg Score",   f'{cur.get("avg_score",0):.3f}')
    cc[3].metric("Promotions",  cur.get("promotions",0))
    cc[4].metric("Demotions",   cur.get("demotions",0))
    cc[5].metric("ZPD Progress",f'{cur.get("zpd_progress",0)*100:.0f}%')

    zpd = cur.get("zpd_progress", 0.0)
    st.markdown(
        f'<div style="font-size:.68rem;color:#6e7681;margin:6px 0 3px;">Progress to promotion:</div>'
        f'{_pb(zpd, "#22c55e")}',
        unsafe_allow_html=True,
    )

    # Level grid
    level_cfgs = ss.brain.curriculum.LEVEL_CONFIGS
    st.markdown("")
    lc = st.columns(5)
    for i, (lvl, lcfg) in enumerate(level_cfgs.items()):
        col       = lc[i % 5]
        is_cur    = (lvl == cur.get("level",1))
        bdr       = "border:2px solid #00f5ff;" if is_cur else "border:1px solid rgba(255,255,255,.07);"
        bg        = "background:rgba(0,245,255,.07);" if is_cur else "background:rgba(255,255,255,.02);"
        feat_str  = " ".join(
            [f for f, cond in [("🌫",lcfg["fog"]),("💀",lcfg["dynamic"]),("🌀",lcfg["portals"])]
             if cond]
        ) or "—"
        tc        = "#00f5ff" if is_cur else "#6e7681"
        col.markdown(
            f'<div style="{bdr}{bg}border-radius:8px;padding:7px;margin-bottom:6px;text-align:center;">'
            f'  <div style="font-size:.95rem;font-weight:700;color:{tc};">L{lvl}</div>'
            f'  <div style="font-size:.58rem;color:#6e7681;">{lcfg["maze_h"]}×{lcfg["maze_w"]}</div>'
            f'  <div style="font-size:.58rem;color:#6e7681;">{lcfg["algorithm"][:4].upper()}</div>'
            f'  <div style="font-size:.68rem;">{feat_str}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Curriculum window chart
    if _PLOTLY:
        window_scores = list(ss.brain.curriculum.window)
        if window_scores:
            fig_cur = _curriculum_bar(
                cur.get("level",1), window_scores,
                ss.brain.curriculum.promote_thresh,
                ss.brain.curriculum.demote_thresh,
            )
            st.plotly_chart(fig_cur, use_container_width=True, key="cht_cur")

    st.markdown("---")

    ba1, ba2 = st.columns(2)

    # ── Architecture ─────────────────────────────────────────────
    with ba1:
        st.markdown('<div class="panel-header">🏗️ NETWORK ARCHITECTURE</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="arch-box">'
            f'<span style="color:#00f5ff;">Input</span>     ─── {STATE_SIZE} neurons (vision+pos+dist+fog)<br>'
            f'<span style="color:#6e7681;">│</span><br>'
            f'<span style="color:#58a6ff;">H1</span>        ─── {cfg.get("h1",256)} neurons  [Leaky ReLU + Adam]<br>'
            f'<span style="color:#6e7681;">│</span><br>'
            f'<span style="color:#58a6ff;">H2</span>        ─── {cfg.get("h2",128)} neurons  [Leaky ReLU + Adam]<br>'
            f'<span style="color:#6e7681;">│</span><br>'
            f'<span style="color:#58a6ff;">H3</span>        ─── {cfg.get("h3",64)} neurons  [Leaky ReLU + Adam]<br>'
            f'<span style="color:#6e7681;">├──┬──</span><br>'
            f'<span style="color:#a855f7;">V(s)</span>      ──── 1 output   [State Value]<br>'
            f'<span style="color:#f97316;">A(s,a)</span>    ──── {ACTION_SIZE} outputs  [Advantage]<br>'
            f'<span style="color:#6e7681;">↓</span><br>'
            f'<span style="color:#22c55e;">Q(s,a)</span>   = V + A − mean(A)<br>'
            f'<span style="color:#6e7681;font-size:.65rem;">'
            f'  τ={cfg.get("tau",0.005)} Polyak | Online + Target nets</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("")
        st.markdown('<div class="panel-header">📐 ALGORITHM STACK</div>',
                    unsafe_allow_html=True)
        stack = [
            ("Dueling DQN",        "#00f5ff", "V(s)+A(s,a) decomposition"),
            ("Double DQN",         "#58a6ff", "Online selects, Target evaluates"),
            (f"N-Step Returns",    "#a855f7", f"N={cfg.get('n_steps',3)} step TD targets"),
            ("PER (Seg Tree)",     "#f97316", "O(log N) priority sampling"),
            ("Intrinsic Curiosity","#22c55e", f"β/√N(s), β={cfg.get('icm_beta',0.05)}"),
            ("Adam Optimizer",     "#ef4444", "β₁=0.9, β₂=0.999, gradient clip"),
            (f"Soft Update",       "#fbbf24", f"τ={cfg.get('tau',0.005)} Polyak"),
            ("Curriculum ACL",     "#ec4899", "10 levels, ZPD-based"),
            ("LR Scheduler",       "#8b5cf6", "Plateau reduce, factor=0.5"),
        ]
        for name, color, desc in stack:
            st.markdown(
                f'<div style="display:flex;align-items:flex-start;gap:8px;margin:3px 0;">'
                f'  <code style="color:{color};font-size:.68rem;white-space:nowrap;'
                f'             min-width:130px;">{name}</code>'
                f'  <span style="font-size:.7rem;color:#6e7681;">{desc}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Weights + ICM ────────────────────────────────────────────
    with ba2:
        st.markdown('<div class="panel-header">⚖️ W1 WEIGHT DISTRIBUTION</div>',
                    unsafe_allow_html=True)
        # W1 weight distribution histogram
        w1_flat = ss.brain.online_net.W1.flatten()
        if _PLOTLY:
            fig_w = go.Figure(go.Histogram(
                x=w1_flat,
                nbinsx=60,
                marker_color="rgba(0,245,255,.55)",
                marker_line=dict(color="rgba(0,245,255,.15)", width=.5),
            ))
            fig_w.update_layout(
                **{**_L,
                   "title": dict(text="W1 Weight Distribution", font=dict(size=10, color="#8b949e")),
                   "height": 190,
                   "bargap": 0.05,
                   "yaxis": dict(**_L["yaxis"], title=""),
                   "xaxis": dict(**_L["xaxis"], title="Weight Value"),
                   },
            )
            st.plotly_chart(fig_w, use_container_width=True, key="cht_w1")

        # W1 stat line
        st.markdown(
            f'<div style="font-size:.68rem;color:#6e7681;display:flex;gap:16px;margin-top:2px;">'
            f'  <span>μ=<b style="color:#00f5ff;">{w1_flat.mean():.4f}</b></span>'
            f'  <span>σ=<b style="color:#a855f7;">{w1_flat.std():.4f}</b></span>'
            f'  <span>|max|=<b style="color:#f97316;">{np.abs(w1_flat).max():.4f}</b></span>'
            f'  <span>sparsity=<b style="color:#22c55e;">'
            f'{(np.abs(w1_flat)<0.01).mean()*100:.1f}%</b></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("")
        st.markdown('<div class="panel-header">🔬 ICM STATE COVERAGE</div>',
                    unsafe_allow_html=True)

        H, W = ss.env.maze.shape
        icm_hm = ss.brain.curiosity.heatmap(H, W)
        if _PLOTLY:
            fig_icm = _heatmap_plot(icm_hm, f"ICM Visit Density — {len(ss.brain.curiosity.counts):,} unique states")
            st.plotly_chart(fig_icm, use_container_width=True, key="cht_icm_hm")

        # ICM stats
        total_visits = sum(ss.brain.curiosity.counts.values()) if ss.brain.curiosity.counts else 0
        top_visited  = max(ss.brain.curiosity.counts.values()) if ss.brain.curiosity.counts else 0
        st.markdown(
            f'<div style="font-size:.68rem;color:#6e7681;line-height:2.0;">'
            f'  Unique states: <b style="color:#00f5ff;">{ss.brain.curiosity.coverage():,}</b> &nbsp;|&nbsp;'
            f'  Total visits: <b style="color:#a855f7;">{total_visits:,}</b> &nbsp;|&nbsp;'
            f'  Max visits/state: <b style="color:#f97316;">{top_visited:,}</b><br>'
            f'  β (curiosity): <b style="color:#22c55e;">{ss.brain.curiosity.beta}</b>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("")
        st.markdown('<div class="panel-header">📉 RECENT LOSS TRACE</div>',
                    unsafe_allow_html=True)
        br_st2   = ss.brain.get_stats()
        rec_loss = list(ss.brain.recent_losses)
        rec_td   = list(ss.brain.recent_td_errors)
        if rec_loss and _PLOTLY:
            fig_lt = _line_chart(
                x=list(range(len(rec_loss))),
                traces=[
                    {"y": rec_loss, "name": "Loss",     "color": "#f97316", "w": 1.3},
                    {"y": rec_td,   "name": "TD Error",  "color": "#ef4444", "w": 1.0},
                ],
                title=f"Last {len(rec_loss)} Steps — Loss & TD Error",
                height=200,
            )
            st.plotly_chart(fig_lt, use_container_width=True, key="cht_brain_loss")

        # Quick stats summary
        st.markdown(
            f'<div style="font-size:.72rem;line-height:2.0;">'
            f'  <div>'
            f'    <span style="color:#6e7681;">Train Steps:</span> '
            f'    <b style="color:#00f5ff;">{br_st2["train_step"]:,}</b>'
            f'  </div>'
            f'  <div>'
            f'    <span style="color:#6e7681;">Avg Loss:</span> '
            f'    <b style="color:#f97316;">{br_st2["avg_loss"]:.6f}</b>'
            f'  </div>'
            f'  <div>'
            f'    <span style="color:#6e7681;">Avg TD-Error:</span> '
            f'    <b style="color:#ef4444;">{br_st2["avg_td_error"]:.5f}</b>'
            f'  </div>'
            f'  <div>'
            f'    <span style="color:#6e7681;">LR (current):</span> '
            f'    <b style="color:#22c55e;">{br_st2["lr"]:.7f}</b>'
            f'  </div>'
            f'  <div>'
            f'    <span style="color:#6e7681;">LR Reductions:</span> '
            f'    <b style="color:#a855f7;">{ss.brain.lr_sched.reductions}</b>'
            f'  </div>'
            f'  <div>'
            f'    <span style="color:#6e7681;">Buffer Fill:</span> '
            f'    <b style="color:#58a6ff;">{br_st2["memory_size"]:,} / '
            f'{ss.brain.memory.capacity:,}</b>'
            f'  </div>'
            f'  <div>'
            f'    <span style="color:#6e7681;">ε (epsilon):</span> '
            f'    <b style="color:#fbbf24;">{br_st2["epsilon"]:.5f}</b>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── PER Priority distribution ───────────────────────────────────
    st.markdown('<div class="panel-header">📊 PER PRIORITY DISTRIBUTION</div>',
                unsafe_allow_html=True)

    per_buf = ss.brain.memory
    if per_buf.size > 0 and _PLOTLY:
        sample_n  = min(per_buf.size, 2000)
        prios     = [per_buf.sum_tree[i] for i in
                     np.random.choice(per_buf.size, sample_n, replace=False).tolist()]
        prios     = [p for p in prios if p > 0]
        if prios:
            fig_per = go.Figure(go.Histogram(
                x=prios, nbinsx=50,
                marker_color="rgba(249,115,22,.55)",
                marker_line=dict(color="rgba(249,115,22,.2)", width=.5),
            ))
            fig_per.update_layout(**{
                **_L,
                "title": dict(text=f"PER Priority Dist. (n={len(prios)}) — β={per_buf.beta:.3f}",
                              font=dict(size=10, color="#8b949e")),
                "height": 185,
                "xaxis": dict(**_L["xaxis"], title="Priority"),
                "yaxis": dict(**_L["yaxis"], title="Count"),
            })
            st.plotly_chart(fig_per, use_container_width=True, key="cht_per")
    else:
        st.caption("Buffer too small to sample priorities yet.")

    # ── N-step buffer peek ──────────────────────────────────────────
    st.markdown("")
    st.markdown('<div class="panel-header">🔗 N-STEP RETURN BUFFER</div>',
                unsafe_allow_html=True)
    ns_buf = list(ss.brain.n_step.buf)
    if ns_buf:
        ns_rows = []
        for i, (_, act, rew, _, done) in enumerate(ns_buf):
            ns_rows.append({
                "Slot": i,
                "Action": ["↑","↓","←","→"][int(act)],
                "Reward": f"{rew:+.3f}",
                "Done":   "✅" if done else "·",
            })
        st.dataframe(
            pd.DataFrame(ns_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("N-step buffer is empty (agent hasn't started stepping yet).")

    # ── Curriculum history table ────────────────────────────────────
    st.markdown("")
    st.markdown('<div class="panel-header">📜 CURRICULUM HISTORY (last 50)</div>',
                unsafe_allow_html=True)
    cur_hist = ss.brain.curriculum.history[-50:]
    if cur_hist:
        df_cur = pd.DataFrame(cur_hist)
        df_cur.index = range(max(0, len(ss.brain.curriculum.history) - len(cur_hist)),
                             len(ss.brain.curriculum.history))
        st.dataframe(
            df_cur.style.applymap(
                lambda v: "color:#22c55e;" if v is True else
                          ("color:#ef4444;" if v is False else ""),
                subset=["success"] if "success" in df_cur.columns else []
            ),
            use_container_width=True,
        )
    else:
        st.caption("No curriculum history yet.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — EPISODE TIMELINE
# ═════════════════════════════════════════════════════════════════════════════
def _tab_timeline():
    ss  = st.session_state
    eps = ss.analytics.tracker.episodes

    st.markdown('<div class="panel-header">📅 EPISODE TIMELINE</div>',
                unsafe_allow_html=True)

    if not eps:
        st.info("No episodes recorded yet. Start the simulation.")
        return

    # ── Scatter: reward over time, colored by success ───────────────
    if _PLOTLY:
        ep_idxs   = list(range(len(eps)))
        rewards   = [e["reward"]  for e in eps]
        successes = [e["success"] for e in eps]
        levels    = [e.get("level", 1) for e in eps]
        steps     = [e["steps"]   for e in eps]
        opts      = [e.get("optimality", 0) for e in eps]

        colors = ["#22c55e" if s else "#ef4444" for s in successes]

        fig_scatter = go.Figure()
        # Failure trace
        fail_x = [i for i, s in enumerate(successes) if not s]
        fail_y = [rewards[i] for i in fail_x]
        fig_scatter.add_trace(go.Scatter(
            x=fail_x, y=fail_y, mode="markers",
            marker=dict(color="rgba(239,68,68,.45)", size=5, symbol="circle"),
            name="Failed",
        ))
        # Success trace
        succ_x = [i for i, s in enumerate(successes) if s]
        succ_y = [rewards[i] for i in succ_x]
        fig_scatter.add_trace(go.Scatter(
            x=succ_x, y=succ_y, mode="markers",
            marker=dict(color="rgba(34,197,94,.65)", size=6, symbol="diamond"),
            name="Success",
        ))
        # EMA overlay
        from analytics import exponential_moving_average
        ema_r = exponential_moving_average(rewards, alpha=0.05)
        fig_scatter.add_trace(go.Scatter(
            x=ep_idxs, y=ema_r, mode="lines",
            line=dict(color="#00f5ff", width=2.0), name="EMA",
        ))
        fig_scatter.update_layout(**{
            **_L,
            "title": dict(text="Reward per Episode (Success=🟢 / Fail=🔴)", font=dict(size=10, color="#8b949e")),
            "height": 280,
            "legend": dict(font=dict(size=9, color="#8b949e"), bgcolor="rgba(0,0,0,0)"),
        })
        st.plotly_chart(fig_scatter, use_container_width=True, key="cht_timeline_scatter")

        # ── 3-panel: Steps / Optimality / Level ─────────────────────
        t1, t2, t3 = st.columns(3)

        with t1:
            fig_st = _line_chart(
                ep_idxs,
                [{"y": steps, "name": "Steps", "color": "#58a6ff", "w": 1.2}],
                "Steps per Episode", 220,
            )
            st.plotly_chart(fig_st, use_container_width=True, key="cht_tl_steps")

        with t2:
            fig_opt = _line_chart(
                ep_idxs,
                [{"y": opts,  "name": "Optimality", "color": "#22c55e", "w": 1.4,
                  "fill": "tozeroy", "fillcolor": "rgba(34,197,94,.06)"}],
                "Path Optimality", 220,
            )
            fig_opt.update_layout(yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig_opt, use_container_width=True, key="cht_tl_opt")

        with t3:
            fig_lv = _line_chart(
                ep_idxs,
                [{"y": levels, "name": "Level", "color": "#a855f7", "w": 1.6}],
                "Curriculum Level", 220,
            )
            fig_lv.update_layout(yaxis=dict(range=[0.5, 10.5]))
            st.plotly_chart(fig_lv, use_container_width=True, key="cht_tl_lvl")

    # ── Episode detail table ─────────────────────────────────────────
    st.markdown("")
    st.markdown('<div class="panel-header">🔍 EPISODE DETAIL (last 100)</div>',
                unsafe_allow_html=True)

    col_filter, col_sort = st.columns([2, 1])
    show_only = col_filter.selectbox(
        "Filter", ["All", "Successes Only", "Failures Only"], key="tl_filter",
        label_visibility="collapsed",
    )
    sort_by = col_sort.selectbox(
        "Sort By", ["Episode ↓", "Reward ↓", "Steps ↓", "Optimality ↓"],
        key="tl_sort", label_visibility="collapsed",
    )

    rows = []
    for idx, e in enumerate(eps):
        if show_only == "Successes Only" and not e["success"]:
            continue
        if show_only == "Failures Only" and e["success"]:
            continue
        rows.append({
            "Ep":          idx,
            "✓":           "✅" if e["success"] else "❌",
            "Reward":      round(e["reward"], 3),
            "Steps":       e["steps"],
            "Optimality":  f'{e.get("optimality",0)*100:.1f}%',
            "Level":       e.get("level", 1),
            "Fog":         f'{e.get("fog_coverage",1.0)*100:.0f}%',
        })

    if rows:
        sort_map = {
            "Episode ↓":    ("Ep",         True),
            "Reward ↓":     ("Reward",      True),
            "Steps ↓":      ("Steps",       True),
            "Optimality ↓": ("Optimality",  True),
        }
        col_s, desc = sort_map.get(sort_by, ("Ep", True))
        rows_sorted = sorted(rows, key=lambda r: r[col_s]
                             if isinstance(r[col_s], (int, float)) else 0, reverse=desc)[-100:]
        st.dataframe(pd.DataFrame(rows_sorted), use_container_width=True, hide_index=True)

    # ── Cumulative success rate ──────────────────────────────────────
    if eps and _PLOTLY:
        st.markdown("")
        st.markdown('<div class="panel-header">📈 CUMULATIVE SUCCESS RATE</div>',
                    unsafe_allow_html=True)
        cumulative = []
        s_count = 0
        for i, e in enumerate(eps):
            s_count += int(e["success"])
            cumulative.append(s_count / (i + 1))
        fig_cum = _line_chart(
            list(range(len(cumulative))),
            [{"y": cumulative, "name": "Cum. Success", "color": "#00f5ff", "w": 2.0,
              "fill": "tozeroy", "fillcolor": "rgba(0,245,255,.05)"}],
            "Cumulative Success Rate",
            200,
        )
        fig_cum.update_layout(yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig_cum, use_container_width=True, key="cht_cum_succ")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 — BENCHMARK & DIAGNOSTICS
# ═════════════════════════════════════════════════════════════════════════════
def _tab_benchmark():
    ss   = st.session_state
    live = ss.analytics.get_live_stats()
    br_st = ss.brain.get_stats()

    st.markdown('<div class="panel-header">🏆 PERFORMANCE BENCHMARKS</div>',
                unsafe_allow_html=True)

    # ── Capability gauge ─────────────────────────────────────────────
    cap = ss.capability_score or 0.0
    cap_pct = cap / 100.0
    tier_label, tier_color = (
        ("NOVICE",    "#ef4444") if cap < 20  else
        ("LEARNING",  "#f97316") if cap < 40  else
        ("COMPETENT", "#fbbf24") if cap < 60  else
        ("PROFICIENT","#22c55e") if cap < 80  else
        ("EXPERT",    "#00f5ff")
    )

    st.markdown(
        f'<div style="text-align:center;padding:18px 0;">'
        f'  <div style="font-size:3.2rem;font-weight:900;font-family:\'JetBrains Mono\',monospace;'
        f'              color:{tier_color};letter-spacing:.04em;">{cap:.1f}</div>'
        f'  <div style="font-size:.75rem;color:#6e7681;letter-spacing:.1em;">CAPABILITY SCORE / 100</div>'
        f'  <div style="margin:10px auto;max-width:350px;">{_pb(cap_pct, tier_color)}</div>'
        f'  <span class="badge" style="background:rgba(0,0,0,.3);'
        f'    border:1px solid {tier_color};color:{tier_color};font-size:.8rem;">'
        f'    {tier_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Score breakdown ──────────────────────────────────────────────
    st.markdown('<div class="panel-header">🔢 SCORE BREAKDOWN</div>',
                unsafe_allow_html=True)

    success_score   = ss.analytics.tracker.success_rate * 40.0
    opt_score       = ss.analytics.tracker.avg_optimality * 25.0
    H, W            = ss.env.maze.shape
    cov             = ss.analytics.heatmap.coverage(H, W, ss.env.maze)
    explore_score   = min(cov, 1.0) * 15.0
    conv_state      = ss.analytics.tracker.convergence.state
    conv_score      = {
        'warming_up': 2.0, 'rapid_learning': 8.0, 'fine_tuning': 7.0,
        'converged': 10.0, 'plateau': 4.0, 'regressing': 0.0,
    }.get(conv_state, 0.0)
    level_score     = ((ss.brain.curriculum.level - 1) / 9.0) * 10.0

    components = [
        ("Success Rate (×40)",   success_score,  40.0, "#22c55e"),
        ("Path Efficiency (×25)", opt_score,      25.0, "#00f5ff"),
        ("Exploration (×15)",    explore_score,  15.0, "#58a6ff"),
        ("Convergence (×10)",    conv_score,     10.0, "#a855f7"),
        ("Curriculum Level (×10)", level_score,  10.0, "#f97316"),
    ]

    bm1, bm2 = st.columns(2)
    for i, (label, score, max_score, color) in enumerate(components):
        col = bm1 if i % 2 == 0 else bm2
        with col:
            st.markdown(
                f'<div style="margin-bottom:10px;">'
                f'  <div style="display:flex;justify-content:space-between;'
                f'              font-size:.72rem;margin-bottom:3px;">'
                f'    <span style="color:#c9d1d9;">{label}</span>'
                f'    <span style="color:{color};font-weight:700;">'
                f'      {score:.1f} / {max_score:.0f}</span>'
                f'  </div>'
                f'  {_pb(score / max_score if max_score > 0 else 0, color)}'
                f'</div>',
                unsafe_allow_html=True,
            )

    if _PLOTLY:
        # Radar chart of score breakdown
        labels   = ["Success", "Efficiency", "Exploration", "Convergence", "Curriculum"]
        vals_raw = [success_score/40.0, opt_score/25.0, explore_score/15.0,
                    conv_score/10.0, level_score/10.0]
        vals_pct = vals_raw + [vals_raw[0]]
        labs_c   = labels + [labels[0]]

        fig_radar = go.Figure(go.Scatterpolar(
            r=vals_pct, theta=labs_c, fill="toself",
            fillcolor="rgba(0,245,255,.10)",
            line=dict(color="#00f5ff", width=2),
            marker=dict(size=6, color="#00f5ff"),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(8,8,24,.7)",
                radialaxis=dict(visible=True, range=[0, 1],
                    gridcolor="rgba(255,255,255,.06)",
                    tickfont=dict(size=7, color="#6e7681")),
                angularaxis=dict(
                    gridcolor="rgba(255,255,255,.06)",
                    tickfont=dict(size=9, color="#c9d1d9")),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=40, r=40, t=30, b=30),
            height=300,
            title=dict(text="Capability Radar", font=dict(size=10, color="#8b949e")),
        )
        st.plotly_chart(fig_radar, use_container_width=True, key="cht_radar_cap")

    st.markdown("---")

    # ── Diagnostics ──────────────────────────────────────────────────
    st.markdown('<div class="panel-header">🩺 SYSTEM DIAGNOSTICS</div>',
                unsafe_allow_html=True)

    checks = [
        ("Buffer filled",       len(ss.brain.memory) >= ss.brain.batch_size,
         f"{len(ss.brain.memory):,} / {ss.brain.batch_size}"),
        ("Learning started",    ss.brain.train_step > 0,
         f"{ss.brain.train_step:,} steps"),
        ("Epsilon below 0.5",   ss.brain.epsilon < 0.5,
         f"ε={ss.brain.epsilon:.4f}"),
        ("Non-zero success",    ss.analytics.tracker.success_rate > 0,
         f"{ss.analytics.tracker.success_rate*100:.1f}%"),
        ("Positive avg reward", ss.analytics.tracker.avg_reward > 0,
         f"{ss.analytics.tracker.avg_reward:+.3f}"),
        ("Not regressing",      conv_state != "regressing",
         conv_state),
        ("Level > 1",           ss.brain.curriculum.level > 1,
         f"L{ss.brain.curriculum.level}"),
        ("LR not bottomed",     ss.brain.learning_rate > 1e-5,
         f"{ss.brain.learning_rate:.7f}"),
    ]

    dc1, dc2 = st.columns(2)
    for i, (name, ok, detail) in enumerate(checks):
        col = dc1 if i % 2 == 0 else dc2
        icon  = "✅" if ok  else "⚠️"
        color = "#22c55e" if ok else "#f97316"
        col.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;'
            f'            margin:4px 0;font-size:.77rem;">'
            f'  <span>{icon}</span>'
            f'  <span style="color:#c9d1d9;">{name}</span>'
            f'  <span style="margin-left:auto;color:{color};font-size:.68rem;">'
            f'    {detail}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Environment diagnostics ──────────────────────────────────────
    st.markdown('<div class="panel-header">🌐 ENVIRONMENT DIAGNOSTICS</div>',
                unsafe_allow_html=True)
    env_st = ss.env.get_stats()
    rd     = ss.env.get_render_data()

    ed1, ed2, ed3, ed4 = st.columns(4)
    ed1.metric("Maze Size",   f'{env_st["maze_h"]}×{env_st["maze_w"]}')
    ed2.metric("Algorithm",   rd["algorithm"].upper())
    ed3.metric("Total Cells", env_st["total_cells"])
    ed4.metric("Passable",    int((ss.env.maze == 0).sum()))

    ed5, ed6, ed7, ed8 = st.columns(4)
    ed5.metric("Traps",       env_st["traps"])
    ed6.metric("Portals",     len(rd["portals_a"]))
    ed7.metric("Fog Active",  "Yes" if rd["use_fog"] else "No")
    ed8.metric("Exploration", f'{cov*100:.1f}%')

    st.markdown("---")

    # ── Session export ───────────────────────────────────────────────
    st.markdown('<div class="panel-header">📤 SESSION EXPORT</div>',
                unsafe_allow_html=True)
    ex1, ex2 = st.columns(2)

    with ex1:
        st.markdown("**Full Session Report**")
        st.code(ss.analytics.get_session_report(), language=None)

    with ex2:
        st.markdown("**JSON Data Preview**")
        json_str = ss.analytics.export_json()
        st.code(json_str[:2000] + ("\n...[truncated]" if len(json_str) > 2000 else ""),
                language="json")

    zb = _export_zip()
    if zb:
        st.download_button(
            "⬇️ Download Full ZIP Checkpoint",
            data=zb,
            file_name=f"ALIVE_NEXUS_ep{ss.episode_count}_L{ss.brain.curriculum.level}.zip",
            mime="application/zip",
            use_container_width=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═════════════════════════════════════════════════════════════════════════════
def _main():
    ss = st.session_state

    # ── Global header ────────────────────────────────────────────────
    h1, h2, h3 = st.columns([4, 3, 2])
    with h1:
        st.markdown(
            '<div class="nexus-title">🧬 A.L.I.V.E. NEXUS</div>'
            '<div style="font-size:.68rem;color:#6e7681;letter-spacing:.06em;margin-top:2px;">'
            '  Adaptive Learning Intelligence &amp; Virtual Evolution</div>',
            unsafe_allow_html=True,
        )
    with h2:
        live = ss.analytics.get_live_stats()
        soul_s = ss.soul.get_status()
        st.markdown(
            f'<div style="text-align:center;font-size:.72rem;line-height:2.0;padding-top:6px;">'
            f'  <span class="badge badge-c">EP {ss.episode_count}</span>'
            f'  <span class="badge badge-g">✓ {live["success_rate"]:.1f}%</span>'
            f'  <span class="badge badge-p">L{ss.brain.curriculum.level}</span>'
            f'  <span class="badge badge-o">ε {ss.brain.epsilon:.4f}</span>'
            f'  <span style="font-size:1.1rem;">{soul_s["mood_emoji"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with h3:
        auto_label = "⏸ PAUSE" if ss.auto_mode else "▶ AUTO RUN"
        if st.button(auto_label, use_container_width=True, key="hdr_toggle"):
            ss.auto_mode = not ss.auto_mode
        if st.button("⏭ STEP ×1", use_container_width=True, key="hdr_step"):
            process_step()
            st.rerun()

    st.markdown("---")

    # ── Backend error banner ─────────────────────────────────────────
    if not _PLOTLY:
        st.warning("⚠️ Plotly not found — charts disabled. `pip install plotly`")

    # ── Sidebar ──────────────────────────────────────────────────────
    _sidebar()

    # ── Tabs ─────────────────────────────────────────────────────────
    tabs = st.tabs([
        "🗺️ Mission Control",
        "📊 Analytics Lab",
        "🧠 Soul Matrix",
        "🗄️ Memory Palace",
        "🔬 Brain Autopsy",
        "📅 Episode Timeline",
        "🏆 Benchmark",
    ])

    with tabs[0]:
        _tab_mission()
    with tabs[1]:
        _tab_analytics()
    with tabs[2]:
        _tab_soul()
    with tabs[3]:
        _tab_memory()
    with tabs[4]:
        _tab_brain()
    with tabs[5]:
        _tab_timeline()
    with tabs[6]:
        _tab_benchmark()

    # ── Auto-run loop ────────────────────────────────────────────────
    if ss.auto_mode:
        spf = ss.config.get("steps_per_frame", 1)
        for _ in range(spf):
            process_step()
        delay = ss.config.get("sim_speed", 0.04)
        if delay > 0:
            time.sleep(delay)
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
_main()
