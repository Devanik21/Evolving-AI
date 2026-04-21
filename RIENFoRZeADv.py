"""
app.py — A.L.I.V.E. NEXUS  v3.1  (Cloud-Optimised)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Speed strategy
  • Plotly figures are CACHED in session_state and only rebuilt every
    CHART_REFRESH_EVERY steps (or on demand).  st.line_chart (native)
    is used for the high-frequency streaming metrics.
  • All tab bodies execute on every rerun (Streamlit limitation) but
    heavy work is guarded behind the cached figure system.
  • HTML-injection is minimised on the simulation hot-path.
  • auto-run uses minimal delay=0 + st.rerun(); UI richness is
    preserved when paused.

Backends (same directory):
  world.py  brain.py  soul.py  memory_palace.py  analytics.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations
import io, json, math, os, random, time, zipfile
from collections import deque
from typing import Dict, List, Optional

import numpy  as np
import pandas as pd
import streamlit as st

# ── backend imports ────────────────────────────────────────────────────────────
try:
    from world         import MazeEnvironment
    from brain         import AgentBrain
    from soul          import SoulCore
    from memory_palace import MemoryPalace
    from analytics     import PerformanceDashboard
    _BACKENDS_OK = True; _BACKEND_ERR = ""
except ImportError as _e:
    _BACKENDS_OK = False; _BACKEND_ERR = str(_e)

try:
    import plotly.graph_objects as go
    from   plotly.subplots      import make_subplots
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

# ── page config (must be first) ────────────────────────────────────────────────
st.set_page_config(page_title="A.L.I.V.E. NEXUS", layout="wide",
                   initial_sidebar_state="expanded", page_icon="🧬")

# ── constants ──────────────────────────────────────────────────────────────────
SAVE_PATH            = "./alive_nexus.json"
STATE_SIZE           = 17
ACTION_SIZE          = 4
CHART_REFRESH_EVERY  = 25   # rebuild Plotly figures every N global steps

DEFAULT_CFG: Dict = dict(
    sim_speed=0.03, steps_per_frame=1, autosave_interval=100,
    gamma=0.99, epsilon_min=0.04, epsilon_decay=0.997,
    lr=0.001, batch_size=64, buffer_size=50_000,
    n_steps=3, icm_beta=0.05, tau=0.005,
    h1=256, h2=128, h3=64,
    show_astar=False, compact_maze=False,
    chart_points=200, override_curriculum=False, manual_level=1,
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&display=swap');
.stApp{background:radial-gradient(ellipse at 20% 10%,#0d0d2e 0%,#080818 55%,#0a1a18 100%);color:#c9d1d9;}
*{box-sizing:border-box;}
.nt{font-family:'JetBrains Mono',monospace;font-size:1.9rem;font-weight:700;
   background:linear-gradient(90deg,#00f5ff 0%,#a855f7 40%,#f97316 70%,#00f5ff 100%);
   background-size:300% 100%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;
   animation:sh 4s linear infinite;}
@keyframes sh{to{background-position:-300% 0;}}
.ph{font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#a855f7;
    text-transform:uppercase;letter-spacing:.14em;
    padding:3px 0 5px;border-bottom:1px solid rgba(168,85,247,.22);margin-bottom:9px;}
.kc{background:rgba(0,245,255,.03);border:1px solid rgba(0,245,255,.11);border-radius:9px;
    padding:9px 12px;text-align:center;transition:border-color .2s,box-shadow .2s;}
.kc:hover{border-color:rgba(0,245,255,.32);box-shadow:0 0 12px rgba(0,245,255,.1);}
.kv{font-family:'JetBrains Mono',monospace;font-size:1.45rem;font-weight:700;color:#00f5ff;line-height:1.15;}
.kl{font-size:.62rem;color:#8b949e;text-transform:uppercase;letter-spacing:.1em;margin-top:2px;}
.ks{font-size:.6rem;color:#58a6ff;margin-top:2px;}
.tb{background:rgba(168,85,247,.07);border-left:3px solid #a855f7;border-radius:0 8px 8px 0;
    padding:9px 13px;font-size:.82rem;font-style:italic;color:#d8b4fe;line-height:1.5;margin:5px 0;}
.cu{background:rgba(88,166,255,.09);border-right:3px solid #58a6ff;border-radius:9px 0 0 9px;
    padding:7px 11px;margin:4px 0;text-align:right;font-size:.84rem;color:#cdd9e5;}
.ca{background:rgba(168,85,247,.08);border-left:3px solid #a855f7;border-radius:0 9px 9px 0;
    padding:7px 11px;margin:4px 0;font-size:.84rem;color:#d8b4fe;}
.cm{font-size:.6rem;color:#6e7681;margin-top:1px;}
.bd{display:inline-block;padding:1px 7px;border-radius:15px;font-size:.63rem;font-weight:700;
    text-transform:uppercase;letter-spacing:.05em;font-family:'JetBrains Mono',monospace;margin:1px 2px;}
.bc{background:rgba(0,245,255,.11);color:#00f5ff;border:1px solid rgba(0,245,255,.28);}
.bp{background:rgba(168,85,247,.11);color:#a855f7;border:1px solid rgba(168,85,247,.28);}
.bg{background:rgba(34,197,94,.11);color:#22c55e;border:1px solid rgba(34,197,94,.28);}
.br{background:rgba(239,68,68,.11);color:#ef4444;border:1px solid rgba(239,68,68,.28);}
.bo{background:rgba(249,115,22,.11);color:#f97316;border:1px solid rgba(249,115,22,.28);}
.pb{background:rgba(255,255,255,.07);border-radius:4px;height:6px;overflow:hidden;}
.pf{height:100%;border-radius:4px;transition:width .35s ease;}
.mc{background:rgba(0,0,0,.55);border:1px solid rgba(0,245,255,.11);border-radius:8px;
    padding:8px;font-size:.68rem;line-height:1.2;overflow:auto;max-height:360px;}
.fc{background:rgba(34,197,94,.05);border:1px solid rgba(34,197,94,.13);border-radius:6px;
    padding:6px 10px;margin:2px 0;font-size:.76rem;}
.mt{background:rgba(168,85,247,.06);border:1px solid rgba(168,85,247,.13);border-radius:6px;
    padding:6px 10px;margin:2px 0;font-size:.76rem;color:#c9d1d9;}
.cv-w{background:rgba(59,130,246,.09);border:1px solid rgba(59,130,246,.22);}
.cv-r{background:rgba(34,197,94,.09);border:1px solid rgba(34,197,94,.22);}
.cv-f{background:rgba(234,179,8,.09);border:1px solid rgba(234,179,8,.22);}
.cv-o{background:rgba(0,245,255,.09);border:1px solid rgba(0,245,255,.22);}
.cv-p{background:rgba(168,85,247,.09);border:1px solid rgba(168,85,247,.22);}
.cv-g{background:rgba(239,68,68,.09);border:1px solid rgba(239,68,68,.22);}
.cv-base{border-radius:7px;padding:9px 13px;margin:5px 0;display:flex;align-items:center;gap:9px;font-size:.81rem;}
.stButton>button{background:linear-gradient(135deg,rgba(0,245,255,.07),rgba(168,85,247,.07));
    border:1px solid rgba(0,245,255,.22);color:#00f5ff;border-radius:7px;
    font-family:'JetBrains Mono',monospace;font-size:.74rem;font-weight:600;letter-spacing:.04em;transition:all .18s;}
.stButton>button:hover{border-color:rgba(0,245,255,.55);background:rgba(0,245,255,.14);
    box-shadow:0 0 10px rgba(0,245,255,.18);transform:translateY(-1px);}
div[data-testid="stMetric"]{background:rgba(255,255,255,.022);border:1px solid rgba(255,255,255,.05);border-radius:8px;padding:6px 9px;}
div[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace;color:#00f5ff;}
.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,.018);border:1px solid rgba(255,255,255,.05);
    border-radius:8px;gap:2px;padding:3px;}
.stTabs [data-baseweb="tab"]{font-family:'JetBrains Mono',monospace;font-size:.71rem;
    letter-spacing:.05em;color:#8b949e;border-radius:5px;}
.stTabs [aria-selected="true"]{background:rgba(0,245,255,.09)!important;color:#00f5ff!important;}
.stCode,.stCodeBlock{background:#0a0a18!important;border:1px solid rgba(0,245,255,.11)!important;
    border-radius:7px!important;font-size:.68rem!important;line-height:1.1!important;}
section[data-testid="stSidebar"]{background:rgba(8,8,24,.98);border-right:1px solid rgba(0,245,255,.07);}

div[data-testid="stDecoration"]{display:none;}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# BACKEND GUARD
# ══════════════════════════════════════════════════════════════════════════════
if not _BACKENDS_OK:
    st.error(f"❌ Backend import failed: `{_BACKEND_ERR}`")
    st.info("Put world.py  brain.py  soul.py  memory_palace.py  analytics.py in the same folder.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SESSION INIT
# ══════════════════════════════════════════════════════════════════════════════
def _init():
    ss = st.session_state
    cfg = ss.get("config", dict(DEFAULT_CFG));  ss.config = cfg
    bk  = {k: cfg[k] for k in ("gamma","epsilon_min","epsilon_decay","lr",
           "batch_size","buffer_size","n_steps","icm_beta","tau","h1","h2","h3")}
    ss.brain     = AgentBrain(STATE_SIZE, ACTION_SIZE, config=bk)
    ss.env       = MazeEnvironment()
    ss.soul      = SoulCore(name="Nik")
    ss.memory    = MemoryPalace(save_path=SAVE_PATH)
    ss.analytics = PerformanceDashboard()
    lc           = ss.brain.curriculum.config
    ss.cur_state = ss.env.reset(config=lc)
    ss._prev_cells = 1
    ss.memory.start_episode(ss.env.seed, ss.env.algorithm,
        ss.env.maze_h, ss.env.maze_w,
        ss.brain.curriculum.level, ss.brain.epsilon, ss.env.max_steps)
    ss.auto_mode      = False
    ss.global_step    = 0
    ss.episode_count  = 0
    ss.cap_score      = 0.0
    ss.last_reward    = 0.0
    ss.last_success   = False
    # figure cache
    ss._figs          = {}        # key → (step_built, figure)
    ss._chat_scroll   = 0

if "brain" not in st.session_state:
    _init()

ss = st.session_state   # convenience alias

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE CACHE  — lazy, throttled Plotly rendering
# ══════════════════════════════════════════════════════════════════════════════
def _fig_stale(key: str) -> bool:
    """True if the cached figure is older than CHART_REFRESH_EVERY steps."""
    entry = ss._figs.get(key)
    if entry is None:
        return True
    step_built, _ = entry
    return (ss.global_step - step_built) >= CHART_REFRESH_EVERY

def _store_fig(key: str, fig):
    ss._figs[key] = (ss.global_step, fig)

def _get_fig(key: str):
    entry = ss._figs.get(key)
    return entry[1] if entry else None

def _show(key: str, builder_fn, **kwargs):
    """Build+cache a Plotly figure only when stale; always display it."""
    if _PLOTLY:
        if _fig_stale(key):
            _store_fig(key, builder_fn(**kwargs))
        fig = _get_fig(key)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key=f"plt_{key}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY BUILDERS  (pure functions, no Streamlit calls)
# ══════════════════════════════════════════════════════════════════════════════
_L = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,8,24,.85)",
          font=dict(color="#8b949e", family="JetBrains Mono,monospace", size=10),
          margin=dict(l=36,r=14,t=32,b=28),
          xaxis=dict(gridcolor="rgba(255,255,255,.04)", zeroline=False),
          yaxis=dict(gridcolor="rgba(255,255,255,.04)", zeroline=False))

def _build_lines(xs, traces, title="", height=210):
    fig = go.Figure()
    for t in traces:
        fig.add_trace(go.Scatter(x=xs, y=t["y"], name=t["name"], mode="lines",
            line=dict(color=t.get("c","#00f5ff"), width=t.get("w",1.5)),
            fill=t.get("fill"), fillcolor=t.get("fc")))
    l = dict(_L); l["title"]=dict(text=title,font=dict(size=10,color="#8b949e")); l["height"]=height
    fig.update_layout(**l)
    return fig

def _build_emotion(v, a):
    fig = go.Figure()
    for cx,cy,r,lbl,col in [(0.65,0.55,.30,"Excited","rgba(249,115,22,.09)"),
                              (0.65,-0.50,.28,"Serene","rgba(34,197,94,.09)"),
                              (-0.60,0.60,.28,"Alarmed","rgba(239,68,68,.07)"),
                              (-0.60,-0.40,.26,"Depressed","rgba(88,166,255,.08)"),
                              (0.60,0.10,.22,"Happy","rgba(0,245,255,.07)"),
                              (0.00,0.00,.16,"Neutral","rgba(168,85,247,.06)")]:
        fig.add_shape(type="circle",x0=cx-r,y0=cy-r,x1=cx+r,y1=cy+r,fillcolor=col,line_width=0)
        fig.add_annotation(x=cx,y=cy,text=lbl,font=dict(size=8,color="rgba(180,180,180,.4)"),showarrow=False)
    for v_ in [-1,0,1]:
        fig.add_hline(y=v_,line_color="rgba(255,255,255,.05)",line_width=.5)
        fig.add_vline(x=v_,line_color="rgba(255,255,255,.05)",line_width=.5)
    fig.add_trace(go.Scatter(x=[v],y=[a],mode="markers",
        marker=dict(size=28,color="rgba(168,85,247,.16)",line=dict(color="rgba(168,85,247,.35)",width=1)),showlegend=False))
    fig.add_trace(go.Scatter(x=[v],y=[a],mode="markers",
        marker=dict(size=12,color="#a855f7",line=dict(color="#d8b4fe",width=2)),showlegend=False))
    l=dict(_L); l.update(
        title=dict(text="Emotion Circumplex (Russell)",font=dict(size=10,color="#8b949e")),
        xaxis=dict(**_L["xaxis"],title="Valence →",range=[-1.2,1.2]),
        yaxis=dict(**_L["yaxis"],title="Arousal ↑",range=[-1.2,1.2]),height=290)
    fig.update_layout(**l); return fig

def _build_ocean(O,C,E,A,N):
    cats=["Openness","Conscientious","Extraversion","Agreeableness","Neuroticism"]
    vals=[O,C,E,A,N,O]; cats_c=cats+[cats[0]]
    fig=go.Figure(go.Scatterpolar(r=vals,theta=cats_c,fill="toself",
        fillcolor="rgba(168,85,247,.13)",line=dict(color="#a855f7",width=2),
        marker=dict(size=5,color="#a855f7")))
    fig.update_layout(polar=dict(bgcolor="rgba(8,8,24,.7)",
        radialaxis=dict(visible=True,range=[0,1],gridcolor="rgba(255,255,255,.05)",tickfont=dict(size=7,color="#6e7681")),
        angularaxis=dict(gridcolor="rgba(255,255,255,.05)",tickfont=dict(size=8,color="#c9d1d9"))),
        paper_bgcolor="rgba(0,0,0,0)",showlegend=False,
        margin=dict(l=28,r=28,t=32,b=28),height=270,
        title=dict(text="Personality (OCEAN)",font=dict(size=10,color="#8b949e")))
    return fig

def _build_heatmap(data, title="Heatmap"):
    fig=go.Figure(go.Heatmap(z=data[::-1],
        colorscale=[[0,"rgba(8,8,24,1)"],[.15,"rgba(59,130,246,.6)"],
                    [.45,"rgba(168,85,247,.85)"],[.75,"rgba(239,68,68,.9)"],[1,"rgba(255,230,0,1)"]],
        showscale=False,hovertemplate="(%{x},%{y}): %{z:.2f}<extra></extra>"))
    l=dict(_L); l.update(title=dict(text=title,font=dict(size=10,color="#8b949e")),
        height=230,xaxis=dict(visible=False),yaxis=dict(visible=False))
    fig.update_layout(**l); return fig

def _build_radar_cap(vals_pct):
    labs=["Success","Efficiency","Exploration","Convergence","Curriculum"]
    r=vals_pct+[vals_pct[0]]; t=labs+[labs[0]]
    fig=go.Figure(go.Scatterpolar(r=r,theta=t,fill="toself",
        fillcolor="rgba(0,245,255,.09)",line=dict(color="#00f5ff",width=2),
        marker=dict(size=6,color="#00f5ff")))
    fig.update_layout(polar=dict(bgcolor="rgba(8,8,24,.7)",
        radialaxis=dict(visible=True,range=[0,1],gridcolor="rgba(255,255,255,.05)",tickfont=dict(size=7,color="#6e7681")),
        angularaxis=dict(gridcolor="rgba(255,255,255,.05)",tickfont=dict(size=8,color="#c9d1d9"))),
        paper_bgcolor="rgba(0,0,0,0)",showlegend=False,
        margin=dict(l=30,r=30,t=32,b=28),height=290,
        title=dict(text="Capability Radar",font=dict(size=10,color="#8b949e")))
    return fig

def _build_curriculum_bar(level, scores, promote, demote):
    xs=list(range(1,len(scores)+1))
    cols=["#22c55e" if s>=promote else ("#ef4444" if s<=demote else "#58a6ff") for s in scores]
    fig=go.Figure(go.Bar(x=xs,y=scores,marker_color=cols))
    fig.add_hline(y=promote,line_color="#22c55e",line_dash="dot",annotation_text="→ promote",annotation_font_size=8)
    fig.add_hline(y=demote, line_color="#ef4444",line_dash="dot",annotation_text="→ demote", annotation_font_size=8)
    l=dict(_L); l.update(title=dict(text=f"Curriculum Window — Level {level}",font=dict(size=10,color="#8b949e")),height=195)
    fig.update_layout(**l); return fig

# ══════════════════════════════════════════════════════════════════════════════
# HTML HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def kpi(v,l,s=""):
    return (f'<div class="kc"><div class="kv">{v}</div>'
            f'<div class="kl">{l}</div>'+(f'<div class="ks">{s}</div>' if s else "")+"</div>")

def pb(frac, color="#00f5ff"):
    p=max(0.,min(1.,frac))*100
    return f'<div class="pb"><div class="pf" style="width:{p:.1f}%;background:{color};"></div></div>'

_CONV_CSS = {"warming_up":"cv-w","rapid_learning":"cv-r","fine_tuning":"cv-f",
             "converged":"cv-o","plateau":"cv-p","regressing":"cv-g"}
_CONV_ICO = {"warming_up":"🔥","rapid_learning":"🚀","fine_tuning":"⚙️",
             "converged":"✅","plateau":"📊","regressing":"⬇️"}
_CONV_TXT = {"warming_up":"Filling buffer. Learning not yet started.",
             "rapid_learning":"Strong positive gradient — policy improving fast.",
             "fine_tuning":"Gradual improvement — policy consolidating.",
             "converged":"Stable performance. Policy at equilibrium.",
             "plateau":"No clear trend. High variance. Possibly stuck.",
             "regressing":"Negative trend detected. Check hyperparams."}

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def _episode_done(info):
    es  = ss.env.get_stats()
    lvl = ss.brain.curriculum.level
    ss.brain.curriculum.record(bool(info.get("reached")), es["step_count"],
                               es["max_steps"], es["episode_reward"])
    ss.memory.end_episode(es["episode_reward"], es["step_count"],
        bool(info.get("reached")), es["cells_visited"], es["astar_optimal"],
        es["fog"], es["traps"]>0, ss.brain.avg_td_error, ss.brain.epsilon)
    H,W = ss.env.maze.shape
    ss.cap_score = ss.analytics.record_episode(
        es["episode_reward"], es["step_count"], bool(info.get("reached")),
        {"optimality":info.get("optimality",0.),"fog_coverage":info.get("fog_coverage",1.),"level":lvl},
        lvl, H, W, ss.env.maze)
    ss.last_reward  = es["episode_reward"]
    ss.last_success = bool(info.get("reached"))
    ss.episode_count += 1
    if ss.episode_count % ss.config.get("autosave_interval",100) == 0:
        _do_save()
    new_cfg = ss.brain.curriculum.config
    if ss.config.get("override_curriculum"):
        new_cfg = dict(ss.brain.curriculum.LEVEL_CONFIGS.get(
            ss.config.get("manual_level",1), new_cfg))
    ss.cur_state = ss.env.reset(config=new_cfg)
    ss._prev_cells = 1
    ss.memory.start_episode(ss.env.seed, ss.env.algorithm,
        ss.env.maze_h, ss.env.maze_w,
        ss.brain.curriculum.level, ss.brain.epsilon, ss.env.max_steps)

def process_step():
    state  = ss.cur_state
    action = ss.brain.act(state)
    nxt, reward, done, info = ss.env.step(action)
    loss, td = ss.brain.step(state, action, reward, nxt, done)
    ss.memory.record_transition(state, action, reward, nxt, done)
    ss.analytics.record_step(ss.env.agent_r, ss.env.agent_c, loss, td, ss.brain.epsilon)
    trap_near = bool(ss.env.traps and any(
        abs(t.r-ss.env.agent_r)+abs(t.c-ss.env.agent_c)<=3 for t in ss.env.traps))
    nc = len(ss.env.cells_visited); is_new = nc > ss._prev_cells; ss._prev_cells = nc
    ss.soul.update_from_rl(
        {"epsilon":ss.brain.epsilon,"avg_loss":ss.brain.avg_loss,
         "avg_td_error":ss.brain.avg_td_error,"train_step":ss.brain.train_step,
         "avg_reward":ss.brain.avg_reward,"curriculum":ss.brain.curriculum.get_stats()},
        {"reward":reward,"reached":info.get("reached",False),"timeout":info.get("timeout",False),
         "trap_hit":info.get("trap_hit",False),"trap_nearby":trap_near,"portal_used":False,
         "is_new_cell":is_new,"success_count":ss.env.success_count,
         "success_rate":ss.env.success_count/max(ss.env.total_episodes,1),
         "cells_visited":nc,"maze_size":f"{ss.env.maze_h}×{ss.env.maze_w}"})
    ss.cur_state = nxt; ss.global_step += 1
    if done: _episode_done(info)

def reset_all():
    for k in ["brain","env","soul","memory","analytics","cur_state","global_step",
              "episode_count","cap_score","last_reward","last_success","_prev_cells","_figs"]:
        st.session_state.pop(k, None)
    st.session_state.auto_mode = False
    _init()

# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════
def _np(o):
    if isinstance(o,np.integer): return int(o)
    if isinstance(o,np.floating): return float(o)
    if isinstance(o,np.ndarray): return o.tolist()
    if isinstance(o,np.bool_): return bool(o)
    if isinstance(o,deque): return list(o)
    return str(o)

def _do_save():
    try:
        return ss.memory.save_all(ss.brain.get_weights(),
            ss.analytics.tracker.session_summary(), ss.soul.get_status())
    except Exception as e:
        st.toast(f"⚠️ Save error: {e}"); return False

def _export_zip():
    try:
        p={"version":"3.1","saved_at":time.time(),"config":ss.config,
           "brain":ss.brain.get_weights(),"soul":ss.soul.get_status(),
           "analytics":ss.analytics.tracker.session_summary(),
           "memory":ss.memory.get_full_status(),
           "global_step":ss.global_step,"episode_count":ss.episode_count,
           "cap_score":ss.cap_score}
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
            z.writestr("alive_nexus.json", json.dumps(p,indent=2,default=_np))
        return buf.getvalue()
    except: return None

def _load_zip(f):
    try:
        with zipfile.ZipFile(f,"r") as z:
            with z.open("alive_nexus.json") as fp: data=json.load(fp)
        if "brain" in data: ss.brain.set_weights(data["brain"]); ss.brain.target_net.copy_from(ss.brain.online_net)
        if "soul" in data:
            p=data["soul"]
            for t in "OCEAN":
                if t in p: setattr(ss.soul.personality,t,float(p[t]))
            if "relationship" in p: ss.soul.relationship.score=float(p["relationship"])
        ss.global_step=data.get("global_step",0); ss.episode_count=data.get("episode_count",0)
        ss.cap_score=data.get("cap_score",0.); ss.config.update(data.get("config",{}))
        return True
    except Exception as e: st.error(f"❌ Load: {e}"); return False

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def _sidebar():
    cfg = ss.config
    with st.sidebar:
        st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:.88rem;font-weight:700;'
                    'color:#00f5ff;letter-spacing:.06em;padding:6px 0 10px;">🧬 A.L.I.V.E. NEXUS</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="ph">⚡ SIMULATION</div>', unsafe_allow_html=True)
        c1,c2=st.columns(2)
        if c1.button("▶ RUN",  use_container_width=True): ss.auto_mode=True
        if c2.button("⏸ PAUSE",use_container_width=True): ss.auto_mode=False
        c3,c4=st.columns(2)
        if c3.button("⏭ STEP", use_container_width=True):
            for _ in range(cfg.get("steps_per_frame",1)): process_step()
            st.rerun()
        if c4.button("🔄 RESET",use_container_width=True): reset_all(); st.rerun()
        cfg["sim_speed"]=st.slider("Delay (s)",0.0,0.5,cfg.get("sim_speed",0.03),0.01)
        cfg["steps_per_frame"]=st.select_slider("Steps/frame",[1,2,4,8,16,32],cfg.get("steps_per_frame",1))
        st.markdown('<div class="ph">🌐 ENVIRONMENT</div>', unsafe_allow_html=True)
        cfg["override_curriculum"]=st.toggle("Override Curriculum",cfg.get("override_curriculum",False))
        if cfg["override_curriculum"]:
            cfg["manual_level"]=st.slider("Force Level",1,10,cfg.get("manual_level",1))
        cfg["show_astar"]=st.toggle("Show A* Overlay",cfg.get("show_astar",False))
        cfg["compact_maze"]=st.toggle("Compact Maze",cfg.get("compact_maze",False))
        st.markdown('<div class="ph">🧠 HYPERPARAMS</div>', unsafe_allow_html=True)
        with st.expander("Tune", expanded=False):
            cfg["gamma"]=st.slider("γ",0.80,0.999,cfg.get("gamma",0.99),format="%.3f")
            cfg["epsilon_decay"]=st.slider("ε decay",0.990,0.9999,cfg.get("epsilon_decay",0.997),format="%.4f")
            cfg["epsilon_min"]=st.slider("ε min",0.01,0.15,cfg.get("epsilon_min",0.04),0.005)
            cfg["lr"]=st.slider("LR",1e-4,5e-3,cfg.get("lr",1e-3),format="%.4f")
            cfg["tau"]=st.slider("τ",0.001,0.05,cfg.get("tau",0.005),0.001)
            cfg["icm_beta"]=st.slider("ICM β",0.0,0.2,cfg.get("icm_beta",0.05),0.01)
            cfg["n_steps"]=st.select_slider("N-Step",[1,2,3,5,8],cfg.get("n_steps",3))
            cfg["batch_size"]=st.select_slider("Batch",[32,64,128,256],cfg.get("batch_size",64))
        with st.expander("Architecture (→ Reset)", expanded=False):
            st.info("Changing these needs Hard Reset.")
            cfg["h1"]=st.select_slider("H1",[64,128,256,512],cfg.get("h1",256))
            cfg["h2"]=st.select_slider("H2",[32,64,128,256],cfg.get("h2",128))
            cfg["h3"]=st.select_slider("H3",[16,32,64,128],cfg.get("h3",64))
            cfg["buffer_size"]=st.select_slider("Buffer",[10_000,25_000,50_000,100_000],cfg.get("buffer_size",50_000))
        st.markdown('<div class="ph">💾 PERSISTENCE</div>', unsafe_allow_html=True)
        if st.button("💾 Save",use_container_width=True):
            st.toast("✅ Saved!" if _do_save() else "❌ Failed")
        zb=_export_zip()
        if zb: st.download_button("⬇️ Export ZIP",zb,f"alive_ep{ss.episode_count}.zip","application/zip",use_container_width=True)
        up=st.file_uploader("📂 Load ZIP",type="zip",label_visibility="collapsed")
        if up and st.button("Restore",use_container_width=True):
            if _load_zip(up): st.toast("✅ Restored!"); st.rerun()
        st.markdown('<div class="ph">👤 SOUL</div>', unsafe_allow_html=True)
        nn=st.text_input("Your Name",value=ss.soul.user_name,label_visibility="collapsed")
        if nn!=ss.soul.user_name: ss.soul.user_name=nn
        cfg["chart_points"]=st.slider("Chart History",50,500,cfg.get("chart_points",200),25)
        cfg["autosave_interval"]=st.slider("Autosave (eps)",25,500,cfg.get("autosave_interval",100),25)
        bs=ss.brain.get_stats(); sl=ss.soul.get_status()
        st.markdown("---")
        st.markdown(f'<div style="font-size:.69rem;line-height:1.9;">'
            f'<b style="color:#00f5ff;">ε</b> {bs["epsilon"]:.4f}&nbsp; '
            f'<b style="color:#22c55e;">Lvl</b> {bs["curriculum"]["level"]}&nbsp; '
            f'<b style="color:#a855f7;">Mood</b> {sl["mood_emoji"]} {sl["mood"]}<br>'
            f'<b style="color:#f97316;">Steps</b> {ss.global_step:,}&nbsp; '
            f'<b style="color:#58a6ff;">Eps</b> {ss.episode_count}'
            f'</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MISSION CONTROL
# ══════════════════════════════════════════════════════════════════════════════
def _tab_mission():
    cfg=ss.config; rd=ss.env.get_render_data(); es=ss.env.get_stats()
    lv=ss.analytics.get_live_stats(); bs=ss.brain.get_stats()

    left,right=st.columns([3,2],gap="large")
    with left:
        st.markdown('<div class="ph">🗺️ LIVE ENVIRONMENT</div>', unsafe_allow_html=True)
        maze_txt, legend = ss.env.render_ascii()
        # A* overlay
        if cfg.get("show_astar"):
            path=ss.env.get_astar_path()
            if path:
                lines=maze_txt.split("\n")
                for (r,c) in path[1:-1]:
                    row=list(lines[r]); row[c*2:c*2+2]="··"
                    lines[r]="".join(row)
                maze_txt="\n".join(lines)
        fs="0.6rem" if cfg.get("compact_maze") else "0.7rem"
        st.markdown(f'<div class="mc" style="font-size:{fs};"><pre>{maze_txt}</pre></div>',
                    unsafe_allow_html=True)
        st.caption(legend)
        prog=es["step_count"]/max(es["max_steps"],1)
        st.markdown(pb(prog,"#22c55e" if prog<.7 else ("#f97316" if prog<.9 else "#ef4444")),
                    unsafe_allow_html=True)
        cl=ss.brain.curriculum.config
        badges=(f'<span class="bd bc">ALG:{cl["algorithm"].upper()}</span>'
                f'<span class="bd bp">L{ss.brain.curriculum.level}/10</span>'
                f'<span class="bd bo">ε:{ss.brain.epsilon:.3f}</span>'
                f'<span class="bd bg">WINS:{ss.env.success_count}</span>'
                +('<span class="bd br">🌫FOG</span>' if cl["fog"] else "")
                +('<span class="bd br">💀TRAPS</span>' if cl["dynamic"] else "")
                +('<span class="bd bp">🌀PORTALS</span>' if cl["portals"] else ""))
        st.markdown(f'<div style="margin-top:6px;">{badges}</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="ph">🧬 CONSCIOUSNESS</div>', unsafe_allow_html=True)
        sl=ss.soul.get_status()
        st.markdown(f'<div class="tb">{sl["thought"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="ph">📡 BRAIN SNAPSHOT</div>', unsafe_allow_html=True)
        m1,m2,m3=st.columns(3)
        m1.metric("Avg Reward",f'{bs["avg_reward"]:.2f}')
        m2.metric("Avg Loss",f'{bs["avg_loss"]:.4f}')
        m3.metric("TD-Error",f'{bs["avg_td_error"]:.3f}')
        m4,m5,m6=st.columns(3)
        m4.metric("Train Steps",f'{bs["train_step"]:,}')
        m5.metric("Memory",f'{bs["memory_size"]:,}')
        m6.metric("LR",f'{bs["lr"]:.5f}')
        conv=lv["convergence"]
        css=_CONV_CSS.get(conv,"cv-w"); ico=_CONV_ICO.get(conv,""); txt=_CONV_TXT.get(conv,"")
        st.markdown(f'<div class="cv-base {css}">'
                    f'<span style="font-size:1.2rem;">{ico}</span>'
                    f'<span><b style="color:#c9d1d9;">{conv.upper()}</b><br>'
                    f'<span style="font-size:.73rem;color:#8b949e;">{txt}</span></span></div>',
                    unsafe_allow_html=True)
        st.markdown(f"**Curriculum L{ss.brain.curriculum.level}/10**")
        st.markdown(pb(ss.brain.curriculum.zpd_progress,"#a855f7"), unsafe_allow_html=True)
        st.caption(f"ZPD → promote at {ss.brain.curriculum.promote_thresh*100:.0f}%")
        trend=lv["capability_trend"]
        st.metric("🎯 Capability",f'{lv["capability"]:.1f}/100',delta=trend)
        st.metric("✓ Success Rate",f'{lv["success_rate"]:.1f}%')

    # ── fast native charts (always fresh, cheap) ─────────────────────────────
    st.markdown("---")
    st.markdown('<div class="ph">📈 STREAMING METRICS</div>', unsafe_allow_html=True)
    cd=ss.analytics.get_chart_data(cfg.get("chart_points",200))
    c_a,c_b=st.columns(2)
    with c_a:
        if cd["ema_rewards"]:
            st.line_chart(pd.DataFrame({"EMA Reward":cd["ema_rewards"],"Raw":cd["rewards"]}),height=145)
    with c_b:
        if cd["epsilons"]:
            st.line_chart(pd.DataFrame({"Epsilon":cd["epsilons"][-300:]}),height=145)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS LAB
# ══════════════════════════════════════════════════════════════════════════════
def _tab_analytics():
    cfg=ss.config; lv=ss.analytics.get_live_stats()
    cd=ss.analytics.get_chart_data(cfg.get("chart_points",200))
    N=len(cd["rewards"]); xs=list(range(N))

    # KPI strip
    kpis=[
        kpi(f'{lv["success_rate"]:.1f}%',"Success Rate",f'±trend {lv["reward_trend"]:+.4f}'),
        kpi(f'{lv["avg_reward"]:+.2f}',"Avg Reward (50ep)"),
        kpi(f'{lv["avg_steps"]:.0f}',"Avg Steps/Ep"),
        kpi(f'{lv["capability"]:.1f}',"Capability /100",lv["capability_trend"]),
        kpi(f'{ss.episode_count}',"Episodes",f'L{ss.brain.curriculum.level}'),
        kpi(f'{ss.global_step:,}',"Global Steps"),
    ]
    cols=st.columns(len(kpis))
    for col,k in zip(cols,kpis): col.markdown(k, unsafe_allow_html=True)
    st.markdown("---")

    # Convergence banner
    conv=lv["convergence"]; css=_CONV_CSS.get(conv,"cv-w"); ico=_CONV_ICO.get(conv,""); txt=_CONV_TXT.get(conv,"")
    st.markdown(f'<div class="cv-base {css}" style="font-size:.84rem;">'
                f'<span style="font-size:1.4rem;">{ico}</span>'
                f'<div><b style="color:#c9d1d9;">{conv.replace("_"," ").title()}</b><br>'
                f'<span style="font-size:.74rem;color:#8b949e;">{txt}</span></div></div>',
                unsafe_allow_html=True)
    st.markdown("---")

    l1,l2=st.columns(2)
    with l1:
        if N>1:
            _show("reward_line", _build_lines,
                  xs=xs,
                  traces=[{"y":cd["rewards"],"name":"Reward","c":"rgba(0,245,255,.5)","w":1},
                          {"y":cd["ema_rewards"],"name":"EMA","c":"#00f5ff","w":2,
                           "fill":"tonexty","fc":"rgba(0,245,255,.06)"}],
                  title="Episode Reward", height=220)
        if cd["losses"]:
            xs_l=list(range(len(cd["losses"])))
            _show("loss_line", _build_lines,
                  xs=xs_l, traces=[{"y":cd["losses"],"name":"Loss","c":"#f97316","w":1.2}],
                  title="Training Loss", height=200)
    with l2:
        if cd["td_errors"]:
            xs_t=list(range(len(cd["td_errors"])))
            _show("td_line", _build_lines,
                  xs=xs_t, traces=[{"y":cd["td_errors"],"name":"TD-Error","c":"#a855f7","w":1.2}],
                  title="TD-Error", height=200)
        if N>1:
            _show("steps_line", _build_lines,
                  xs=xs, traces=[{"y":cd["steps"],"name":"Steps","c":"#58a6ff","w":1.2}],
                  title="Steps per Episode", height=200)

    st.markdown("---")
    # Heatmaps
    H,W=ss.env.maze.shape
    hm_g=ss.analytics.get_heatmap(H,W,episode=False)
    hm_e=ss.analytics.get_heatmap(H,W,episode=True)
    hc1,hc2=st.columns(2)
    with hc1: _show("heatmap_global",_build_heatmap,data=hm_g,title="Global Visit Heatmap")
    with hc2: _show("heatmap_ep",    _build_heatmap,data=hm_e,title="Episode Visit Heatmap")

    st.markdown("---")
    # Curriculum bar
    win=list(ss.brain.curriculum.window)
    if win:
        _show("curr_bar",_build_curriculum_bar,
              level=ss.brain.curriculum.level, scores=win,
              promote=ss.brain.curriculum.promote_thresh,
              demote=ss.brain.curriculum.demote_thresh)
    c1,c2,c3=st.columns(3)
    c1.metric("Promotions",ss.brain.curriculum.promotions)
    c2.metric("Demotions", ss.brain.curriculum.demotions)
    c3.metric("Avg Score", f'{ss.brain.curriculum.avg_score:.3f}')

    if not _PLOTLY:
        st.warning("⚠️ Install plotly for charts: `pip install plotly`")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SOUL MATRIX
# ══════════════════════════════════════════════════════════════════════════════
def _tab_soul():
    sl=ss.soul.get_status()
    left,right=st.columns([5,4],gap="large")

    with left:
        st.markdown("### 💬 Cognitive Interface")
        st.markdown(f'<div class="tb">{sl["thought"]}</div>', unsafe_allow_html=True)
        chat_box=st.container(height=370)
        with chat_box:
            for msg in ss.soul.get_chat_history():
                if msg["role"]=="user":
                    st.markdown(f'<div class="cu"><b>YOU</b> '
                                f'<span class="cm">intent:{msg.get("intent","?")}</span><br>{msg["text"]}</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="ca"><b>A.L.I.V.E.</b> '
                                f'<span class="cm">mood:{msg.get("emotion","?")}</span><br>{msg["text"]}</div>',
                                unsafe_allow_html=True)
        ui=st.chat_input("Speak to A.L.I.V.E. ...")
        if ui: ss.soul.chat(ui); st.rerun()
        st.markdown("**Quick Prompts**")
        qc=st.columns(4)
        for col,p in zip(qc,["Hello!","How do you feel?","What are you?","Am I your friend?"]):
            if col.button(p,use_container_width=True,key=f"qp_{p}"): ss.soul.chat(p); st.rerun()

    with right:
        st.markdown("### 🌀 Identity Core")
        v=sl["valence"]; a=sl["arousal"]
        st.markdown(f'<div style="text-align:center;padding:8px 0;">'
                    f'<div style="font-size:2.4rem;">{sl["mood_emoji"]}</div>'
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:1rem;color:#00f5ff;">'
                    f'{sl["mood"].upper()}</div>'
                    f'<div style="font-size:.73rem;color:#8b949e;margin-top:4px;">'
                    f'V:{v:+.3f} | A:{a:+.3f} | I:{sl["intensity"]:.3f}</div></div>',
                    unsafe_allow_html=True)
        _show("emotion_plot",_build_emotion,v=v,a=a)
        st.markdown(f'**Relationship:** {sl["stage"]} — {sl["relationship"]}%')
        st.markdown(pb(sl["relationship"]/100,"#22c55e"), unsafe_allow_html=True)
        st.caption(sl["stage_desc"])
        st.markdown("---")
        _show("ocean_radar",_build_ocean,
              O=sl["O"],C=sl["C"],E=sl["E"],A=sl["A"],N=sl["N"])
        st.markdown("**Emotional Memories**")
        for m in sl.get("strongest_memories",[]):
            st.markdown(f'<div class="mt">💭 {m}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MEMORY PALACE
# ══════════════════════════════════════════════════════════════════════════════
def _tab_memory():
    fs=ss.memory.get_full_status(); es=fs.get("episodic_stats",{})
    left,right=st.columns(2,gap="medium")
    with left:
        st.markdown('<div class="ph">📖 EPISODIC MEMORY</div>', unsafe_allow_html=True)
        if es:
            m1,m2,m3,m4=st.columns(4)
            m1.metric("Stored",es.get("total_stored",0))
            m2.metric("Success",f'{es.get("success_rate",0)*100:.1f}%')
            m3.metric("Landmarks",es.get("landmarks",0))
            m4.metric("Max Level",es.get("max_level_reached",1))
        rec=fs.get("episodic_recent",[])
        if rec:
            df=pd.DataFrame(rec)[["episode_id","curriculum_level","total_reward","success","total_steps","efficiency","maze_alg"]]
            df.columns=["EP#","LVL","REWARD","WIN","STEPS","EFFIC","ALG"]
            df["WIN"]=df["WIN"].map({True:"✅",False:"❌"})
            st.dataframe(df.tail(10),use_container_width=True,hide_index=True)
        st.markdown('<div class="ph">🌟 LANDMARKS</div>', unsafe_allow_html=True)
        for lm in fs.get("landmark_episodes",[])[:4]:
            cls="ep-success" if lm["success"] else "ep-fail"
            st.markdown(f'<div class="mt {cls}">'
                        f'<b>EP#{lm["episode_id"]}</b> L{lm["curriculum_level"]} {lm["maze_alg"]}&nbsp;'
                        f'R:<b>{lm["total_reward"]:.1f}</b> eff:{lm["efficiency"]:.2%} '
                        f'{"✅" if lm["success"] else "❌"}</div>',
                        unsafe_allow_html=True)
    with right:
        st.markdown('<div class="ph">🧬 SEMANTIC MEMORY</div>', unsafe_allow_html=True)
        facts=fs.get("semantic_facts",[])
        if facts:
            fdf=pd.DataFrame(facts)[["key","value","confidence","source"]].head(14)
            fdf.columns=["FACT","VALUE","CONF","SOURCE"]
            st.dataframe(fdf,use_container_width=True,hide_index=True)
        else: st.info("Semantic memory empty. Play more episodes.")
        st.markdown('<div class="ph">💡 INSIGHTS</div>', unsafe_allow_html=True)
        for ins in fs.get("insights",[]): st.markdown(f"• {ins}")
        st.markdown('<div class="ph">💾 STORAGE</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="fc">'
                    f'Path: <code>{fs["save_path"]}</code><br>'
                    f'Size: <code>{fs["save_size_kb"]:.1f} KB</code>&nbsp;'
                    f'Episodes: <code>{fs["total_episodes"]}</code>&nbsp;'
                    f'Loaded: <code>{"YES" if fs["loaded_from_disk"] else "NO"}</code></div>',
                    unsafe_allow_html=True)
        st.text(ss.analytics.get_session_report())
        zb=_export_zip()
        if zb: st.download_button("⬇️ Full ZIP Export",zb,f"alive_ep{ss.episode_count}.zip","application/zip",use_container_width=True)
        if st.button("🗑 Clear Episodic Memory",use_container_width=True):
            ss.memory.episodic.episodes.clear(); ss.memory.semantic.facts.clear()
            st.toast("Memory cleared."); st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — BRAIN AUTOPSY
# ══════════════════════════════════════════════════════════════════════════════
def _tab_brain():
    bs=ss.brain.get_stats(); bn=ss.brain.online_net
    total_p=sum(np.prod(getattr(bn,p).shape) for p in
                ["W1","b1","W2","b2","W3","b3","W_val","b_val","W_adv","b_adv"])
    left,right=st.columns([3,2],gap="large")
    with left:
        st.markdown('<div class="ph">🏗️ NETWORK ARCHITECTURE</div>', unsafe_allow_html=True)
        st.code(f"""Input  [{ss.brain.state_size}]  — 17-dim state encoding
   ↓  Leaky ReLU (α=0.01)
H1    [{bn.W2.shape[0]}]  — He init, Adam (β₁=0.9,β₂=0.999)
   ↓  Leaky ReLU
H2    [{bn.W3.shape[0]}]  — gradient clip ±10
   ↓  Leaky ReLU
H3    [{bn.W_val.shape[0]}]  — dual heads split
   ↓ Dueling
V(s)  [1]    A(s,a)  [{ss.brain.action_size}]
   ↓
Q(s,a) = V(s) + A(s,a) − mean(A)  [{ss.brain.action_size} actions]

Parameters  : {total_p:,}
Optimizer   : Adam  lr={ss.brain.learning_rate:.5f}
Soft-update : τ={ss.brain.tau}
PER         : α=0.6, β→1.0 ({ss.brain.memory.beta:.3f} now)
N-Step      : {ss.brain.n_step.n}
ICM β       : {ss.brain.curiosity.beta}""", language=None)

        st.markdown('<div class="ph">🔬 WEIGHT SAMPLE (H1 col 0-7)</div>', unsafe_allow_html=True)
        w=bn.W1[:,0:min(8,bn.W1.shape[1])].flatten()[:24]
        st.bar_chart(pd.DataFrame({"W1 weights":w}),height=130)

        st.markdown('<div class="ph">📊 TRAINING DIAGNOSTICS</div>', unsafe_allow_html=True)
        checks=[
            ("Buffer filled", len(ss.brain.memory)>=ss.brain.batch_size, f'{len(ss.brain.memory):,}/{ss.brain.batch_size}'),
            ("Learning started", ss.brain.train_step>0, f'{ss.brain.train_step:,} steps'),
            ("Epsilon < 0.5",   ss.brain.epsilon<0.5, f'ε={ss.brain.epsilon:.4f}'),
            ("Positive reward", ss.brain.avg_reward>0, f'{ss.brain.avg_reward:+.3f}'),
            ("Not regressing",  ss.analytics.tracker.convergence.state!="regressing",
             ss.analytics.tracker.convergence.state),
            ("Level > 1",       ss.brain.curriculum.level>1, f'L{ss.brain.curriculum.level}'),
            ("LR not bottomed", ss.brain.learning_rate>1e-5, f'{ss.brain.learning_rate:.6f}'),
            ("Success > 0",     ss.analytics.tracker.success_rate>0,
             f'{ss.analytics.tracker.success_rate*100:.1f}%'),
        ]
        dc1,dc2=st.columns(2)
        for i,(name,ok,detail) in enumerate(checks):
            col=dc1 if i%2==0 else dc2
            ico="✅" if ok else "⚠️"; clr="#22c55e" if ok else "#f97316"
            col.markdown(f'<div style="display:flex;align-items:center;gap:7px;margin:3px 0;font-size:.76rem;">'
                         f'{ico} <span style="color:#c9d1d9;">{name}</span>'
                         f'<span style="margin-left:auto;color:{clr};font-size:.67rem;">{detail}</span></div>',
                         unsafe_allow_html=True)
    with right:
        st.markdown('<div class="ph">🎛️ LIVE COUNTERS</div>', unsafe_allow_html=True)
        st.metric("Train Steps",    f'{bs["train_step"]:,}')
        st.metric("Buffer Fill",    f'{len(ss.brain.memory):,}/{ss.brain.memory.capacity:,}')
        st.metric("Buffer β",       f'{ss.brain.memory.beta:.3f}')
        st.metric("Unique States",  bs["unique_states"])
        st.metric("ICM Coverage",   bs["unique_states"])
        st.metric("LR Reductions",  ss.brain.lr_sched.reductions)
        st.markdown("**Buffer fill**")
        st.markdown(pb(len(ss.brain.memory)/ss.brain.memory.capacity,"#a855f7"), unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="ph">📡 CAPABILITY RADAR</div>', unsafe_allow_html=True)
        t=ss.analytics.tracker; cur_lvl=ss.brain.curriculum.level
        H,W=ss.env.maze.shape
        cov=ss.analytics.heatmap.coverage(H,W,ss.env.maze)
        sr=t.success_rate; opt=t.avg_optimality
        conv_sc={"warming_up":.2,"rapid_learning":.8,"fine_tuning":.7,
                 "converged":1.,"plateau":.4,"regressing":.0}.get(t.convergence.state,.2)
        lvl_sc=(cur_lvl-1)/9.; exp_sc=min(cov,1.)
        vals=[sr,opt,exp_sc,conv_sc,lvl_sc]
        _show("cap_radar",_build_radar_cap,vals_pct=vals)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — EPISODE TIMELINE
# ══════════════════════════════════════════════════════════════════════════════
def _tab_timeline():
    eps=ss.memory.episodic.episodes[-100:] if ss.memory.episodic.episodes else []
    st.markdown("### 📅 Episode Timeline (last 100)")
    if not eps: st.info("No episodes recorded yet."); return
    df=pd.DataFrame([e.to_dict() for e in eps])
    # colour success
    s1,s2,s3,s4=st.columns(4)
    s1.metric("Win rate",f'{df["success"].mean()*100:.1f}%')
    s2.metric("Avg reward",f'{df["total_reward"].mean():.2f}')
    s3.metric("Avg efficiency",f'{df["efficiency"].mean():.2%}')
    s4.metric("Levels seen",f'{df["curriculum_level"].nunique()}')
    # episode table
    # episode table
    disp=df[["episode_id","curriculum_level","total_reward","success","total_steps","efficiency","maze_alg","tags"]].copy()
    disp["success"]=disp["success"].map({True:"✅",False:"❌"})
    disp["efficiency"]=disp["efficiency"].apply(lambda x:f"{x:.2%}")
    disp.columns=["EP#","LVL","REWARD","WIN","STEPS","EFFIC","ALG","TAGS"]
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.markdown("---")
    # timeline charts (cached)
    xs=list(df["episode_id"])
    _show("tl_reward",_build_lines,
          xs=xs,
          traces=[{"y":list(df["total_reward"]),"name":"Reward","c":"#00f5ff","w":1.5}],
          title="Reward Timeline",height=200)
    c1,c2=st.columns(2)
    with c1:
        _show("tl_steps",_build_lines,
              xs=xs,
              traces=[{"y":list(df["total_steps"]),"name":"Steps","c":"#58a6ff","w":1.2}],
              title="Steps per Episode",height=180)
    with c2:
        _show("tl_eff",_build_lines,
              xs=xs,
              traces=[{"y":list(df["efficiency"]),"name":"Efficiency","c":"#22c55e","w":1.2}],
              title="Path Efficiency vs A*",height=180)
    st.markdown("---")
    # Level distribution
    lvl_counts=df["curriculum_level"].value_counts().sort_index()
    st.markdown("**Level Distribution**")
    st.bar_chart(lvl_counts,height=140)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
def _tab_benchmark():
    st.markdown("### 🏆 Benchmark & Capability Analysis")
    t=ss.analytics.tracker; cur_lvl=ss.brain.curriculum.level
    H,W=ss.env.maze.shape
    cov=ss.analytics.heatmap.coverage(H,W,ss.env.maze)
    sr=t.success_rate; opt=t.avg_optimality; conv_st=t.convergence.state
    conv_sc={"warming_up":.2,"rapid_learning":.8,"fine_tuning":.7,
             "converged":1.,"plateau":.4,"regressing":.0}.get(conv_st,.2)
    lvl_sc=(cur_lvl-1)/9.
    success_score=sr*40; opt_score=opt*25; explore_score=min(cov,1.)*15
    conv_score=conv_sc*10; level_score=lvl_sc*10
    total=success_score+opt_score+explore_score+conv_score+level_score
    cap_color="#22c55e" if total>70 else ("#f97316" if total>40 else "#ef4444")

    # Big score display
    st.markdown(f'<div style="text-align:center;padding:1.2rem 0;">'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:3.5rem;'
                f'font-weight:700;color:{cap_color};line-height:1;">{total:.1f}</div>'
                f'<div style="color:#8b949e;font-size:.8rem;letter-spacing:.15em;">CAPABILITY SCORE / 100</div></div>',
                unsafe_allow_html=True)

    # Score breakdown
    components=[
        ("Success Rate (×40)",  success_score,  40, "#22c55e", f'{sr*100:.1f}%'),
        ("Path Efficiency (×25)",opt_score,     25, "#00f5ff", f'{opt*100:.1f}%'),
        ("Exploration (×15)",   explore_score,  15, "#a855f7", f'{cov*100:.1f}%'),
        ("Convergence (×10)",   conv_score,     10, "#58a6ff", conv_st),
        ("Curriculum (×10)",    level_score,    10, "#f97316", f'L{cur_lvl}/10'),
    ]
    for name,score,mx,clr,detail in components:
        frac=score/mx if mx>0 else 0
        st.markdown(f'<div style="margin:6px 0;">'
                    f'<div style="display:flex;justify-content:space-between;font-size:.76rem;margin-bottom:2px;">'
                    f'<span style="color:#c9d1d9;">{name}</span>'
                    f'<span style="color:{clr};">{score:.1f} / {mx} &nbsp; <i>{detail}</i></span></div>'
                    f'{pb(frac,clr)}</div>', unsafe_allow_html=True)
    st.markdown("---")
    _show("bm_radar",_build_radar_cap,vals_pct=[sr,opt,cov,conv_sc,lvl_sc])
    st.markdown("---")
    # Environment diagnostics
    st.markdown('<div class="ph">🌐 ENVIRONMENT DIAGNOSTICS</div>', unsafe_allow_html=True)
    es=ss.env.get_stats(); rd=ss.env.get_render_data()
    e1,e2,e3,e4=st.columns(4)
    e1.metric("Maze Size", es["maze_size"])
    e2.metric("Algorithm",rd["algorithm"].upper())
    e3.metric("Passable Cells",int((ss.env.maze==0).sum()))
    e4.metric("A* Optimal",es["astar_optimal"])
    e5,e6,e7,e8=st.columns(4)
    e5.metric("Traps",es["traps"]); e6.metric("Portals",len(rd["portals_a"]))
    e7.metric("Fog",  "On" if rd["use_fog"] else "Off"); e8.metric("Coverage",f'{cov*100:.1f}%')
    st.markdown("---")
    st.markdown('<div class="ph">📤 SESSION EXPORT</div>', unsafe_allow_html=True)
    xc1,xc2=st.columns(2)
    with xc1:
        st.markdown("**Session Report**")
        st.code(ss.analytics.get_session_report(), language=None)
    with xc2:
        st.markdown("**JSON Preview**")
        j=ss.analytics.export_json()
        st.code(j[:1500]+("\n...[truncated]" if len(j)>1500 else ""), language="json")
    zb=_export_zip()
    if zb: st.download_button("⬇️ Full ZIP Checkpoint",zb,
        f"ALIVE_ep{ss.episode_count}_L{ss.brain.curriculum.level}.zip","application/zip",use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def _main():
    # ── Global header ────────────────────────────────────────────────
    h1,h2,h3=st.columns([4,3,2])
    with h1:
        st.markdown('<div class="nt">🧬 A.L.I.V.E. NEXUS</div>'
                    '<div style="font-size:.65rem;color:#6e7681;letter-spacing:.06em;margin-top:1px;">'
                    'Adaptive Learning Intelligence &amp; Virtual Evolution — v3.1</div>',
                    unsafe_allow_html=True)
    with h2:
        lv=ss.analytics.get_live_stats(); sl=ss.soul.get_status()
        st.markdown(f'<div style="text-align:center;font-size:.71rem;line-height:2.1;padding-top:7px;">'
                    f'<span class="bd bc">EP {ss.episode_count}</span>'
                    f'<span class="bd bg">✓ {lv["success_rate"]:.1f}%</span>'
                    f'<span class="bd bp">L{ss.brain.curriculum.level}</span>'
                    f'<span class="bd bo">ε {ss.brain.epsilon:.4f}</span>'
                    f'<span style="font-size:1.1rem;vertical-align:middle;margin-left:4px;">{sl["mood_emoji"]}</span>'
                    f'</div>', unsafe_allow_html=True)
    with h3:
        lbl="⏸ PAUSE" if ss.auto_mode else "▶ AUTO RUN"
        if st.button(lbl, use_container_width=True, key="hdr_toggle"):
            ss.auto_mode = not ss.auto_mode
        if st.button("⏭ STEP ×1", use_container_width=True, key="hdr_step"):
            process_step(); st.rerun()
    st.markdown("---")

    _sidebar()

    tabs=st.tabs(["🗺️ Mission Control","📊 Analytics Lab","🧠 Soul Matrix",
                  "🗄️ Memory Palace","🔬 Brain Autopsy","📅 Timeline","🏆 Benchmark"])
    with tabs[0]: _tab_mission()
    with tabs[1]: _tab_analytics()
    with tabs[2]: _tab_soul()
    with tabs[3]: _tab_memory()
    with tabs[4]: _tab_brain()
    with tabs[5]: _tab_timeline()
    with tabs[6]: _tab_benchmark()

    # ── Auto-run: execute AFTER tabs so UI renders first ─────────────
    if ss.auto_mode:
        spf=ss.config.get("steps_per_frame",1)
        for _ in range(spf): process_step()
        delay=ss.config.get("sim_speed",0.03)
        if delay>0: time.sleep(delay)
        st.rerun()

_main()
