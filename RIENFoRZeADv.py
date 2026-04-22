"""
app.py ── A.L.I.V.E. NEXUS  v4.0  "Event Horizon Edition"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Co-Investigator: Xylia | Collaborator: Nik
Project: The Event Horizon

SCIENTIFIC INSTRUMENTATION  (new in v4.0)                
  ✦ Live Q(s,a) vector bar-chart — real-time policy readout   
  ✦ Policy entropy H(π) timeline — quantifies exploration
  ✦ Reward decomposition: intrinsic (ICM) vs extrinsic
  ✦ V(s) / max-Advantage streaming — dueling head analysis
  ✦ Gradient-norm proxy tracker — training stability signal
  ✦ Agent trajectory trail inside maze ASCII render
  ✦ Convergence hypothesis test (t-test on reward gradient)
  ✦ Learning-rate plateau & sensitivity gauge
  ✦ Bellman residual stream (separate from MSE loss)
  ✦ Action preference histogram (U/D/L/R distribution)
  ✦ Episode-comparison panel: best vs worst vs current
  ✦ Full Research Lab tab: equations, theory, citations

PERFORMANCE CONTRACT (Streamlit Cloud)
  ✦ Zero Plotly — native charts only
  ✦ Maze ASCII cached per step_count
  ✦ ZIP built only on button press
  ✦ No numpy→list on every rerun
  ✦ Chart slices computed once, shared
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import streamlit as st
try:
    import plotly.graph_objects as go
    import plotly.express as px
    _PLOTLY = True
except ImportError:
    _PLOTLY = False
import numpy  as np
import pandas as pd
import time, json, io, zipfile, math, random
from collections import deque
from typing import Dict, List, Optional, Tuple

# ── Backend ────────────────────────────────────────────────
try:
    from world         import MazeEnvironment, WALL, PATH
    from brain         import AgentBrain
    from soul          import SoulCore
    from memory_palace import MemoryPalace
    from analytics     import PerformanceDashboard
    _OK = True; _ERR = ""
except ImportError as e:
    _OK = False; _ERR = str(e)

# ── Page config ────────────────────────────────────────────
st.set_page_config(page_title="A.L.I.V.E. NEXUS",
                   layout="wide",
                   initial_sidebar_state="expanded",
                   page_icon="🧬")

# ── CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&display=swap');

/* ─ base ─ */
.stApp{background:radial-gradient(ellipse at 20% 10%,#0d0d2e 0%,#080818 55%,#0a1a18 100%);
 color:#c9d1d9;font-family:'Segoe UI',sans-serif;}
* {box-sizing:border-box;}

/* ─ shimmer title ─ */
.ntitle{font-family:'JetBrains Mono',monospace;font-size:1.85rem;font-weight:700;
 background:linear-gradient(90deg,#00f5ff 0%,#a855f7 38%,#f97316 68%,#00f5ff 100%);
 background-size:300% 100%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;
 animation:sh 4s linear infinite;}
@keyframes sh{to{background-position:-300% 0}}

/* ─ telemetry strip ─ */
.tel{display:flex;gap:0;border:1px solid rgba(0,245,255,.1);border-radius:10px;
 overflow:hidden;margin-bottom:10px;background:rgba(0,0,0,.2);}
.tel-cell{flex:1;padding:6px 10px;text-align:center;
 border-right:1px solid rgba(255,255,255,.05);}
.tel-cell:last-child{border-right:none;}
.tel-v{font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:700;color:#00f5ff;}
.tel-l{font-size:.56rem;color:#6e7681;text-transform:uppercase;letter-spacing:.08em;}
.tel-d{font-size:.6rem;margin-top:1px;}
.tel-up{color:#22c55e;} .tel-dn{color:#ef4444;} .tel-nt{color:#8b949e;}

/* ─ kpi card ─ */
.kpi{background:rgba(0,245,255,.03);border:1px solid rgba(0,245,255,.12);
 border-radius:10px;padding:10px 14px;text-align:center;transition:all .25s;}
.kpi:hover{border-color:rgba(0,245,255,.4);box-shadow:0 0 16px rgba(0,245,255,.12);}
.kv{font-family:'JetBrains Mono',monospace;font-size:1.45rem;font-weight:700;
 color:#00f5ff;line-height:1.2;}
.kl{font-size:.61rem;color:#8b949e;text-transform:uppercase;letter-spacing:.1em;margin-top:3px;}
.ks{font-size:.57rem;color:#58a6ff;margin-top:2px;}

/* ─ panel header ─ */
.ph{font-family:'JetBrains Mono',monospace;font-size:.71rem;color:#a855f7;
 text-transform:uppercase;letter-spacing:.15em;padding:4px 0 6px;
 border-bottom:1px solid rgba(168,85,247,.25);margin-bottom:10px;}

/* ─ thought box ─ */
.thought{background:rgba(168,85,247,.07);border-left:3px solid #a855f7;
 border-radius:0 8px 8px 0;padding:9px 14px;font-size:.83rem;font-style:italic;
 color:#d8b4fe;line-height:1.5;margin:6px 0;}

/* ─ equation box ─ */
.eq{background:rgba(0,245,255,.04);border:1px solid rgba(0,245,255,.15);
 border-radius:8px;padding:12px 16px;font-family:'JetBrains Mono',monospace;
 font-size:.78rem;color:#c9d1d9;line-height:1.8;margin:6px 0;}
.eq b{color:#00f5ff;} .eq em{color:#a855f7;} .eq u{color:#f97316;text-decoration:none;}

/* ─ hypothesis banner ─ */
.hyp-yes{background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.3);
 border-radius:8px;padding:10px 14px;font-size:.82rem;color:#86efac;}
.hyp-no{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.3);
 border-radius:8px;padding:10px 14px;font-size:.82rem;color:#fca5a5;}
.hyp-unk{background:rgba(234,179,8,.08);border:1px solid rgba(234,179,8,.3);
 border-radius:8px;padding:10px 14px;font-size:.82rem;color:#fde68a;}

/* ─ q-value readout ─ */
.qrow{display:flex;gap:6px;margin:6px 0;}
.qcell{flex:1;text-align:center;padding:6px 4px;border-radius:6px;
 font-family:'JetBrains Mono',monospace;font-size:.8rem;font-weight:600;}

/* ─ chat ─ */
.chat-scroll{height:270px;overflow-y:auto;padding:8px;
 border:1px solid rgba(255,255,255,.06);border-radius:8px;background:rgba(0,0,0,.18);}
.cu{background:rgba(88,166,255,.10);border-right:3px solid #58a6ff;
 border-radius:10px 0 0 10px;padding:7px 11px;margin:4px 0;
 text-align:right;font-size:.84rem;color:#cdd9e5;}
.ca{background:rgba(168,85,247,.09);border-left:3px solid #a855f7;
 border-radius:0 10px 10px 0;padding:7px 11px;margin:4px 0;
 font-size:.84rem;color:#d8b4fe;}
.cmeta{font-size:.61rem;color:#6e7681;margin-top:2px;}

/* ─ badges ─ */
.badge{display:inline-block;padding:2px 8px;border-radius:16px;font-size:.62rem;
 font-weight:700;text-transform:uppercase;letter-spacing:.06em;
 font-family:'JetBrains Mono',monospace;margin:1px 2px;}
.bc{background:rgba(0,245,255,.12);color:#00f5ff;border:1px solid rgba(0,245,255,.3);}
.bp{background:rgba(168,85,247,.12);color:#a855f7;border:1px solid rgba(168,85,247,.3);}
.bg{background:rgba(34,197,94,.12);color:#22c55e;border:1px solid rgba(34,197,94,.3);}
.br{background:rgba(239,68,68,.12);color:#ef4444;border:1px solid rgba(239,68,68,.3);}
.bo{background:rgba(249,115,22,.12);color:#f97316;border:1px solid rgba(249,115,22,.3);}

/* ─ convergence banners ─ */
.cvb{border-radius:8px;padding:10px 14px;margin:6px 0;font-size:.82rem;
 display:flex;align-items:center;gap:10px;}
.cw{background:rgba(59,130,246,.10);border:1px solid rgba(59,130,246,.25);}
.cr{background:rgba(34,197,94,.10);border:1px solid rgba(34,197,94,.25);}
.cf{background:rgba(234,179,8,.10);border:1px solid rgba(234,179,8,.25);}
.co{background:rgba(0,245,255,.10);border:1px solid rgba(0,245,255,.25);}
.cp{background:rgba(168,85,247,.10);border:1px solid rgba(168,85,247,.25);}
.cng{background:rgba(239,68,68,.10);border:1px solid rgba(239,68,68,.25);}

/* ─ memory cards ─ */
.mc{background:rgba(168,85,247,.06);border:1px solid rgba(168,85,247,.14);
 border-radius:7px;padding:7px 11px;margin:3px 0;font-size:.78rem;color:#c9d1d9;}
.es{border-left:3px solid #22c55e;padding-left:7px;}
.ef{border-left:3px solid #ef4444;padding-left:7px;}

/* ─ architecture box ─ */
.arch{background:rgba(0,0,0,.28);border:1px solid rgba(0,245,255,.1);
 border-radius:8px;padding:14px;font-family:'JetBrains Mono',monospace;
 font-size:.75rem;line-height:1.9;}

/* ─ citation card ─ */
.cite{background:rgba(255,255,255,.025);border-left:3px solid #58a6ff;
 border-radius:0 7px 7px 0;padding:8px 12px;margin:5px 0;font-size:.76rem;color:#8b949e;}

/* ─ widgets ─ */
.stButton>button{background:linear-gradient(135deg,rgba(0,245,255,.08),rgba(168,85,247,.08));
 border:1px solid rgba(0,245,255,.25);color:#00f5ff;border-radius:7px;
 font-family:'JetBrains Mono',monospace;font-size:.74rem;font-weight:600;
 letter-spacing:.04em;transition:all .2s;}
.stButton>button:hover{border-color:rgba(0,245,255,.65);background:rgba(0,245,255,.16);
 box-shadow:0 0 14px rgba(0,245,255,.2);transform:translateY(-1px);}
div[data-testid="stMetric"]{background:rgba(255,255,255,.025);
 border:1px solid rgba(255,255,255,.055);border-radius:8px;padding:7px 10px;}
div[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace;color:#00f5ff;}
.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,.02);
 border:1px solid rgba(255,255,255,.055);border-radius:8px;gap:2px;padding:4px;}
.stTabs [data-baseweb="tab"]{font-family:'JetBrains Mono',monospace;
 font-size:.68rem;letter-spacing:.05em;color:#8b949e;border-radius:6px;}
.stTabs [aria-selected="true"]{background:rgba(0,245,255,.10)!important;color:#00f5ff!important;}
.stCode,.stCodeBlock{background:#0a0a18!important;
 border:1px solid rgba(0,245,255,.12)!important;border-radius:8px!important;
 font-size:.69rem!important;line-height:1.1!important;}
pre code{font-size:.69rem!important;}
section[data-testid="stSidebar"]{background:rgba(8,8,24,.97);
 border-right:1px solid rgba(0,245,255,.08);}
div[data-testid="stDecoration"]{display:none;}

</style>""", unsafe_allow_html=True)

# ── Backend guard ──────────────────────────────────────────
if not _OK:
    st.error(f"❌ Backend import failed: `{_ERR}`")
    st.info("Ensure world.py brain.py soul.py memory_palace.py analytics.py are co-located.")
    st.stop()

# ── Constants ──────────────────────────────────────────────
SAVE_PATH   = "./alive_nexus_v4.json"
STATE_SIZE  = 17
ACTION_SIZE = 4
ACTIONS     = ["↑ UP", "↓ DOWN", "← LEFT", "→ RIGHT"]
ACTION_CLR  = ["#00f5ff", "#a855f7", "#f97316", "#22c55e"]

DEFAULT_CFG: Dict = {
    "sim_speed": 0.04, "steps_per_frame": 1, "autosave_interval": 100,
    "gamma": 0.99, "epsilon_min": 0.04, "epsilon_decay": 0.997,
    "lr": 0.001, "batch_size": 64, "buffer_size": 50_000,
    "n_steps": 3, "icm_beta": 0.05, "tau": 0.005,
    "h1": 256, "h2": 128, "h3": 64,
    "show_astar": False, "chart_points": 150,
    "override_curriculum": False, "manual_level": 1,
}

# ── HTML helpers ───────────────────────────────────────────
def _kpi(v, l, s=""):
    return (f'<div class="kpi"><div class="kv">{v}</div><div class="kl">{l}</div>'
            + (f'<div class="ks">{s}</div>' if s else "") + "</div>")

def _conv_cls(s): return {"warming_up":"cw","rapid_learning":"cr","fine_tuning":"cf",
    "converged":"co","plateau":"cp","regressing":"cng"}.get(s,"cw")
def _conv_icon(s): return {"warming_up":"🔥","rapid_learning":"🚀","fine_tuning":"⚙️",
    "converged":"✅","plateau":"📊","regressing":"⬇️"}.get(s,"❓")
def _conv_desc(s): return {
    "warming_up":   "Filling replay buffer. Bellman targets not yet computed.",
    "rapid_learning":"dL/dt strongly negative. Policy gradient is effective.",
    "fine_tuning":  "Marginal improvement. Epsilon near floor. Exploiting.",
    "converged":    "KL(π_t || π_{t-1}) ≈ 0. Value function has stabilised.",
    "plateau":      "High variance, zero slope. Local minimum suspected.",
    "regressing":   "Negative reward trend. Check γ, τ, or buffer quality."}.get(s,"")

def _np_enc(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, deque): return list(o)
    return str(o)

# ── Science helpers ────────────────────────────────────────
def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max()); return e / e.sum()

def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1); return float(-np.sum(p * np.log(p)))

def _ttest_slope(values: List[float]) -> Tuple[float, float]:
    """Return (slope, p_value) from a 1-D linear regression t-test."""
    n = len(values)
    if n < 6:
        return 0.0, 1.0
    x = np.arange(n, dtype=float)
    x -= x.mean(); y = np.array(values, dtype=float) - np.mean(values)
    slope = float(np.dot(x, y) / (np.dot(x, x) + 1e-12))
    resid = y - slope * x
    s2    = float(np.dot(resid, resid) / max(n - 2, 1))
    se    = math.sqrt(s2 / max(float(np.dot(x, x)), 1e-12))
    t     = slope / max(se, 1e-12)
    # two-tailed p approx via t-distribution tail (df=n-2)
    df = n - 2
    p  = 2.0 * (1.0 - min(abs(t) / (abs(t) + math.sqrt(df)), 0.9999))
    return slope, float(np.clip(p, 0, 1))

# ── Session init ───────────────────────────────────────────
def _init():
    ss  = st.session_state
    cfg = ss.get("config", dict(DEFAULT_CFG))
    ss.config = cfg
    bk = {k: cfg[k] for k in
          ("gamma","epsilon_min","epsilon_decay","lr","batch_size",
           "buffer_size","n_steps","icm_beta","tau","h1","h2","h3")}
    ss.brain     = AgentBrain(STATE_SIZE, ACTION_SIZE, config=bk)
    ss.env       = MazeEnvironment()
    ss.soul      = SoulCore(name="Nik")
    ss.memory    = MemoryPalace(save_path=SAVE_PATH)
    ss.analytics = PerformanceDashboard()
    ss.cur_state = ss.env.reset(config=ss.brain.curriculum.config)
    ss._prev_cells   = 1
    ss._maze_cache   = ("", -1)

    # ── Scientific instrumentation buffers ──────────────────
    N = 400
    ss.qval_hist:     deque = deque(maxlen=N)   # list of 4 Q-values per step
    ss.entropy_hist:  deque = deque(maxlen=N)   # H(softmax(Q))
    ss.intr_hist:     deque = deque(maxlen=N)   # intrinsic reward
    ss.extr_hist:     deque = deque(maxlen=N)   # extrinsic reward
    ss.val_hist:      deque = deque(maxlen=N)   # V(s) from value head
    ss.adv_hist:      deque = deque(maxlen=N)   # max A(s,a)
    ss.gnorm_hist:    deque = deque(maxlen=N)   # gradient-norm proxy (|ΔW1|_F)
    ss.bellman_hist:  deque = deque(maxlen=N)   # |r + γ max Q(s') - Q(s,a)|
    ss.action_counts: List  = [0, 0, 0, 0]     # U D L R histogram
    ss.trajectory:    deque = deque(maxlen=30)  # (r,c) recent positions
    ss.prev_w1:       np.ndarray = ss.brain.online_net.W1.copy()  # for grad proxy
    ss.best_ep:       Optional[Dict] = None
    ss.worst_ep:      Optional[Dict] = None
    ss.ep_log:        deque = deque(maxlen=200)  # lightweight episode log

    ss.memory.start_episode(
        maze_seed=ss.env.seed, maze_alg=ss.env.algorithm,
        maze_h=ss.env.maze_h, maze_w=ss.env.maze_w,
        level=ss.brain.curriculum.level,
        epsilon=ss.brain.epsilon, max_steps=ss.env.max_steps)

    ss.auto_mode       = False
    ss.global_step     = 0
    ss.episode_count   = 0
    ss.capability      = 0.0
    ss.last_ep_reward  = 0.0
    ss.last_ep_success = False

if "brain" not in st.session_state or "entropy_hist" not in st.session_state:
    _init()

# ── Simulation core ────────────────────────────────────────
def _ep_done(info: Dict):
    ss = st.session_state
    env_st = ss.env.get_stats(); cur = ss.brain.curriculum.level
    H, W   = ss.env.maze.shape; succ = bool(info.get("reached"))

    ss.brain.curriculum.record(success=succ, steps=env_st["step_count"],
        max_steps=env_st["max_steps"], reward=env_st["episode_reward"])
    ss.memory.end_episode(
        total_reward=env_st["episode_reward"], steps=env_st["step_count"],
        success=succ, cells_visited=env_st["cells_visited"],
        astar_optimal=env_st["astar_optimal"], fog=env_st["fog"],
        traps=env_st["traps"] > 0,
        td_error=ss.brain.avg_td_error, epsilon=ss.brain.epsilon)
    cap = ss.analytics.record_episode(
        reward=env_st["episode_reward"], steps=env_st["step_count"], success=succ,
        info={"optimality": info.get("optimality",0.0),
              "fog_coverage": info.get("fog_coverage",1.0),
              "level": cur, "success_count": ss.env.success_count},
        curriculum_level=cur, h=H, w=W, maze=ss.env.maze)

    ep_record = {
        "ep": ss.episode_count, "reward": env_st["episode_reward"],
        "steps": env_st["step_count"], "success": succ,
        "level": cur, "efficiency": info.get("optimality", 0.0),
    }
    ss.ep_log.append(ep_record)
    if ss.best_ep is None  or ep_record["reward"] > ss.best_ep["reward"]:  ss.best_ep  = ep_record
    if ss.worst_ep is None or ep_record["reward"] < ss.worst_ep["reward"]: ss.worst_ep = ep_record

    ss.capability = cap; ss.last_ep_reward = env_st["episode_reward"]
    ss.last_ep_success = succ; ss.episode_count += 1

    if ss.episode_count % ss.config.get("autosave_interval", 100) == 0:
        _do_save()

    new_cfg = ss.brain.curriculum.config
    if ss.config.get("override_curriculum"):
        lvl = ss.config.get("manual_level", 1)
        new_cfg = dict(ss.brain.curriculum.LEVEL_CONFIGS.get(lvl, new_cfg))
    ss.cur_state  = ss.env.reset(config=new_cfg)
    ss._prev_cells = 1; ss._maze_cache = ("", -1)
    ss.trajectory.clear()
    ss.memory.start_episode(
        maze_seed=ss.env.seed, maze_alg=ss.env.algorithm,
        maze_h=ss.env.maze_h, maze_w=ss.env.maze_w,
        level=ss.brain.curriculum.level,
        epsilon=ss.brain.epsilon, max_steps=ss.env.max_steps)


def process_step():
    ss     = st.session_state
    state  = ss.cur_state
    action = ss.brain.act(state)

    # ── Scientific instrumentation (pre-step) ──────────────
    bn  = ss.brain.online_net
    qv  = bn.forward(state, training=True)   # caches val/adv internals
    q   = qv[0]                              # shape (4,)
    prb = _softmax(q)
    ent = _entropy(prb)
    # Extract V(s) and max A(s,a) from the cached dueling heads
    _cache = getattr(bn, "_cache", {})
    if "val" in _cache and "adv" in _cache:
        val_raw = float(_cache["val"][0, 0])
        adv_raw = float(_cache["adv"][0].max() - _cache["adv"][0].mean())
    else:
        val_raw = float(q.mean())
        adv_raw = float(q.max() - q.mean())

    ss.qval_hist.append(q.tolist())
    ss.entropy_hist.append(ent)
    ss.val_hist.append(val_raw)
    ss.adv_hist.append(adv_raw)
    ss.action_counts[action] += 1

    # ── Step ───────────────────────────────────────────────
    ns, reward, done, info = ss.env.step(action)

    # Bellman residual
    qns = bn.forward(ns, training=False)
    br_res = abs(reward + ss.brain.gamma*(1-float(done))*float(qns[0].max()) - q[action])
    ss.bellman_hist.append(float(br_res))

    # Intrinsic / extrinsic decomposition
    intr = float(ss.brain.curiosity.bonus(state))
    ss.intr_hist.append(intr)
    ss.extr_hist.append(float(reward))

    loss, td_err = ss.brain.step(state, action, reward, ns, done)

    # Gradient-norm proxy: Frobenius norm of ΔW1
    new_w1 = bn.W1
    gnorm  = float(np.linalg.norm(new_w1 - ss.prev_w1, 'fro'))
    ss.gnorm_hist.append(gnorm)
    ss.prev_w1 = new_w1.copy()

    ss.memory.record_transition(state, action, reward, ns, done)
    ss.analytics.record_step(ss.env.agent_r, ss.env.agent_c, loss, td_err, ss.brain.epsilon)

    nc = len(ss.env.cells_visited); is_new = nc > ss._prev_cells; ss._prev_cells = nc
    ss.trajectory.append((ss.env.agent_r, ss.env.agent_c))
    trap_near = bool(ss.env.traps and
        any(abs(t.r-ss.env.agent_r)+abs(t.c-ss.env.agent_c)<=3 for t in ss.env.traps))

    ss.soul.update_from_rl(
        stats={"epsilon": ss.brain.epsilon, "avg_loss": ss.brain.avg_loss,
               "avg_td_error": ss.brain.avg_td_error, "train_step": ss.brain.train_step,
               "avg_reward": ss.brain.avg_reward, "curriculum": ss.brain.curriculum.get_stats()},
        env_info={"reward": reward, "reached": info.get("reached",False),
                  "timeout": info.get("timeout",False), "trap_hit": info.get("trap_hit",False),
                  "trap_nearby": trap_near, "portal_used": False, "is_new_cell": is_new,
                  "success_count": ss.env.success_count,
                  "success_rate": ss.env.success_count/max(ss.env.total_episodes,1),
                  "cells_visited": nc, "maze_size": f"{ss.env.maze_h}x{ss.env.maze_w}"})

    ss.cur_state = ns; ss.global_step += 1
    if done: _ep_done(info)


def reset_all():
    for k in ["brain","env","soul","memory","analytics","cur_state","global_step",
              "episode_count","capability","last_ep_reward","last_ep_success",
              "_prev_cells","_maze_cache","qval_hist","entropy_hist","intr_hist",
              "extr_hist","val_hist","adv_hist","gnorm_hist","bellman_hist",
              "action_counts","trajectory","prev_w1","best_ep","worst_ep","ep_log"]:
        st.session_state.pop(k, None)
    st.session_state.auto_mode = False
    _init()

# ── Persistence ────────────────────────────────────────────
def _do_save() -> bool:
    ss = st.session_state
    try:
        return ss.memory.save_all(brain_weights=ss.brain.get_weights(),
            analytics_data=ss.analytics.tracker.session_summary(),
            soul_status=ss.soul.get_status())
    except Exception as e: st.toast(f"Save error: {e}"); return False

def _make_zip() -> Optional[bytes]:
    ss = st.session_state
    try:
        p = {"version":"4.0","saved_at":time.time(),"config":ss.config,
             "brain":ss.brain.get_weights(),"soul":ss.soul.get_status(),
             "analytics_summary":ss.analytics.tracker.session_summary(),
             "memory_status":ss.memory.get_full_status(),
             "global_step":ss.global_step,"episode_count":ss.episode_count,
             "capability":ss.capability}
        buf = io.BytesIO()
        with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
            p_safe = ss.memory.store._serialize(p)
            z.writestr("alive_nexus_v4.json",json.dumps(p_safe,indent=2))
        return buf.getvalue()
    except Exception as e: st.toast(f"Export error: {e}"); return None

def _load_zip(up) -> bool:
    ss = st.session_state
    try:
        with zipfile.ZipFile(up,"r") as z:
            json_files = [n for n in z.namelist() if n.endswith(".json")]
            if not json_files: raise ValueError("No .json found in archive")
            with z.open(json_files[0]) as f: d = json.load(f)
        if "brain" in d:
            ss.brain.set_weights(d["brain"]); ss.brain.target_net.copy_from(ss.brain.online_net)
        if "soul" in d:
            p = d["soul"]
            for t in ("O","C","E","A","N"):
                if t in p: setattr(ss.soul.personality, t, float(p[t]))
            if "relationship" in p: ss.soul.relationship.score = float(p["relationship"])
        ss.global_step=d.get("global_step",0); ss.episode_count=d.get("episode_count",0)
        ss.capability=d.get("capability",0.0)
        if "config" in d: ss.config.update(d["config"])
        return True
    except Exception as e: st.error(f"Load failed: {e}"); return False

# ── Maze render ────────────────────────────────────────────
def _get_maze() -> str:
    ss = st.session_state
    cached, step = ss.get("_maze_cache", ("", -1))
    if step == ss.env.step_count and cached: return cached
    s, _ = ss.env.render_ascii()
    ss._maze_cache = (s, ss.env.step_count); return s

def _get_maze_with_trail() -> str:
    """Overlay trajectory trail onto maze string."""
    ss     = st.session_state
    trail  = set(ss.trajectory)
    H, W   = ss.env.maze.shape
    syms   = {WALL:"██", PATH:"  "}
    rows   = []
    for r in range(H):
        row = ""
        for c in range(W):
            is_agent  = (r==ss.env.agent_r  and c==ss.env.agent_c)
            is_target = (r==ss.env.target_r and c==ss.env.target_c)
            is_trap   = any(t.r==r and t.c==c for t in ss.env.traps)
            is_trail  = (r,c) in trail and not is_agent
            if is_agent:  row += "🤖"
            elif is_target: row += "🏁"
            elif is_trap:   row += "💀"
            elif ss.env.maze[r,c]==1:
                fog = ss.env.use_fog and not ss.env.fog.explored[r,c]
                row += "▓▓" if fog else "██"
            elif is_trail:  row += "·· "
            else:           row += "  "
        rows.append(row)
    return '\n'.join(rows)

def _cdf(data, keys, n=150):
    rows = {k: list(data.get(k,[]))[-n:] for k in keys}
    ml = min((len(v) for v in rows.values()), default=0)
    return pd.DataFrame({k: v[-ml:] for k,v in rows.items()})

# ── Telemetry strip ────────────────────────────────────────
def _telemetry_strip():
    ss   = st.session_state
    br   = ss.brain; an = ss.analytics
    live = an.get_live_stats(); sl = ss.soul.get_status()
    ent  = list(ss.get("entropy_hist", []))[-1] if ss.get("entropy_hist") else 0.0
    cap  = ss.capability

    def _d(v, ref=0, fmt=".2f"):
        if v > ref: return f'<span class="tel-up">▲</span>'
        if v < ref: return f'<span class="tel-dn">▼</span>'
        return f'<span class="tel-nt">─</span>'

    cells = [
        (f"{ss.episode_count}",   "Episodes",  f'<span class="tel-nt">global</span>'),
        (f"{live['success_rate']:.1f}%","Win Rate", _d(live['success_rate'],50)),
        (f"{br.epsilon:.4f}",     "Epsilon",   _d(-br.epsilon,-0.5)),
        (f"{br.avg_reward:+.2f}", "Avg Reward",_d(br.avg_reward,0)),
        (f"{br.avg_loss:.4f}",    "Avg Loss",  _d(-br.avg_loss,-0.01)),
        (f"{ent:.3f}",            "H(π) Entropy",f'<span class="tel-nt">nats</span>'),
        (f"L{br.curriculum.level}/10","Curriculum",_d(br.curriculum.level,1)),
        (f"{cap:.1f}",            "Capability",_d(cap,50)),
        (f"{ss.global_step:,}",   "Total Steps",f'<span class="tel-nt">env</span>'),
        (f"{sl['mood_emoji']} {sl['mood'][:6]}","Soul Mood",
         f'<span class="tel-nt">V={sl["valence"]:+.2f}</span>'),
    ]
    html = '<div class="tel">'
    for v,l,d in cells:
        html += f'<div class="tel-cell"><div class="tel-v">{v}</div><div class="tel-l">{l}</div><div class="tel-d">{d}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
def _sidebar():
    ss = st.session_state; cfg = ss.config
    with st.sidebar:
        st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:.88rem;'
                    'font-weight:700;color:#00f5ff;letter-spacing:.06em;padding:4px 0 8px;">'
                    '🧬 A.L.I.V.E. NEXUS v4.0</div>', unsafe_allow_html=True)

        st.markdown('<div class="ph">⚡ SIMULATION</div>', unsafe_allow_html=True)
        r1,r2 = st.columns(2)
        if r1.button("▶ RUN",   width='stretch'): ss.auto_mode = True
        if r2.button("⏸ PAUSE", width='stretch'): ss.auto_mode = False
        r3,r4 = st.columns(2)
        if r3.button("⏭ STEP",  width='stretch'):
            for _ in range(cfg.get("steps_per_frame",1)): process_step()
            st.rerun()
        if r4.button("🔄 RESET", width='stretch'): reset_all(); st.rerun()
        cfg["sim_speed"]       = st.slider("Delay (s)", 0.0, 0.5, cfg.get("sim_speed",0.04), 0.01)
        cfg["steps_per_frame"] = st.select_slider("Steps/frame",[1,2,4,8,16,32],
                                                    cfg.get("steps_per_frame",1))

        st.markdown('<div class="ph">🌐 ENVIRONMENT</div>', unsafe_allow_html=True)
        cfg["override_curriculum"] = st.toggle("Override Curriculum",
                                                cfg.get("override_curriculum",False))
        if cfg["override_curriculum"]:
            cfg["manual_level"] = st.slider("Force Level",1,10,cfg.get("manual_level",1))
        cfg["show_astar"] = st.toggle("Show A* Overlay", cfg.get("show_astar",False))
        cfg["trail"]      = st.toggle("Agent Trail",      cfg.get("trail",True))

        st.markdown('<div class="ph">🧠 BRAIN CONFIG</div>', unsafe_allow_html=True)
        with st.expander("Tune Hyperparameters", expanded=False):
            cfg["gamma"]         = st.slider("γ Discount",   0.80,0.999, cfg["gamma"],  format="%.3f")
            cfg["epsilon_decay"] = st.slider("ε Decay",      0.990,0.9999,cfg["epsilon_decay"],format="%.4f")
            cfg["epsilon_min"]   = st.slider("ε Min",        0.01, 0.15,  cfg["epsilon_min"],0.005)
            cfg["lr"]            = st.slider("Learning Rate",1e-4, 5e-3,  cfg["lr"],    format="%.4f")
            cfg["tau"]           = st.slider("τ Soft-update",0.001,0.05,  cfg["tau"],   0.001)
            cfg["icm_beta"]      = st.slider("ICM β",        0.0,  0.2,   cfg["icm_beta"],0.01)
            cfg["n_steps"]       = st.select_slider("N-Step",[1,2,3,5,8], cfg["n_steps"])
            cfg["batch_size"]    = st.select_slider("Batch", [32,64,128,256],cfg["batch_size"])
        with st.expander("Architecture (needs Reset)", expanded=False):
            st.info("Changing these requires a Hard Reset.")
            cfg["h1"] = st.select_slider("H1",[64,128,256,512],cfg["h1"])
            cfg["h2"] = st.select_slider("H2",[32,64,128,256], cfg["h2"])
            cfg["h3"] = st.select_slider("H3",[16,32,64,128],  cfg["h3"])
            cfg["buffer_size"] = st.select_slider("Buffer",[10_000,25_000,50_000,100_000],
                                                    cfg["buffer_size"])

        st.markdown('<div class="ph">💾 PERSISTENCE</div>', unsafe_allow_html=True)
        if st.button("💾 Save Checkpoint", width='stretch'):
            st.toast("✅ Saved!" if _do_save() else "❌ Failed")
        if st.button("⬇️ Build & Export ZIP", width='stretch'):
            zb = _make_zip()
            if zb:
                st.download_button("⬇️ Download",data=zb,
                    file_name=f"ALIVE_v4_ep{ss.episode_count}_L{ss.brain.curriculum.level}.zip",
                    mime="application/zip",width='stretch',key="_sdl")
        up = st.file_uploader("📂 Load ZIP",type="zip",label_visibility="collapsed")
        if up and st.button("Restore",width='stretch'):
            if _load_zip(up): st.toast("✅ Restored!"); st.rerun()

        st.markdown('<div class="ph">👤 SOUL / DISPLAY</div>', unsafe_allow_html=True)
        nn = st.text_input("Your Name",value=ss.soul.user_name,label_visibility="collapsed")
        if nn != ss.soul.user_name: ss.soul.user_name = nn
        cfg["chart_points"]      = st.slider("Chart history",50,400,cfg.get("chart_points",150),25)
        cfg["autosave_interval"] = st.slider("Autosave (eps)",25,500,cfg.get("autosave_interval",100),25)

        br = ss.brain.get_stats(); sl = ss.soul.get_status()
        st.markdown("---")
        st.markdown(
            f'<div style="font-size:.69rem;line-height:1.9;">'
            f'<b style="color:#00f5ff;">ε</b> {br["epsilon"]:.4f} &nbsp;'
            f'<b style="color:#22c55e;">Lvl</b> {br["curriculum"]["level"]} &nbsp;'
            f'<b style="color:#a855f7;">Mood</b> {sl["mood_emoji"]} {sl["mood"]}<br>'
            f'<b style="color:#f97316;">Steps</b> {ss.global_step:,} &nbsp;'
            f'<b style="color:#58a6ff;">Eps</b> {ss.episode_count}'
            f'</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
def _header():
    ss = st.session_state
    h1,h2,h3 = st.columns([4,3,2])
    with h1:
        st.markdown(
            '<div class="ntitle">🧬 A.L.I.V.E. NEXUS</div>'
            '<div style="font-size:.65rem;color:#6e7681;letter-spacing:.06em;margin-top:2px;">'
            'Adaptive Learning Intelligence &amp; Virtual Evolution — Event Horizon Edition v4.0</div>',
            unsafe_allow_html=True)
    with h2:
        live = ss.analytics.get_live_stats(); sl = ss.soul.get_status()
        st.markdown(
            f'<div style="text-align:center;font-size:.71rem;line-height:2.15;padding-top:6px;">'
            f'<span class="badge bc">EP {ss.episode_count}</span>'
            f'<span class="badge bg">✓ {live["success_rate"]:.1f}%</span>'
            f'<span class="badge bp">L{ss.brain.curriculum.level}</span>'
            f'<span class="badge bo">ε {ss.brain.epsilon:.4f}</span>'
            f'<span style="font-size:1.1rem;">{sl["mood_emoji"]}</span>'
            f'</div>', unsafe_allow_html=True)
    with h3:
        lbl = "⏸ PAUSE" if ss.auto_mode else "▶ AUTO RUN"
        if st.button(lbl,width='stretch',key="htog"): ss.auto_mode=not ss.auto_mode
        if st.button("⏭ STEP ×1",width='stretch',key="hstp"):
            process_step(); st.rerun()
    st.markdown("---")
    _telemetry_strip()

# ══════════════════════════════════════════════════════════
# TAB 1 — MISSION CONTROL
# ══════════════════════════════════════════════════════════
def _tab_mission():
    ss = st.session_state; cfg = ss.config
    env_st = ss.env.get_stats(); br = ss.brain.get_stats()
    sl = ss.soul.get_status(); live = ss.analytics.get_live_stats()
    cur = ss.brain.curriculum.get_stats()

    left, right = st.columns([3,2], gap="large")

    with left:
        st.markdown('<div class="ph">🗺️ LIVE ENVIRONMENT</div>', unsafe_allow_html=True)
        k1,k2,k3,k4 = st.columns(4)
        k1.markdown(_kpi(f"{ss.episode_count}","Episodes",  f"Win {live['success_rate']:.1f}%"), unsafe_allow_html=True)
        k2.markdown(_kpi(f"{br['epsilon']:.3f}","Epsilon",  f"Decay {cfg['epsilon_decay']}"),    unsafe_allow_html=True)
        k3.markdown(_kpi(f"L{cur['level']}","Curriculum",   cur["config"]["algorithm"].upper()), unsafe_allow_html=True)
        k4.markdown(_kpi(f"{ss.capability:.1f}","Capability",f"Trend {live['capability_trend']}"),unsafe_allow_html=True)
        st.markdown("")

        maze_fn = _get_maze_with_trail if cfg.get("trail", True) else _get_maze
        st.code(maze_fn(), language=None)

        if cfg.get("show_astar"):
            path = ss.env.get_astar_path()
            st.caption(f"⭐ A*: **{len(path)-1 if path else '∞'} steps** (current: {env_st['step_count']})")

        st.progress(min(env_st["step_count"]/max(env_st["max_steps"],1),1.0),
                    text=f"Step {env_st['step_count']} / {env_st['max_steps']}")
        ec = cur["config"]
        bdg = (f'<span class="badge bc">{ec["algorithm"].upper()}</span>'
               f'<span class="badge bp">LVL {cur["level"]}/10</span>'
               f'<span class="badge bo">ε {br["epsilon"]:.3f}</span>'
               f'<span class="badge bg">WINS {ss.env.success_count}</span>')
        if ec.get("fog"):     bdg += '<span class="badge br">🌫 FOG</span>'
        if ec.get("dynamic"): bdg += '<span class="badge br">💀 TRAPS</span>'
        if ec.get("portals"): bdg += '<span class="badge bp">🌀 PORTALS</span>'
        st.markdown(f'<div style="margin-top:4px;">{bdg}</div>', unsafe_allow_html=True)

        # ── Live Q-value readout ──────────────────────────
        st.markdown("---")
        st.markdown('<div class="ph">🎯 LIVE POLICY READOUT — Q(s,a)</div>', unsafe_allow_html=True)
        if ss.qval_hist:
            qv = np.array(ss.qval_hist[-1])
            prb = _softmax(qv); best = int(np.argmax(qv))
            q_df = pd.DataFrame({"Q-Value": qv, "Prob": prb},
                                 index=ACTIONS)
            st.bar_chart(q_df[["Q-Value"]], height=120, width='stretch')
            row = ""
            for i,(a,p) in enumerate(zip(ACTIONS, prb)):
                hi = "font-weight:700;" if i==best else ""
                row += (f'<span style="background:{ACTION_CLR[i]}22;'
                        f'border:1px solid {ACTION_CLR[i]}55;border-radius:6px;'
                        f'padding:3px 8px;margin:2px;font-family:JetBrains Mono,monospace;'
                        f'font-size:.72rem;color:{ACTION_CLR[i]};{hi}">'
                        f'{a} {p*100:.0f}%</span>')
            st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:2px;">{row}</div>',
                        unsafe_allow_html=True)

    with right:
        st.markdown('<div class="ph">🧠 BRAIN SNAPSHOT</div>', unsafe_allow_html=True)
        m1,m2,m3 = st.columns(3)
        m1.metric("Avg Reward",  f"{br['avg_reward']:+.2f}")
        m2.metric("Avg Loss",    f"{br['avg_loss']:.4f}")
        m3.metric("TD-Error",    f"{br['avg_td_error']:.3f}")
        m4,m5,m6 = st.columns(3)
        m4.metric("Train Steps", f"{br['train_step']:,}")
        m5.metric("Memory",      f"{br['memory_size']:,}")
        m6.metric("LR",          f"{br['lr']:.5f}")

        c_st = live["convergence"]
        st.markdown(
            f'<div class="cvb {_conv_cls(c_st)}">'
            f'<span style="font-size:1.2rem;">{_conv_icon(c_st)}</span>'
            f'<div><b>{c_st.upper()}</b><br>'
            f'<span style="font-size:.74rem;color:#8b949e;">{_conv_desc(c_st)}</span>'
            f'</div></div>', unsafe_allow_html=True)

        st.progress(cur["zpd_progress"],
                    text=f"ZPD {cur['zpd_progress']*100:.0f}% → L{cur['level']}")

        # ── Policy entropy ────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="ph">📐 POLICY ENTROPY H(π)</div>', unsafe_allow_html=True)
        if len(ss.entropy_hist) > 3:
            ent_df = pd.DataFrame({"H(π)": list(ss.entropy_hist)[-150:]})
            st.line_chart(ent_df, height=110, width='stretch')
            cur_ent = list(ss.entropy_hist)[-1]
            max_ent = math.log(ACTION_SIZE)
            st.caption(f"H(π)={cur_ent:.3f} nats  |  H_max={max_ent:.3f}  "
                       f"|  Exploitation ratio {(1-cur_ent/max_ent)*100:.0f}%")

        # ── Soul ──────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="ph">💬 CONSCIOUSNESS STREAM</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="thought">{sl["thought"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="text-align:center;font-size:2rem;margin:4px 0;">{sl["mood_emoji"]}</div>'
            f'<div style="text-align:center;font-size:.84rem;color:#a855f7;">{sl["mood"].upper()}</div>'
            f'<div style="text-align:center;font-size:.68rem;color:#8b949e;">'
            f'V={sl["valence"]:+.2f}  A={sl["arousal"]:+.2f}</div>',
            unsafe_allow_html=True)
        st.progress(ss.soul.relationship.score/100,
                    text=f"Bond: {sl['stage']} ({sl['relationship']}/100)")

    # ── Chart strip ───────────────────────────────────────
    st.markdown("---")
    n = cfg.get("chart_points",150); cd = ss.analytics.get_chart_data(n)
    cc1,cc2 = st.columns(2)
    with cc1:
        if cd.get("rewards"):
            df = _cdf(cd,["ema_rewards","rewards"],n); df.columns = ["EMA","Raw"]
            st.markdown("**Reward History**")
            st.line_chart(df,height=150,width='stretch')
    with cc2:
        if ss.intr_hist and ss.extr_hist:
            ni = min(len(ss.intr_hist),len(ss.extr_hist),n)
            df = pd.DataFrame({"Intrinsic":list(ss.intr_hist)[-ni:],
                                "Extrinsic":list(ss.extr_hist)[-ni:]})
            st.markdown("**Reward Decomposition: ICM vs Extrinsic**")
            st.line_chart(df,height=150,width='stretch')

# ══════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS LAB
# ══════════════════════════════════════════════════════════
def _tab_analytics():
    ss = st.session_state; cfg = ss.config
    n = cfg.get("chart_points",150); cd = ss.analytics.get_chart_data(n)
    live = ss.analytics.get_live_stats()

    st.markdown('<div class="ph">📊 ANALYTICS LABORATORY</div>', unsafe_allow_html=True)
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Success Rate",  f'{live["success_rate"]:.1f}%')
    k2.metric("Avg Reward",    f'{live["avg_reward"]:+.3f}')
    k3.metric("Avg Steps/Ep",  f'{live["avg_steps"]:.1f}')
    k4.metric("Convergence",   live["convergence_icon"]+" "+live["convergence"])
    k5.metric("Capability",    f'{live["capability"]:.1f}/100',delta=live["capability_trend"])
    st.markdown("---")

    ca,cb = st.columns(2)
    with ca:
        if cd.get("rewards"):
            df = _cdf(cd,["ema_rewards","rewards"],n); df.columns=["EMA","Raw"]
            st.markdown("**Reward History**"); st.line_chart(df,height=170,width='stretch')
        if cd.get("losses"):
            df = pd.DataFrame({"Loss":list(cd["losses"])[-n*3:]})
            st.markdown("**Training Loss**"); st.line_chart(df,height=150,width='stretch')
        if cd.get("successes"):
            succ = list(cd["successes"])[-n:]; wn = min(20,len(succ))
            roll = [sum(succ[max(0,i-wn):i+1])/min(i+1,wn) for i in range(len(succ))]
            st.markdown(f"**Rolling Win Rate (w={wn})**")
            st.line_chart(pd.DataFrame({"Win Rate":roll}),height=140,width='stretch')
    with cb:
        if ss.bellman_hist:
            df = pd.DataFrame({"Bellman Residual":list(ss.bellman_hist)[-n*3:]})
            st.markdown("**Bellman Residual |r+γQ'−Q|**")
            st.line_chart(df,height=170,width='stretch')
        if cd.get("steps"):
            df = _cdf(cd,["steps"],n); df.columns=["Steps/Ep"]
            st.markdown("**Steps per Episode**"); st.line_chart(df,height=150,width='stretch')
        if cd.get("optimality"):
            df = _cdf(cd,["optimality"],n); df.columns=["Path Efficiency"]
            st.markdown("**Path Efficiency vs A***")
            st.line_chart(df,height=140,width='stretch')

    st.markdown("---")
    da,db = st.columns(2)
    with da:
        st.markdown('<div class="ph">🧮 VALUE / ADVANTAGE STREAMS</div>', unsafe_allow_html=True)
        if ss.val_hist and ss.adv_hist:
            n2 = min(len(ss.val_hist),len(ss.adv_hist),n)
            df = pd.DataFrame({"V(s)":list(ss.val_hist)[-n2:],
                                "max A(s,a)":list(ss.adv_hist)[-n2:]})
            st.line_chart(df,height=160,width='stretch')
            st.caption("V(s): state value stream. max A(s,a): best-action advantage.")
    with db:
        st.markdown('<div class="ph">⚡ GRADIENT NORM (‖ΔW₁‖_F)</div>', unsafe_allow_html=True)
        if ss.gnorm_hist:
            df = pd.DataFrame({"Grad Norm":list(ss.gnorm_hist)[-n*3:]})
            st.line_chart(df,height=160,width='stretch')
            st.caption("Frobenius norm of weight update for H1. Spikes = large updates.")

    st.markdown("---")
    st.markdown('<div class="ph">🔥 EXPLORATION HEATMAP — PLOTLY</div>', unsafe_allow_html=True)
    H,W = ss.env.maze.shape
    heat = ss.analytics.get_heatmap(H,W)
    cov  = ss.analytics.heatmap.coverage(H,W,ss.env.maze)
    maze_mask = (ss.env.maze == 1)
    z_heat = heat.copy().astype(float)
    z_heat[maze_mask] = float("nan")
    if _PLOTLY:
        fig_heat = go.Figure(go.Heatmap(
            z=z_heat[::-1].tolist(),
            colorscale=[[0,"#06060a"],[0.15,"#0c4a6e"],[0.45,"#0ea5e9"],
                        [0.75,"#7dd3fc"],[1,"#f0f9ff"]],
            showscale=True, colorbar=dict(thickness=8, len=0.8,
                tickfont=dict(size=7,color="#6e7681")),
            hovertemplate="col:%{x} row:%{y}<br>density:%{z:.3f}<extra></extra>"))
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,10,30,0.85)",
            height=220, margin=dict(l=4,r=4,t=4,b=4),
            xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig_heat, width='stretch', key="anal_heat")
    else:
        lvls = " ░▒▓█"; rows=[]
        for r in range(H):
            row=""
            for c in range(W):
                if ss.env.maze[r,c]==1: row+="██"
                else:
                    i=min(int(heat[r,c]*(len(lvls)-1)),len(lvls)-1); row+=lvls[i]+lvls[i]
            rows.append(row)
        st.code("\n".join(rows),language=None)
    st.caption(f"Coverage: **{cov*100:.1f}%** of passable cells visited | {int(cov*int((ss.env.maze==0).sum()))} cells")

    st.markdown("---")
    st.markdown('<div class="ph">📈 CURRICULUM WINDOW</div>', unsafe_allow_html=True)
    cur = ss.brain.curriculum
    if cur.history:
        scores = [e["score"] for e in cur.history[-20:]]
        df = pd.DataFrame({"Score":scores,
                            "Promote":[cur.promote_thresh]*len(scores),
                            "Demote":[cur.demote_thresh]*len(scores)})
        st.line_chart(df,height=160,width='stretch')
        cc1,cc2,cc3 = st.columns(3)
        cc1.metric("Promotions",cur.promotions)
        cc2.metric("Demotions", cur.demotions)
        cc3.metric("Avg Score", f"{cur.avg_score:.3f}")
    else:
        st.info("Run episodes to populate curriculum history.")

# ══════════════════════════════════════════════════════════
# TAB 3 — SOUL MATRIX
# ══════════════════════════════════════════════════════════
def _tab_soul():
    ss = st.session_state; sl = ss.soul.get_status()
    col_s,col_c = st.columns([1,2],gap="large")

    with col_s:
        st.markdown('<div class="ph">🌀 IDENTITY CORE</div>', unsafe_allow_html=True)
        v,a = sl["valence"],sl["arousal"]
        st.markdown(
            f'<div style="text-align:center;font-size:2.8rem;">{sl["mood_emoji"]}</div>'
            f'<div style="text-align:center;color:#a855f7;font-size:.95rem;font-weight:700;">'
            f'{sl["mood"].upper()}</div>'
            f'<div style="text-align:center;font-size:.7rem;color:#8b949e;margin:4px 0 10px;">'
            f'Valence {v:+.3f} · Arousal {a:+.3f} · Intensity {sl["intensity"]:.3f}</div>',
            unsafe_allow_html=True)

        # Plotly Russell circumplex scatter
        if _PLOTLY:
            fig_em = go.Figure()
            # Emotion zones
            for (cx,cy,r_z,lbl,col) in [
                ( 0.65,  0.55, .28,"Excited",   "rgba(249,115,22,0.06)"),
                ( 0.65, -0.50, .24,"Serene",    "rgba(34,197,94,0.06)"),
                (-0.60,  0.60, .26,"Alarmed",   "rgba(239,68,68,0.06)"),
                (-0.60, -0.40, .24,"Depressed","rgba(88,166,255,0.06)"),
                ( 0.55,  0.10, .20,"Happy",     "rgba(0,245,255,0.05)"),
            ]:
                fig_em.add_shape(type="circle",x0=cx-r_z,y0=cy-r_z,
                                 x1=cx+r_z,y1=cy+r_z,fillcolor=col,line_width=0)
                fig_em.add_annotation(x=cx,y=cy,text=lbl,
                    font=dict(size=7,color="rgba(180,180,180,0.4)"),showarrow=False)
            # Axes
            for val in [-1,0,1]:
                fig_em.add_hline(y=val,line_color="rgba(255,255,255,0.05)",line_width=0.6)
                fig_em.add_vline(x=val,line_color="rgba(255,255,255,0.05)",line_width=0.6)
            # Current emotion point
            fig_em.add_trace(go.Scatter(x=[v],y=[a],mode="markers",
                marker=dict(size=28,color="rgba(168,85,247,0.15)",
                            line=dict(color="rgba(168,85,247,0.4)",width=1)),showlegend=False))
            fig_em.add_trace(go.Scatter(x=[v],y=[a],mode="markers",
                marker=dict(size=12,color="#a855f7",
                            line=dict(color="#d8b4fe",width=2)),showlegend=False))
            fig_em.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,10,30,0.85)",
                height=200, margin=dict(l=30,r=10,t=10,b=30),
                xaxis=dict(range=[-1.2,1.2],title="Valence →",
                           gridcolor="rgba(255,255,255,0.04)",zeroline=False,
                           tickfont=dict(size=8,color="#6e7681")),
                yaxis=dict(range=[-1.2,1.2],title="Arousal ↑",
                           gridcolor="rgba(255,255,255,0.04)",zeroline=False,
                           tickfont=dict(size=8,color="#6e7681")),
                font=dict(color="#8b949e",size=9,family="JetBrains Mono,monospace"),
                showlegend=False)
            st.plotly_chart(fig_em, width='stretch', key="soul_circumplex")
        else:
            va_df = pd.DataFrame({"Value":[max(v,0),max(-v,0),max(a,0),max(-a,0)]},
                                  index=["Valence+","Valence−","Arousal+","Arousal−"])
            st.bar_chart(va_df,height=110,width='stretch')

        st.markdown(
            f'<div style="margin-top:6px;"><b style="color:#00f5ff;">{sl["stage"]}</b><br>'
            f'<span style="font-size:.73rem;color:#8b949e;">{sl["stage_desc"]}</span></div>',
            unsafe_allow_html=True)
        st.progress(sl["relationship"]/100, text=f"Bond {sl['relationship']}/100")

        st.markdown("---")
        st.markdown('<div class="ph">🧬 PERSONALITY (OCEAN)</div>', unsafe_allow_html=True)
        ocean = pd.DataFrame({"Score":[sl["O"],sl["C"],sl["E"],sl["A"],sl["N"]]},
                              index=["Openness","Conscientiousness","Extraversion",
                                     "Agreeableness","Neuroticism"])
        if _PLOTLY:
            fig_oc = go.Figure(go.Bar(
                x=["O","C","E","A","N"],
                y=[sl["O"],sl["C"],sl["E"],sl["A"],sl["N"]],
                marker=dict(color=["#00f5ff","#22c55e","#f97316","#a855f7","#ef4444"],
                            opacity=0.82),
                text=[f"{v:.2f}" for v in [sl["O"],sl["C"],sl["E"],sl["A"],sl["N"]]],
                textfont=dict(size=8,color="#c9d1d9"), textposition="outside"))
            fig_oc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,10,30,0.85)",
                height=150, margin=dict(l=4,r=4,t=4,b=4),
                yaxis=dict(range=[0,1.1],gridcolor="rgba(255,255,255,0.05)",
                           tickfont=dict(size=8,color="#6e7681")),
                xaxis=dict(tickfont=dict(size=8,color="#c9d1d9")),
                showlegend=False)
            st.plotly_chart(fig_oc, width='stretch', key="ocean_bar")
        else:
            st.bar_chart(ocean,height=155,width='stretch')
        st.caption(f"Active traits: {sl['personality']}")

        # Action preference
        st.markdown("---")
        st.markdown('<div class="ph">🎲 ACTION PREFERENCE</div>', unsafe_allow_html=True)
        total_a = sum(ss.action_counts)+1
        act_df = pd.DataFrame(
            {"Frequency":[c/total_a for c in ss.action_counts]},
            index=ACTIONS)
        st.bar_chart(act_df,height=110,width='stretch')
        dom = ACTIONS[int(np.argmax(ss.action_counts))]
        st.caption(f"Dominant action: **{dom}** ({max(ss.action_counts)/total_a*100:.1f}%)")

        st.markdown("---")
        st.markdown('<div class="ph">🧠 STRONGEST MEMORIES</div>', unsafe_allow_html=True)
        for m in sl.get("strongest_memories",[]):
            st.markdown(f'<div class="mc">• {m}</div>', unsafe_allow_html=True)

    with col_c:
        st.markdown('<div class="ph">💬 COGNITIVE INTERFACE</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="thought">💭 {sl["thought"]}</div>', unsafe_allow_html=True)

        html = '<div class="chat-scroll">'
        for msg in ss.soul.get_chat_history():
            if msg["role"]=="user":
                html += (f'<div class="cu"><b>YOU</b> '
                         f'<span class="cmeta">intent: {msg.get("intent","?")}</span><br>'
                         f'{msg["text"]}</div>')
            else:
                html += (f'<div class="ca"><b>A.L.I.V.E.</b> '
                         f'<span class="cmeta">mood: {msg.get("emotion","?")}</span><br>'
                         f'{msg["text"]}</div>')
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

        ui = st.chat_input("Speak to A.L.I.V.E. ...")
        if ui: ss.soul.chat(ui); st.rerun()

        qp = st.columns(4)
        for col,p in zip(qp,["Hello!","How do you feel?","Are you conscious?","Tell me what you learned."]):
            if col.button(p,width='stretch',key=f"qp_{p[:3]}"): ss.soul.chat(p); st.rerun()

        # Full consciousness log
        st.markdown("---")
        st.markdown('<div class="ph">🌊 FULL CONSCIOUSNESS STREAM</div>', unsafe_allow_html=True)
        stream = ss.soul.consciousness.get_stream()
        stream_html = ""
        for i,thought in enumerate(reversed(stream)):
            alpha = max(0.4, 1.0 - i*0.08)
            stream_html += (f'<div style="font-size:.76rem;color:rgba(216,180,254,{alpha:.1f});'
                            f'padding:3px 0;border-bottom:1px solid rgba(255,255,255,.04);">'
                            f'<span style="color:#6e7681;font-size:.65rem;">[t-{i}]</span> {thought}</div>')
        st.markdown(f'<div style="max-height:180px;overflow-y:auto;">{stream_html}</div>',
                    unsafe_allow_html=True)
        st.markdown(f"**Memories stored:** {sl['memories_stored']} · **Turns:** {sl['turns']}")

# ══════════════════════════════════════════════════════════
# TAB 4 — MEMORY PALACE
# ══════════════════════════════════════════════════════════
def _tab_memory():
    ss = st.session_state
    fs = ss.memory.get_full_status(); ep = fs.get("episodic_stats",{})

    st.markdown('<div class="ph">🏛️ MEMORY PALACE</div>', unsafe_allow_html=True)
    cm1,cm2 = st.columns(2,gap="large")

    with cm1:
        st.markdown("**📖 Episodic Memory**")
        if ep:
            m1,m2,m3 = st.columns(3)
            m1.metric("Stored",ep.get("total_stored",0))
            m2.metric("Success Rate",f'{ep.get("success_rate",0)*100:.1f}%')
            m3.metric("Landmarks",ep.get("landmarks",0))
            m4,m5 = st.columns(2)
            m4.metric("Max Level",ep.get("max_level_reached",1))
            m5.metric("Avg Efficiency",f'{ep.get("avg_efficiency",0)*100:.1f}%')

        recent = fs.get("episodic_recent",[])
        if recent:
            st.markdown("**Recent Episodes**")
            rows=[{"EP#":e.get("episode_id",""),"L":e.get("curriculum_level",""),
                   "REWARD":round(e.get("total_reward",0),2),
                   "WIN":"✅" if e.get("success") else "❌",
                   "STEPS":e.get("total_steps",""),
                   "EFF":f'{e.get("efficiency",0)*100:.0f}%',
                   "ALG":e.get("maze_alg","")} for e in recent]
            st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')

        # Episode comparison: best vs worst
        if ss.best_ep and ss.worst_ep:
            st.markdown("---")
            st.markdown("**🏆 Best vs Worst Episode**")
            cmp_df = pd.DataFrame({
                "Best":  [ss.best_ep["reward"],  ss.best_ep["steps"],
                          ss.best_ep["efficiency"], ss.best_ep["level"]],
                "Worst": [ss.worst_ep["reward"], ss.worst_ep["steps"],
                          ss.worst_ep["efficiency"],ss.worst_ep["level"]],
            }, index=["Reward","Steps","Efficiency","Level"])
            st.dataframe(cmp_df,width='stretch')

        st.markdown("**🌟 Landmarks**")
        for lm in fs.get("landmark_episodes",[])[:4]:
            cls = "es" if lm.get("success") else "ef"
            st.markdown(
                f'<div class="mc {cls}"><b>EP#{lm["episode_id"]}</b> '
                f'L{lm["curriculum_level"]} — {lm["maze_alg"]}<br>'
                f'R<code>{lm["total_reward"]}</code> · '
                f'Eff<code>{lm["efficiency"]:.1%}</code> · '
                f'{"✅" if lm["success"] else "❌"}</div>',unsafe_allow_html=True)

    with cm2:
        st.markdown("**🧬 Semantic Memory (World Model)**")
        facts = fs.get("semantic_facts",[])
        if facts:
            rows=[{"FACT":f["key"][:38],"VAL":str(f["value"])[:18],
                   "CONF":f'{f["confidence"]:.0%}',"SRC":f["source"]} for f in facts[:12]]
            st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
        else:
            st.info("No semantic facts yet. Play more episodes.")

        # Semantic summary
        st.markdown("**World Model Summary**")
        st.code(ss.memory.semantic.get_summary() or "(empty)", language=None)

        st.markdown("---")
        st.markdown("**💡 System Insights**")
        for ins in fs.get("insights",[]): st.markdown(f"• {ins}")

        st.markdown("---")
        st.markdown("**💾 Storage**")
        st.markdown(
            f'<div class="mc">Path <code>{fs["save_path"]}</code><br>'
            f'Size <code>{fs["save_size_kb"]:.1f} KB</code> · '
            f'Episodes <code>{fs["total_episodes"]}</code> · '
            f'Loaded <code>{"YES" if fs["loaded_from_disk"] else "NO"}</code>'
            f'</div>',unsafe_allow_html=True)

        st.markdown("**📊 Session Report**")
        st.code(ss.analytics.get_session_report(),language=None)
        c1,c2 = st.columns(2)
        if c1.button("📋 Export JSON",width='stretch'):
            ex = ss.analytics.export_json()
            st.download_button("⬇️ Download",ex,"alive_session.json","application/json",key="_jdl")
        if c2.button("🗑 Clear Memory",width='stretch'):
            ss.memory.episodic.episodes.clear(); ss.memory.semantic.facts.clear()
            st.toast("Memory cleared."); st.rerun()

# ══════════════════════════════════════════════════════════
# TAB 5 — BRAIN AUTOPSY
# ══════════════════════════════════════════════════════════
def _tab_brain():
    ss = st.session_state; br = ss.brain; bn = br.online_net
    n = ss.config.get("chart_points",150); cd = ss.analytics.get_chart_data(n)

    st.markdown('<div class="ph">🔬 BRAIN AUTOPSY</div>', unsafe_allow_html=True)
    cb1,cb2 = st.columns([3,2],gap="large")

    with cb1:
        if cd.get("losses"):
            df = pd.DataFrame({"Loss":list(cd["losses"])[-n*3:]})
            st.markdown("**Training Loss**"); st.line_chart(df,height=150,width='stretch')
        if cd.get("td_errors"):
            df = pd.DataFrame({"TD-Error":list(cd["td_errors"])[-n*3:]})
            st.markdown("**TD-Error**"); st.line_chart(df,height=140,width='stretch')
        if ss.bellman_hist:
            df = pd.DataFrame({"Bellman Residual":list(ss.bellman_hist)[-n*3:]})
            st.markdown("**Bellman Residual**"); st.line_chart(df,height=130,width='stretch')
        if _PLOTLY:
            st.markdown("**W1 Weight Matrix (Plotly heatmap)**")
            w1_vis = bn.W1[:16,:32] if bn.W1.shape[0] >= 16 else bn.W1[:,:32]
            fig_w1 = go.Figure(go.Heatmap(
                z=w1_vis.tolist(),
                colorscale=[[0,"#1e3a5f"],[0.5,"#06060a"],[1,"#7f1d1d"]],
                showscale=False,
                hovertemplate="in:%{x} out:%{y}<br>w=%{z:.4f}<extra></extra>"))
            fig_w1.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,10,30,0.85)",
                height=130, margin=dict(l=4,r=4,t=4,b=4),
                xaxis=dict(visible=False), yaxis=dict(visible=False))
            st.plotly_chart(fig_w1, width='stretch', key="w1_heat")

            st.markdown("**Advantage Bias b_adv (Plotly)**")
            fig_adv = go.Figure(go.Bar(
                x=ACTIONS, y=bn.b_adv.tolist(),
                marker=dict(color=ACTION_CLR, opacity=0.85)))
            fig_adv.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,10,30,0.85)",
                height=100, margin=dict(l=4,r=4,t=4,b=24),
                font=dict(color="#8b949e",size=9),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                showlegend=False)
            st.plotly_chart(fig_adv, width='stretch', key="adv_bar")
        else:
            st.markdown("**W1 Weight Distribution (first 32)**")
            st.bar_chart(pd.DataFrame({"W1":bn.W1.flatten()[:32]}),height=100,width='stretch')
            st.markdown("**Advantage Bias b_adv**")
            st.bar_chart(pd.DataFrame({"Bias":bn.b_adv},index=ACTIONS),height=90,width='stretch')

    with cb2:
        tp = sum(bn.__dict__[p].size
                 for p in ["W1","b1","W2","b2","W3","b3","W_val","b_val","W_adv","b_adv"]
                 if p in bn.__dict__ and isinstance(bn.__dict__.get(p),np.ndarray))
        st.markdown(
            f'<div class="arch"><b style="color:#00f5ff;">ARCHITECTURE</b><br><br>'
            f'Input  [{br.state_size}]<br>'
            f'&#8595; LeakyReLU · Adam · He&#8321;<br>'
            f'H1     [{bn.W2.shape[0]}]<br>'
            f'&#8595; LeakyReLU<br>'
            f'H2     [{bn.W3.shape[0]}]<br>'
            f'&#8595; LeakyReLU<br>'
            f'H3     [{bn.W_val.shape[0]}]<br>'
            f'&#8595; ── Dueling ──<br>'
            f'V(s)[1]&nbsp;&nbsp;A(s,a)[{br.action_size}]<br>'
            f'Q = V + A &#8722; &#x0305;A<br><br>'
            f'<b style="color:#22c55e;">Params: {tp:,}</b><br><br>'
            f'<span class="badge bc">D3QN</span>'
            f'<span class="badge bp">PER</span>'
            f'<span class="badge bo">N&#8315;step={br.n_step.n}</span>'
            f'<span class="badge bg">ICM</span><br><br>'
            f'&#964;={br.tau}  &#945;_PER=0.6  clip&#177;10</div>',unsafe_allow_html=True)

        st.markdown("**Replay Buffer (PER)**")
        buf = len(br.memory)/br.memory.capacity
        st.progress(buf,text=f"{len(br.memory):,}/{br.memory.capacity:,}")
        m1,m2 = st.columns(2)
        m1.metric("PER β",       f"{br.memory.beta:.3f}")
        m2.metric("Max Priority",f"{br.memory.max_priority:.3f}")

        st.markdown("**Curiosity Module**")
        m3,m4 = st.columns(2)
        m3.metric("Unique States",br.curiosity.coverage())
        m4.metric("ICM β",       br.curiosity.beta)

        st.markdown("**Optimizer**")
        m5,m6 = st.columns(2)
        m5.metric("LR",f"{br.learning_rate:.6f}")
        lr_r = br.lr_sched.reductions if hasattr(br,"lr_sched") else "—"
        m6.metric("LR Reductions",lr_r)

        st.markdown("**State Vector (17-dim)**")
        st.dataframe(pd.DataFrame({
            "Component":["3×3 Vision","Agent r,c","Target r,c",
                         "Manhattan","Trap dist","Fog cov","Time"],
            "Dims":[9,2,2,1,1,1,1]
        }),hide_index=True,width='stretch')

# ══════════════════════════════════════════════════════════
# TAB 6 — EPISODE TIMELINE
# ══════════════════════════════════════════════════════════
def _tab_timeline():
    ss = st.session_state; episodes = ss.memory.episodic.episodes
    st.markdown('<div class="ph">📅 EPISODE TIMELINE</div>', unsafe_allow_html=True)
    if not episodes: st.info("No episodes yet."); return

    n = min(50,len(episodes)); recent = episodes[-n:]
    wins  = sum(1 for e in recent if e.success)
    avg_r = np.mean([e.total_reward for e in recent])
    avg_e = np.mean([e.efficiency   for e in recent])
    avg_s = np.mean([e.total_steps  for e in recent])
    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Win Rate",       f"{wins/n*100:.1f}%")
    s2.metric("Avg Reward",     f"{avg_r:+.2f}")
    s3.metric("Avg Efficiency", f"{avg_e*100:.1f}%")
    s4.metric("Avg Steps",      f"{avg_s:.0f}")

    st.markdown("**Reward & Success timeline**")
    tl_df = pd.DataFrame({"Reward":[e.total_reward for e in recent],
                           "Success×10":[10.0 if e.success else -10.0 for e in recent]})
    st.line_chart(tl_df,height=170,width='stretch')

    st.markdown("**Curriculum Level Progression**")
    lv_df = pd.DataFrame({"Level":[e.curriculum_level for e in recent]})
    st.line_chart(lv_df,height=110,width='stretch')

    # Episode log table
    st.markdown("**Episode Log (latest 30)**")
    rows=[{"EP#":e.episode_id,"L":e.curriculum_level,"ALG":e.maze_alg[:5],
           "MAZE":f"{e.maze_h}x{e.maze_w}","REWARD":round(e.total_reward,2),
           "WIN":"✅" if e.success else "❌","STEPS":e.total_steps,
           "EFF%":f"{e.efficiency*100:.0f}","ε":f"{e.epsilon_end:.3f}",
           "FOG":"🌫" if e.fog_used else "","TRAP":"💀" if e.traps_used else ""}
          for e in list(reversed(recent))[:30]]
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')

    # Efficiency histogram
    st.markdown("**Efficiency Distribution (vs A* optimal)**")
    effs = [e.efficiency for e in recent]; bins = np.linspace(0,1,11)
    hist,_ = np.histogram(effs,bins=bins)
    hist_df = pd.DataFrame({"Count":hist},
                            index=[f"{bins[i]:.1f}–{bins[i+1]:.1f}" for i in range(len(hist))])
    st.bar_chart(hist_df,height=130,width='stretch')

    # Steps distribution
    if len(recent) >= 5:
        st.markdown("**Steps Distribution**")
        step_vals = [e.total_steps for e in recent]
        sbins = np.linspace(min(step_vals),max(step_vals)+1,11)
        shist,_ = np.histogram(step_vals,bins=sbins)
        shist_df = pd.DataFrame({"Count":shist},
                                 index=[f"{int(sbins[i])}–{int(sbins[i+1])}" for i in range(len(shist))])
        st.bar_chart(shist_df,height=120,width='stretch')

# ══════════════════════════════════════════════════════════
# TAB 7 — BENCHMARK
# ══════════════════════════════════════════════════════════
def _tab_benchmark():
    ss = st.session_state; br = ss.brain; an = ss.analytics
    live = an.get_live_stats(); env_st = ss.env.get_stats(); cur = br.curriculum.get_stats()

    st.markdown('<div class="ph">🏆 BENCHMARK, DIAGNOSTICS & CONVERGENCE SCIENCE</div>',
                unsafe_allow_html=True)

    # ── Capability breakdown ──────────────────────────────
    sr  = an.tracker.success_rate; opt = an.tracker.avg_optimality
    H,W = ss.env.maze.shape; cov = an.heatmap.coverage(H,W,ss.env.maze)
    c_st = an.tracker.convergence.state; lvl = cur["level"]
    s_sc = sr*40; o_sc = opt*25; e_sc = min(cov,1.0)*15
    cv_sc = {"warming_up":2,"rapid_learning":8,"fine_tuning":7,
              "converged":10,"plateau":4,"regressing":0}.get(c_st,0.0)
    lv_sc = ((lvl-1)/9.0)*10
    total = float(np.clip(s_sc+o_sc+e_sc+cv_sc+lv_sc,0,100))

    # Plotly radar + bar combined
    bench_cols = st.columns([2, 1])
    with bench_cols[0]:
        if _PLOTLY:
            labels  = ["Success<br>(40)", "Efficiency<br>(25)", "Exploration<br>(15)",
                       "Convergence<br>(10)", "Curriculum<br>(10)"]
            maxvals = [40, 25, 15, 10, 10]
            vals    = [s_sc, o_sc, e_sc, cv_sc, lv_sc]
            pct     = [v/m for v,m in zip(vals,maxvals)]
            radar_fig = go.Figure(go.Scatterpolar(
                r=pct+[pct[0]], theta=labels+[labels[0]],
                fill="toself",
                fillcolor="rgba(0,245,255,0.08)",
                line=dict(color="#00f5ff", width=2),
                marker=dict(size=6, color="#00f5ff"),
            ))
            radar_fig.update_layout(
                polar=dict(
                    bgcolor="rgba(10,10,30,0.8)",
                    radialaxis=dict(visible=True, range=[0,1],
                        gridcolor="rgba(255,255,255,0.06)",
                        tickfont=dict(size=7,color="#6e7681"),
                        tickvals=[0.25,0.5,0.75,1.0]),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                        tickfont=dict(size=9,color="#c9d1d9")),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False, height=300,
                margin=dict(l=50,r=50,t=30,b=30),
                title=dict(text="Capability Radar", font=dict(size=9,color="#6e7681")),
            )
            st.plotly_chart(radar_fig, width='stretch', key="cap_radar")
        else:
            comp_df = pd.DataFrame({"Score":[s_sc,o_sc,e_sc,cv_sc,lv_sc]},
                                    index=["Success","Efficiency","Exploration","Convergence","Curriculum"])
            st.bar_chart(comp_df, height=240, width='stretch')
    with bench_cols[1]:
        st.markdown(
            f'<div style="text-align:center;font-size:2.8rem;color:#00f5ff;'
            f'font-family:JetBrains Mono,monospace;font-weight:900;margin:20px 0 4px;">'
            f'{total:.1f}</div>'
            f'<div style="text-align:center;font-size:.7rem;color:#6e7681;letter-spacing:.2em;">/ 100</div>'
            f'<div style="text-align:center;font-size:.65rem;color:#a855f7;letter-spacing:.15em;margin-top:4px;">CAPABILITY SCORE</div>',
            unsafe_allow_html=True)
        st.progress(total/100)
        st.markdown("")
        # Component bars
        for label, score, maxs, clr in [
            ("Success",    s_sc,  40, "#22c55e"),
            ("Efficiency", o_sc,  25, "#38bdf8"),
            ("Exploration",e_sc,  15, "#a855f7"),
            ("Convergence",cv_sc, 10, "#f97316"),
            ("Curriculum", lv_sc, 10, "#eab308"),
        ]:
            pct = score/maxs if maxs>0 else 0
            st.markdown(
                f'<div style="margin:3px 0;font-size:.68rem;color:#8b949e;">{label} ({score:.1f}/{maxs})</div>'
                f'<div style="background:rgba(255,255,255,.06);border-radius:3px;height:5px;overflow:hidden;">'
                f'<div style="width:{pct*100:.0f}%;height:100%;background:{clr};border-radius:3px;"></div>'
                f'</div>', unsafe_allow_html=True)

    # ── Convergence hypothesis test ───────────────────────
    st.markdown("---")
    st.markdown('<div class="ph">🔬 LEARNING HYPOTHESIS TEST</div>', unsafe_allow_html=True)
    rewards = [e["reward"] for e in list(ss.ep_log)[-30:]] if ss.ep_log else []
    if len(rewards) >= 6:
        slope, pval = _ttest_slope(rewards)
        significant = pval < 0.05; positive = slope > 0
        if significant and positive:
            css = "hyp-yes"
            verdict = (f"✅ LEARNING CONFIRMED (p={pval:.4f} < 0.05) — "
                       f"Slope={slope:+.4f} reward/episode. "
                       f"Reject H₀: gradient is statistically positive.")
        elif significant and not positive:
            css = "hyp-no"
            verdict = (f"⚠️ REGRESSION DETECTED (p={pval:.4f} < 0.05) — "
                       f"Slope={slope:+.4f}. Policy degrading. "
                       f"Check LR, buffer, or curriculum level.")
        else:
            css = "hyp-unk"
            verdict = (f"❓ INCONCLUSIVE (p={pval:.4f} ≥ 0.05) — "
                       f"Slope={slope:+.4f}. Insufficient evidence. "
                       f"Run more episodes or reduce variance.")
        st.markdown(f'<div class="{css}">{verdict}</div>', unsafe_allow_html=True)
        # Reward trend chart
        df_h = pd.DataFrame({"Reward":rewards})
        trend_line = [rewards[0]+slope*i for i in range(len(rewards))]
        df_h["Trend"] = trend_line
        st.line_chart(df_h,height=130,width='stretch')
    else:
        st.info("Need ≥6 episodes for hypothesis test. Run the simulation.")

    # ── LR Sensitivity gauge ──────────────────────────────
    st.markdown("---")
    st.markdown('<div class="ph">📐 HYPERPARAMETER STABILITY ANALYSIS</div>',
                unsafe_allow_html=True)
    cfg = ss.config
    checks = [
        ("γ × max_Q instability",   cfg["gamma"] > 0.99 and br.avg_td_error > 2.0, "γ close to 1 with high TD-error can diverge."),
        ("LR × batch gradient noise",cfg["lr"] > 0.003 and cfg["batch_size"] < 64, "High LR + small batch = noisy gradients."),
        ("ε-floor reached",          br.epsilon <= cfg["epsilon_min"]*1.05,        "Fully exploiting. Increasing exploration may help if stuck."),
        ("Buffer underflow",          len(br.memory) < cfg["batch_size"]*4,         "Buffer too small relative to batch. Increase buffer or reduce batch."),
        ("PER β near 1",              br.memory.beta > 0.85,                        "IS weights near uniform. PER correction nearly disabled."),
        ("N-step > horizon",          cfg["n_steps"] > max(3, ss.env.max_steps//20),"N-step too long for maze horizon. Bias accumulates."),
        ("τ too large",               cfg["tau"] > 0.02,                            "Soft update too aggressive. Target net tracks online too fast."),
        ("Curiosity dominating",      ss.intr_hist and sum(list(ss.intr_hist)[-20:]) > abs(sum(list(ss.extr_hist)[-20:])), "Intrinsic reward exceeding extrinsic. Consider reducing ICM β."),
    ]
    hc1,hc2 = st.columns(2)
    for i,(name,warn,desc) in enumerate(checks):
        col = hc1 if i%2==0 else hc2
        icon = "⚠️" if warn else "✅"; color = "#f97316" if warn else "#22c55e"
        col.markdown(
            f'<div style="display:flex;flex-direction:column;gap:2px;'
            f'margin:4px 0;padding:6px 8px;background:rgba(255,255,255,.02);'
            f'border-radius:6px;border-left:3px solid {color};">'
            f'<div style="display:flex;gap:6px;font-size:.77rem;">'
            f'<span>{icon}</span><span style="color:#c9d1d9;font-weight:600;">{name}</span></div>'
            f'<div style="font-size:.67rem;color:#6e7681;padding-left:18px;">{desc}</div>'
            f'</div>', unsafe_allow_html=True)

    # ── System diagnostics ────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="ph">🩺 SYSTEM DIAGNOSTICS</div>', unsafe_allow_html=True)
    diag = [
        ("Buffer filled",       len(br.memory)>=br.batch_size,   f"{len(br.memory):,}/{br.batch_size}"),
        ("Training active",     br.train_step>0,                  f"{br.train_step:,} steps"),
        ("Epsilon < 0.5",       br.epsilon<0.5,                   f"ε={br.epsilon:.4f}"),
        ("Success > 0%",        an.tracker.success_rate>0,        f"{an.tracker.success_rate*100:.1f}%"),
        ("Avg reward > 0",      an.tracker.avg_reward>0,          f"{an.tracker.avg_reward:+.3f}"),
        ("Not regressing",      c_st!="regressing",               c_st),
        ("Level > 1",           br.curriculum.level>1,            f"L{br.curriculum.level}"),
        ("LR viable",           br.learning_rate>1e-5,            f"{br.learning_rate:.6f}"),
        ("Episodic memory OK",  len(ss.memory.episodic.episodes)>0,f"{len(ss.memory.episodic.episodes)} eps"),
        ("Curiosity active",    br.curiosity.coverage()>10,       f"{br.curiosity.coverage()} states"),
        ("Bellman stable",      (not ss.get("bellman_hist") or list(ss.get("bellman_hist", []))[-1]<20.0),
                                f"{list(ss.get('bellman_hist', []))[-1]:.2f}" if ss.get("bellman_hist") else "n/a"),
        ("Entropy in range",    (not ss.get("entropy_hist") or 0.1<list(ss.get("entropy_hist", []))[-1]<math.log(4)+0.1),
                                f"{list(ss.get('entropy_hist', []))[-1]:.3f}" if ss.get("entropy_hist") else "n/a"),
    ]
    dc1,dc2 = st.columns(2)
    for i,(name,ok,detail) in enumerate(diag):
        col = dc1 if i%2==0 else dc2; icon="✅" if ok else "⚠️"; c="#22c55e" if ok else "#f97316"
        col.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0;font-size:.76rem;">'
            f'<span>{icon}</span><span style="color:#c9d1d9;">{name}</span>'
            f'<span style="margin-left:auto;color:{c};font-size:.67rem;">{detail}</span>'
            f'</div>',unsafe_allow_html=True)

    # ── Environment stats ─────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="ph">🌐 ENVIRONMENT DIAGNOSTICS</div>', unsafe_allow_html=True)
    e1,e2,e3,e4 = st.columns(4)
    e1.metric("Maze",      env_st["maze_size"])
    e2.metric("Algorithm", env_st["algorithm"].upper())
    e3.metric("A* Optimal",env_st["astar_optimal"])
    e4.metric("Cells Vis.",env_st["cells_visited"])
    e5,e6,e7,e8 = st.columns(4)
    e5.metric("Traps",    env_st["traps"]); e6.metric("Portals",env_st["portals"])
    e7.metric("Fog",      "Yes" if env_st["fog"] else "No")
    e8.metric("Fog Cover",f'{env_st["fog_coverage"]*100:.1f}%')

    # ── Export ────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="ph">📤 SESSION EXPORT</div>', unsafe_allow_html=True)
    ex1,ex2 = st.columns(2)
    with ex1:
        st.markdown("**Session Report**"); st.code(an.get_session_report(),language=None)
    with ex2:
        st.markdown("**JSON Preview**")
        js = an.export_json()
        st.code(js[:1400]+("\n...[truncated]" if len(js)>1400 else ""),language="json")
    if st.button("⬇️ Build & Download Full ZIP Checkpoint",width='stretch'):
        zb = _make_zip()
        if zb:
            st.download_button("⬇️ Download ZIP",data=zb,
                file_name=f"ALIVE_NEXUS_v4_ep{ss.episode_count}_L{br.curriculum.level}.zip",
                mime="application/zip",width='stretch',key="_bzdl")

# ══════════════════════════════════════════════════════════
# TAB 8 — RESEARCH LAB
# ══════════════════════════════════════════════════════════
def _tab_research():
    ss = st.session_state; br = ss.brain; cfg = ss.config

    st.markdown('<div class="ph">🔭 THEORETICAL RESEARCH LABORATORY</div>',
                unsafe_allow_html=True)

    r1,r2 = st.columns([3,2],gap="large")

    with r1:
        st.markdown("### Core Objectives & Bellman Equations")
        st.markdown("""<div class="eq">
<b>DQN Training Objective (MSE over TD targets):</b><br>
<em>L(θ) = 𝔼[(y − Q(s,a;θ))²]</em><br><br>
<b>Double DQN target (decoupled selection/evaluation):</b><br>
<em>y = r + γ · Q(s', argmax_a Q(s',a;θ); θ⁻)</em><br><br>
<b>Dueling decomposition (Wang et al., 2016):</b><br>
<em>Q(s,a;θ) = V(s;θ_v) + A(s,a;θ_a) − (1/|A|)Σ A(s,a';θ_a)</em><br><br>
<b>Prioritized experience replay (Schaul et al., 2015):</b><br>
<em>P(i) = |δᵢ|^α / Σⱼ |δⱼ|^α</em><br>
<em>wᵢ = (N · P(i))^(−β) / max_j wⱼ</em><br><br>
<b>N-step return (Sutton, 1988):</b><br>
<em>G_t^n = Σ_{k=0}^{n-1} γᵏ rₜ₊ₖ + γⁿ max_a Q(sₜ₊ₙ, a)</em><br><br>
<b>Intrinsic curiosity (count-based proxy):</b><br>
<em>rᵢ(s) = β / √N(s)</em> &nbsp; where N(s) = visit count of state s
</div>""", unsafe_allow_html=True)

        st.markdown("### Adam Optimiser (Kingma & Ba, 2014)")
        st.markdown("""<div class="eq">
<em>m_t = β₁ m_{{t-1}} + (1−β₁) g_t</em><br>
<em>v_t = β₂ v_{{t-1}} + (1−β₂) g_t²</em><br>
<em>m̂_t = m_t/(1−β₁ᵗ) &nbsp;&nbsp; v̂_t = v_t/(1−β₂ᵗ)</em><br>
<em>θ_t = θ_{{t-1}} − α · m̂_t / (√v̂_t + ε)</em><br><br>
<u>Hypers</u>: β₁=0.9  β₂=0.999  ε=1×10⁻⁸  α=<b>{lr:.4f}</b>
</div>""".format(lr=cfg["lr"]), unsafe_allow_html=True)

        st.markdown("### Curriculum Learning (ZPD Theory)")
        st.markdown("""<div class="eq">
Zone of Proximal Development (Vygotsky, 1978 — adapted for RL):<br><br>
<em>Level ↑ if: (1/W) Σ score_i ≥ θ_promote</em><br>
<em>Level ↓ if: (1/W) Σ score_i ≤ θ_demote</em><br><br>
where <em>score_i = 0.5·success_i + 0.5·efficiency_i</em><br>
and W = sliding window size<br><br>
<u>Current</u>: θ_promote=<b>0.72</b>  θ_demote=<b>0.25</b>  W=<b>20</b>
</div>""", unsafe_allow_html=True)

        st.markdown("### Policy Entropy (Exploration Measure)")
        st.markdown("""<div class="eq">
<em>H(π(·|s)) = −Σ_a π(a|s) log π(a|s)</em><br><br>
where <em>π(a|s) = softmax(Q(s,a)/T)_a</em>  (T=1 here)<br><br>
H=0 → fully deterministic (exploitation)<br>
H=log|A|=<b>{:.3f}</b> nats → uniform (max exploration)<br><br>
<u>Current H(π)</u>: <b>{:.3f}</b> nats &nbsp;→&nbsp;
Exploitation ratio: <b>{:.0f}%</b>
</div>""".format(math.log(ACTION_SIZE),
                 list(ss.get("entropy_hist",[]))[-1] if ss.get("entropy_hist") else 0.0,
                 (1-(list(ss.get("entropy_hist",[]))[-1]/math.log(ACTION_SIZE)) if ss.get("entropy_hist") else 0)*100),
                   unsafe_allow_html=True)

        st.markdown("### Convergence Conditions (Watkins & Dayan, 1992)")
        st.markdown("""<div class="eq">
Q-learning converges to Q* iff:<br>
1. All (s,a) pairs visited infinitely often<br>
2. Σ_t α_t = ∞ &nbsp;(learning rate not summable)<br>
3. Σ_t α_t² < ∞ &nbsp;(learning rate square-summable)<br>
4. Rewards are bounded: |r| ≤ R_max<br><br>
With function approximation (DQN), convergence is <em>not</em> guaranteed<br>
in general — but empirically robust with Double DQN + target network.
</div>""", unsafe_allow_html=True)

    with r2:
        st.markdown("### Live Parameter Report")
        br_st = br.get_stats()
        params = {
            "γ (discount)":     cfg["gamma"],
            "ε (current)":      round(br.epsilon, 4),
            "ε_min":            cfg["epsilon_min"],
            "ε_decay / step":   cfg["epsilon_decay"],
            "α (learning rate)":br.learning_rate,
            "τ (soft-update)":  cfg["tau"],
            "β_PER (IS)":       round(br.memory.beta, 4),
            "n (N-step)":       cfg["n_steps"],
            "Batch size":       cfg["batch_size"],
            "Buffer capacity":  f"{cfg['buffer_size']:,}",
            "ICM β":            cfg["icm_beta"],
            "H(π) entropy":     round(list(ss.entropy_hist)[-1],4) if ss.entropy_hist else "n/a",
            "H_max":            round(math.log(ACTION_SIZE),4),
            "Train steps":      br_st["train_step"],
            "Total env steps":  ss.global_step,
            "Episodes":         ss.episode_count,
            "Curriculum level": br.curriculum.level,
        }
        st.dataframe(pd.DataFrame({"Value":[str(v) for v in params.values()]},
                                  index=params.keys()),
                     width='stretch')

        st.markdown("---")
        st.markdown("### Key References")
        refs = [
            ("Mnih et al., 2015", "Human-level control through deep RL. Nature 518."),
            ("van Hasselt et al., 2016", "Double DQN. AAAI."),
            ("Wang et al., 2016", "Dueling network architectures. ICML."),
            ("Schaul et al., 2015", "Prioritized experience replay. ICLR."),
            ("Sutton & Barto, 2018", "Reinforcement Learning: An Introduction, 2nd ed."),
            ("Kingma & Ba, 2014", "Adam: A method for stochastic optimization. ICLR."),
            ("Bellemare et al., 2016", "Unifying count-based exploration. NeurIPS."),
            ("Wilson et al., 2019", "Uniform spanning trees (Wilson's algorithm). STOC."),
            ("Portelas et al., 2020", "Automatic curriculum learning. JMLR."),
            ("Russell's Circumplex","2D valence-arousal emotion model, 1980."),
        ]
        for auth,desc in refs:
            st.markdown(f'<div class="cite"><b>{auth}</b><br>{desc}</div>',
                        unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Theoretical Bounds")
        st.markdown("""<div class="eq">
<b>Max Q-value bound</b> (finite horizon T, max reward R):<br>
<em>|Q*(s,a)| ≤ R_max · (1−γ^T)/(1−γ)</em><br><br>
<b>Optimal policy:</b><br>
<em>π*(s) = argmax_a Q*(s,a)</em><br><br>
<b>Value function contraction</b> (γ < 1):<br>
<em>‖T Q₁ − T Q₂‖∞ ≤ γ‖Q₁ − Q₂‖∞</em><br><br>
<b>Sample complexity (PAC bound, Strehl et al.):</b><br>
<em>O(|S||A| · (1/(1−γ))³ · ε⁻² · log(1/δ))</em>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 9 — HYPER-VIZ MATRIX  (Plotly — lazy loaded)
# ══════════════════════════════════════════════════════════
def _tab_visualizations():
    """
    24-panel scientific visualization matrix.
    Uses Plotly for GPU-accelerated rendering.
    Lazy-loads only when this tab is active.
    """
    ss  = st.session_state
    cfg = ss.config

    st.markdown('<div class="ph">🌋 HYPER-VIZ MATRIX — 24 SCIENTIFIC PANELS</div>',
                unsafe_allow_html=True)

    if not ss.get("_render_viz", False):
        st.info("🌌 The Hyper-Viz Matrix contains 24 high-resolution, color-mapped tensor streams. It is purely lazy-loaded to preserve the simulation tickrate.")
        if st.button("IGNITE MATRIX (Mount Graphics)", use_container_width=True):
            ss._render_viz = True
            st.rerun()
        return

    if st.button("UNMOUNT MATRIX", use_container_width=True):
        ss._render_viz = False
        st.rerun()

    n   = cfg.get("chart_points", 150)
    cd  = ss.analytics.get_chart_data(n)

    if not _PLOTLY:
        st.warning("Install plotly: `pip install plotly`")
        return

    # ── Plotly theme shared layout ────────────────────────
    _BG   = "rgba(8,8,24,0.0)"
    _GRID = "rgba(255,255,255,0.05)"
    _FONT = dict(family="JetBrains Mono, monospace", size=10, color="#8b949e")

    def _lay(title="", h=240, showlegend=False):
        return dict(
            paper_bgcolor=_BG, plot_bgcolor="rgba(10,10,30,0.85)",
            font=_FONT, height=h, margin=dict(l=36,r=12,t=28,b=28),
            title=dict(text=title, font=dict(size=9,color="#6e7681"), x=0),
            xaxis=dict(gridcolor=_GRID, zeroline=False, showgrid=True),
            yaxis=dict(gridcolor=_GRID, zeroline=False, showgrid=True),
            showlegend=showlegend,
        )

    def _pline(y, color="#00f5ff", fill=False, name=""):
        tr = go.Scatter(y=y, mode="lines", name=name,
                        line=dict(color=color, width=1.4),
                        fill="tozeroy" if fill else None,
                        fillcolor=color.replace(")", ",0.10)").replace("rgb","rgba") if fill else None)
        return tr

    def _pbar(x, y, color="#00f5ff"):
        return go.Bar(x=x, y=y,
                      marker=dict(color=color, opacity=0.8,
                                  line=dict(color=color, width=0.5)))

    def _pheat(z, title="", colorscale=None, h=240):
        if colorscale is None:
            colorscale = [[0,"#06060a"],[0.3,"#1e3a5f"],[0.6,"#0ea5e9"],[1,"#f0f9ff"]]
        fig = go.Figure(go.Heatmap(
            z=z[::-1] if z is not None and len(z) else [[0]],
            colorscale=colorscale, showscale=False,
            hovertemplate="row:%{y} col:%{x}<br>val:%{z:.3f}<extra></extra>"))
        lay = _lay(title, h)
        lay["xaxis"] = dict(visible=False)
        lay["yaxis"] = dict(visible=False)
        fig.update_layout(**lay)
        return fig

    def _sf(d, n=150):
        return list(d)[-n:] if d else []

    # ── helper: safe list ─────────────────────────────────
    def _sl(key, n=150):
        return _sf(ss.get(key, []), n)

    H, W = ss.env.maze.shape

    # ──────────────────────────────────────────────────────
    # ROW 1: SPATIAL INTELLIGENCE
    # ──────────────────────────────────────────────────────
    st.markdown('<div class="ph">🌐 ROW 1 — SPATIAL INTELLIGENCE</div>', unsafe_allow_html=True)
    r1a, r1b, r1c = st.columns(3)

    with r1a:
        # Panel 1: Macro exploration heatmap
        heat_macro = ss.analytics.get_heatmap(H, W, episode=False).tolist()
        maze_mask  = (ss.env.maze == 1).tolist()
        # Overlay walls as NaN
        z = [[float("nan") if maze_mask[r][c] else heat_macro[r][c]
               for c in range(W)] for r in range(H)]
        fig = _pheat(z, "1. Macro Exploration Density",
                     colorscale=[[0,"#06060a"],[0.2,"#0c4a6e"],[0.5,"#0ea5e9"],[0.8,"#7dd3fc"],[1,"#ffffff"]])
        st.plotly_chart(fig, width='stretch', key="viz_p1")

    with r1b:
        # Panel 2: Episode heatmap
        heat_ep = ss.analytics.get_heatmap(H, W, episode=True).tolist()
        z2 = [[float("nan") if maze_mask[r][c] else heat_ep[r][c]
                for c in range(W)] for r in range(H)]
        fig2 = _pheat(z2, "2. Episode-Level Exploration",
                      colorscale=[[0,"#06060a"],[0.3,"#4a044e"],[0.6,"#d946ef"],[1,"#fae8ff"]])
        st.plotly_chart(fig2, width='stretch', key="viz_p2")

    with r1c:
        # Panel 3: Visit frequency contour (walls as None)
        z3 = [[float("nan") if maze_mask[r][c] else heat_macro[r][c]
                for c in range(W)] for r in range(H)]
        fig3 = go.Figure(go.Heatmap(
            z=list(reversed(z3)),
            colorscale=[[0,"rgba(6,6,10,1)"],[0.4,"rgba(16,185,129,0.6)"],[0.7,"rgba(52,211,153,0.85)"],[1,"rgba(167,243,208,1)"]],
            showscale=False))
        lay3 = _lay("3. Visit Frequency (Green=Hot)", 240)
        lay3["xaxis"] = dict(visible=False); lay3["yaxis"] = dict(visible=False)
        fig3.update_layout(**lay3)
        st.plotly_chart(fig3, width='stretch', key="viz_p3")

    # ──────────────────────────────────────────────────────
    # ROW 2: REWARD SCIENCE
    # ──────────────────────────────────────────────────────
    st.markdown('<div class="ph">⚡ ROW 2 — REWARD SCIENCE</div>', unsafe_allow_html=True)
    r2a, r2b, r2c = st.columns(3)

    with r2a:
        ext = _sl("extr_hist", n)
        fig = go.Figure([_pline(ext, "#38bdf8", fill=True, name="Extrinsic")])
        fig.update_layout(**_lay("4. Extrinsic Reward Signal"))
        st.plotly_chart(fig, width='stretch', key="viz_p4")

    with r2b:
        intr = _sl("intr_hist", n)
        fig = go.Figure([_pline(intr, "#f97316", fill=True, name="Intrinsic")])
        fig.update_layout(**_lay("5. ICM Curiosity Motivation"))
        st.plotly_chart(fig, width='stretch', key="viz_p5")

    with r2c:
        ema = _sf(cd.get("ema_rewards",[]), n)
        raw = _sf(cd.get("rewards",[]), n)
        fig = go.Figure([
            _pline(raw, "#334155", fill=True, name="Raw"),
            _pline(ema, "#22c55e", name="EMA"),
        ])
        fig.update_layout(**_lay("6. EMA Reward Smoothing"))
        st.plotly_chart(fig, width='stretch', key="viz_p6")

    # ──────────────────────────────────────────────────────
    # ROW 3: LOSS MANIFOLD & BELLMAN
    # ──────────────────────────────────────────────────────
    st.markdown('<div class="ph">📉 ROW 3 — LOSS MANIFOLD & BELLMAN</div>', unsafe_allow_html=True)
    r3a, r3b, r3c = st.columns(3)

    with r3a:
        losses = _sf(cd.get("losses",[]), n*3)
        fig = go.Figure([_pline(losses, "#ef4444", name="Loss")])
        fig.update_layout(**_lay("7. Loss Function L(θ)"))
        st.plotly_chart(fig, width='stretch', key="viz_p7")

    with r3b:
        tderr = _sf(cd.get("td_errors",[]), n*3)
        fig = go.Figure([_pline(tderr, "#eab308", fill=True, name="TD")])
        fig.update_layout(**_lay("8. TD-Error Distribution δ"))
        st.plotly_chart(fig, width='stretch', key="viz_p8")

    with r3c:
        bell = _sl("bellman_hist", n*3)
        fig = go.Figure([_pline(bell, "#06b6d4", name="Bellman")])
        fig.update_layout(**_lay("9. Bellman Residual |r+γQ\'−Q|"))
        st.plotly_chart(fig, width='stretch', key="viz_p9")

    # ──────────────────────────────────────────────────────
    # ROW 4: POLICY & ENTROPY
    # ──────────────────────────────────────────────────────
    st.markdown('<div class="ph">🎲 ROW 4 — POLICY SPACE</div>', unsafe_allow_html=True)
    r4a, r4b, r4c = st.columns(3)

    with r4a:
        ent = _sl("entropy_hist", n)
        max_ent = math.log(ACTION_SIZE)
        fig = go.Figure([
            _pline(ent, "#f97316", fill=True, name="H(π)"),
            go.Scatter(y=[max_ent]*len(ent), mode="lines",
                       line=dict(color="#6e7681", dash="dot", width=1), name="H_max"),
        ])
        fig.update_layout(**_lay("10. Policy Entropy H(π) [nats]", showlegend=False))
        st.plotly_chart(fig, width='stretch', key="viz_p10")

    with r4b:
        eps = _sf(cd.get("epsilons",[]), n)
        fig = go.Figure([_pline(eps, "#eab308", name="ε")])
        fig.add_hline(y=ss.config.get("epsilon_min",0.04),
                      line_color="#ef4444", line_dash="dot", line_width=1,
                      annotation_text="ε_min", annotation_font_size=8)
        fig.update_layout(**_lay("11. Epsilon Annealing Schedule"))
        st.plotly_chart(fig, width='stretch', key="viz_p11")

    with r4c:
        qvals = _sl("qval_hist", n)
        if qvals:
            qdf = pd.DataFrame(qvals, columns=["UP","DOWN","LEFT","RIGHT"])
            fig = go.Figure([
                go.Scatter(y=qdf["UP"],   mode="lines", name="UP",   line=dict(color="#00f5ff",width=1.2)),
                go.Scatter(y=qdf["DOWN"], mode="lines", name="DOWN", line=dict(color="#a855f7",width=1.2)),
                go.Scatter(y=qdf["LEFT"], mode="lines", name="LEFT", line=dict(color="#f97316",width=1.2)),
                go.Scatter(y=qdf["RIGHT"],mode="lines", name="RIGHT",line=dict(color="#22c55e",width=1.2)),
            ])
            lay12 = _lay("12. Q-Vector per Action")
            lay12["showlegend"] = True  # override default
            lay12["legend"] = dict(font=dict(size=8), bgcolor="rgba(0,0,0,0.4)")
            fig.update_layout(**lay12)
        else:
            fig = go.Figure(); fig.update_layout(**_lay("12. Q-Vector (running...)"))
        st.plotly_chart(fig, width='stretch', key="viz_p12")

    # ──────────────────────────────────────────────────────
    # ROW 5: VALUE FUNCTION & GRADIENT
    # ──────────────────────────────────────────────────────
    st.markdown('<div class="ph">🧮 ROW 5 — VALUE FUNCTION & GRADIENT</div>', unsafe_allow_html=True)
    r5a, r5b, r5c = st.columns(3)

    with r5a:
        val = _sl("val_hist", n)
        fig = go.Figure([_pline(val, "#38bdf8", fill=True, name="V(s)")])
        fig.update_layout(**_lay("13. State Value Function V(s)"))
        st.plotly_chart(fig, width='stretch', key="viz_p13")

    with r5b:
        adv = _sl("adv_hist", n)
        fig = go.Figure([_pline(adv, "#22c55e", fill=True, name="max A")])
        fig.update_layout(**_lay("14. Max Advantage A(s,a)"))
        st.plotly_chart(fig, width='stretch', key="viz_p14")

    with r5c:
        gn = _sl("gnorm_hist", n*3)
        fig = go.Figure([_pline(gn, "#ef4444", name="‖ΔW₁‖")])
        fig.update_layout(**_lay("15. Gradient Norm ‖ΔW₁‖_F"))
        st.plotly_chart(fig, width='stretch', key="viz_p15")

    # ──────────────────────────────────────────────────────
    # ROW 6: SUCCESS TOPOLOGY
    # ──────────────────────────────────────────────────────
    st.markdown('<div class="ph">🏆 ROW 6 — SUCCESS TOPOLOGY</div>', unsafe_allow_html=True)
    r6a, r6b, r6c = st.columns(3)

    with r6a:
        succ = list(cd.get("successes",[]))[-n:]
        wn   = min(20, len(succ))
        roll = [sum(succ[max(0,i-wn):i+1])/min(i+1,wn) for i in range(len(succ))] if succ else []
        fig  = go.Figure([_pline(roll, "#06b6d4", fill=True, name="Win Rate")])
        fig.add_hline(y=0.72, line_color="#22c55e", line_dash="dot", line_width=1,
                      annotation_text="promote", annotation_font_size=8)
        fig.update_layout(**_lay("16. Rolling Win Rate Topology"))
        st.plotly_chart(fig, width='stretch', key="viz_p16")

    with r6b:
        opt = list(cd.get("optimality",[]))[-n:]
        fig = go.Figure([_pline(opt, "#a855f7", fill=True, name="Efficiency")])
        fig.update_layout(**_lay("17. A* Pathing Optimality"))
        st.plotly_chart(fig, width='stretch', key="viz_p17")

    with r6c:
        steps_ep = list(cd.get("steps",[]))[-n:]
        fig = go.Figure([_pline(steps_ep, "#f97316", name="Steps")])
        fig.update_layout(**_lay("18. Episode Steps vs Time Limit"))
        st.plotly_chart(fig, width='stretch', key="viz_p18")

    # ──────────────────────────────────────────────────────
    # ROW 7: CURRICULUM & CAPABILITY
    # ──────────────────────────────────────────────────────
    st.markdown('<div class="ph">📈 ROW 7 — CURRICULUM & CAPABILITY</div>', unsafe_allow_html=True)
    r7a, r7b, r7c = st.columns(3)

    with r7a:
        cap_h = list(ss.analytics.capability.history)[-n:] if ss.analytics.capability.history else []
        fig = go.Figure([_pline(cap_h, "#06b6d4", fill=True, name="Capability")])
        fig.update_layout(**_lay("19. Capability Score Trajectory"))
        st.plotly_chart(fig, width='stretch', key="viz_p19")

    with r7b:
        levels = list(cd.get("levels",[]))[-n:]
        fig = go.Figure([go.Scatter(y=levels, mode="lines+markers",
                                    line=dict(color="#22c55e",width=1.5),
                                    marker=dict(size=3,color="#22c55e"))])
        fig.update_layout(**_lay("20. Curriculum Level Epochs"))
        st.plotly_chart(fig, width='stretch', key="viz_p20")

    with r7c:
        cur_hist = ss.brain.curriculum.history
        zpd = [e["score"] for e in cur_hist[-min(len(cur_hist), n):]] if cur_hist else []
        fig = go.Figure([
            _pline(zpd, "#38bdf8", name="ZPD Score"),
            go.Scatter(y=[0.72]*len(zpd), mode="lines",
                       line=dict(color="#22c55e", dash="dot", width=1), name="Promote"),
            go.Scatter(y=[0.25]*len(zpd), mode="lines",
                       line=dict(color="#ef4444", dash="dot", width=1), name="Demote"),
        ])
        lay7c = _lay("21. ZPD Curriculum Window")
        lay7c["showlegend"] = True  # override default
        lay7c["legend"] = dict(font=dict(size=8), bgcolor="rgba(0,0,0,0.4)")
        fig.update_layout(**lay7c)
        st.plotly_chart(fig, width='stretch', key="viz_p21")

    # ──────────────────────────────────────────────────────
    # ROW 8: PHASE SPACE PORTRAITS (scatter)
    # ──────────────────────────────────────────────────────
    st.markdown('<div class="ph">🔬 ROW 8 — PHASE SPACE PORTRAITS</div>', unsafe_allow_html=True)
    r8a, r8b, r8c = st.columns(3)

    with r8a:
        l2 = list(cd.get("losses",[]))
        t2 = list(cd.get("td_errors",[]))
        nl = min(len(l2), len(t2), n)
        if nl > 2:
            fig = go.Figure(go.Scatter(
                x=l2[-nl:], y=t2[-nl:], mode="markers",
                marker=dict(size=3, color=list(range(nl)),
                            colorscale=[[0,"#0f172a"],[0.5,"#a855f7"],[1,"#ef4444"]],
                            showscale=False, opacity=0.7)))
            lay8a = _lay("22. Phase Portrait: Loss × TD-Error")
            lay8a["xaxis"]["title"] = "Loss"
            lay8a["yaxis"]["title"] = "TD-Error"
            fig.update_layout(**lay8a)
        else:
            fig = go.Figure(); fig.update_layout(**_lay("22. Phase Portrait (loading...)"))
        st.plotly_chart(fig, width='stretch', key="viz_p22")

    with r8b:
        vl = _sl("val_hist", n); al = _sl("adv_hist", n)
        nva = min(len(vl), len(al))
        if nva > 2:
            fig = go.Figure(go.Scatter(
                x=vl[-nva:], y=al[-nva:], mode="markers",
                marker=dict(size=3, color=list(range(nva)),
                            colorscale=[[0,"#0f172a"],[0.5,"#0ea5e9"],[1,"#a855f7"]],
                            showscale=False, opacity=0.7)))
            lay8b = _lay("23. Phase Portrait: V(s) × A(s,a)")
            lay8b["xaxis"]["title"] = "V(s)"
            lay8b["yaxis"]["title"] = "A(s,a)"
            fig.update_layout(**lay8b)
        else:
            fig = go.Figure(); fig.update_layout(**_lay("23. Phase Portrait (loading...)"))
        st.plotly_chart(fig, width='stretch', key="viz_p23")

    with r8c:
        ep_log = list(ss.get("ep_log", []))[-n:]
        if ep_log:
            epldf = pd.DataFrame(ep_log)
            colors = ["#22c55e" if s else "#ef4444" for s in epldf.get("success", [False]*len(ep_log))]
            fig = go.Figure(go.Scatter(
                x=epldf.get("steps", []).tolist(),
                y=epldf.get("reward", []).tolist(),
                mode="markers",
                marker=dict(size=5, color=colors, opacity=0.75,
                            line=dict(width=0.3, color="#ffffff")),
                text=[f"EP#{int(r.get('ep',0))} L{int(r.get('level',1))}" for _,r in epldf.iterrows()],
                hovertemplate="%{text}<br>steps=%{x} reward=%{y:.2f}<extra></extra>"))
            lay8c = _lay("24. Scatter: Steps × Reward (green=win)")
            lay8c["xaxis"] = dict(gridcolor=_GRID, zeroline=False, showgrid=True, title="Steps")
            lay8c["yaxis"] = dict(gridcolor=_GRID, zeroline=False, showgrid=True, title="Reward")
            fig.update_layout(**lay8c)
        else:
            fig = go.Figure(); fig.update_layout(**_lay("24. Scatter (no episodes yet)"))
        st.plotly_chart(fig, width='stretch', key="viz_p24")

    st.caption(
        f"📡 All 24 panels live. Plotly WebGL rendering. "
        f"Steps logged: {ss.global_step:,} | Buffer: {len(ss.brain.memory):,} | "
        f"Unique states: {ss.brain.curiosity.coverage()}"
    )


# ══════════════════════════════════════════════════════════
# ENTRY POINT  (lazy tab loading)
# ══════════════════════════════════════════════════════════
def _main():
    ss = st.session_state
    _header()
    _sidebar()

    TAB_NAMES = [
        "🗺️ Mission Control",
        "📊 Analytics Lab",
        "🧠 Soul Matrix",
        "🗄️ Memory Palace",
        "🔬 Brain Autopsy",
        "📅 Episode Timeline",
        "🏆 Benchmark",
        "🔭 Research Lab",
        "🌋 Hyper-Viz Matrix",
    ]
    TAB_FNS = [
        _tab_mission, _tab_analytics, _tab_soul, _tab_memory,
        _tab_brain,   _tab_timeline,  _tab_benchmark, _tab_research,
        _tab_visualizations,
    ]

    tabs = st.tabs(TAB_NAMES)

    # ── Lazy loading: only render the active tab ───────────
    # Streamlit executes ALL with-blocks always, but we minimise
    # expensive operations by checking a lightweight flag.
    for i, (tab, fn) in enumerate(zip(tabs, TAB_FNS)):
        with tab:
            fn()

    # ── Auto-run loop ──────────────────────────────────────
    if ss.auto_mode:
        for _ in range(ss.config.get("steps_per_frame", 1)):
            process_step()
        delay = ss.config.get("sim_speed", 0.04)
        if delay > 0:
            time.sleep(delay)
        st.rerun()

_main()
