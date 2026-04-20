"""
app.py — Project A.L.I.V.E. NEXUS v2.0
═══════════════════════════════════════════════════════════════
A.daptive L.earning I.ntelligence V.ia E.volution

Frontend for the multi-agent RL system with:
  • Real-time maze simulation + ASCII rendering
  • Brain Lab: live neural network metrics
  • Soul Interface: contextual AI conversation
  • Memory Palace: episodic/semantic memory viewer
  • Configuration Lab: full hyperparameter control

Run with: streamlit run app.py
═══════════════════════════════════════════════════════════════
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import random
import json
import sys
import os

# ── Local modules ──────────────────────────────────────────
from brain       import AgentBrain
from world       import MazeEnvironment
from soul        import SoulCore
from analytics   import PerformanceDashboard
from memory_palace import MemoryPalace


# ═══════════════════════════════════════════════════════════
#  PAGE CONFIG & GLOBAL CSS
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "A.L.I.V.E. NEXUS",
    page_icon   = "🧿",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

st.markdown("""
<style>
/* ── Root variables ───────────────────────────────────── */
:root {
  --bg-deep:   #080b14;
  --bg-card:   #0d1117;
  --bg-glass:  rgba(15, 20, 40, 0.85);
  --accent-1:  #00e5ff;
  --accent-2:  #7c3aed;
  --accent-3:  #f97316;
  --accent-ok: #10b981;
  --accent-ng: #ef4444;
  --text-main: #e2e8f0;
  --text-dim:  #64748b;
  --border:    rgba(0, 229, 255, 0.15);
  --glow:      0 0 20px rgba(0, 229, 255, 0.25);
  --font-mono: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
  --font-ui:   'Orbitron', 'Space Grotesk', sans-serif;
}

/* ── Base ─────────────────────────────────────────────── */
.stApp {
  background: radial-gradient(ellipse at 20% 50%, #0a0f2e 0%, #080b14 60%, #060a18 100%);
  color: var(--text-main);
}
.stApp::before {
  content: '';
  position: fixed; inset: 0;
  background: repeating-linear-gradient(
    0deg, transparent, transparent 40px,
    rgba(0,229,255,0.012) 40px, rgba(0,229,255,0.012) 41px
  ),
  repeating-linear-gradient(
    90deg, transparent, transparent 60px,
    rgba(0,229,255,0.008) 60px, rgba(0,229,255,0.008) 61px
  );
  pointer-events: none; z-index: 0;
}

/* ── Header ───────────────────────────────────────────── */
.nexus-header {
  text-align: center;
  padding: 1.5rem 0 0.5rem;
  font-family: var(--font-ui);
  letter-spacing: 0.25em;
  font-size: 2.2rem;
  font-weight: 900;
  background: linear-gradient(135deg, #00e5ff, #7c3aed, #f97316);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-shadow: none;
  filter: drop-shadow(0 0 30px rgba(0,229,255,0.4));
}
.nexus-sub {
  text-align: center;
  color: var(--text-dim);
  font-size: 0.8rem;
  letter-spacing: 0.3em;
  font-family: var(--font-mono);
  margin-bottom: 1rem;
}

/* ── Cards ────────────────────────────────────────────── */
.nexus-card {
  background: var(--bg-glass);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.2rem;
  margin-bottom: 1rem;
  backdrop-filter: blur(10px);
  box-shadow: var(--glow), inset 0 1px 0 rgba(255,255,255,0.05);
}
.nexus-card-title {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  letter-spacing: 0.25em;
  color: var(--accent-1);
  text-transform: uppercase;
  margin-bottom: 0.8rem;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.4rem;
}

/* ── Status pills ─────────────────────────────────────── */
.pill {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 20px;
  font-size: 0.7rem;
  font-family: var(--font-mono);
  margin: 2px;
  font-weight: 600;
}
.pill-cyan   { background: rgba(0,229,255,0.15); color: var(--accent-1); border: 1px solid rgba(0,229,255,0.3); }
.pill-violet { background: rgba(124,58,237,0.2); color: #a78bfa;         border: 1px solid rgba(124,58,237,0.4); }
.pill-orange { background: rgba(249,115,22,0.15); color: #fb923c;        border: 1px solid rgba(249,115,22,0.3); }
.pill-green  { background: rgba(16,185,129,0.15); color: var(--accent-ok); border: 1px solid rgba(16,185,129,0.3); }
.pill-red    { background: rgba(239,68,68,0.15);  color: var(--accent-ng); border: 1px solid rgba(239,68,68,0.3); }

/* ── Maze display ─────────────────────────────────────── */
.maze-viewport {
  background: #030509;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.8rem;
  font-family: var(--font-mono);
  font-size: 0.9rem;
  line-height: 1.3;
  overflow: auto;
  max-height: 380px;
  box-shadow: inset 0 0 30px rgba(0,0,0,0.8), var(--glow);
}

/* ── Chat bubbles ─────────────────────────────────────── */
.chat-user {
  background: rgba(249,115,22,0.1);
  border-right: 3px solid var(--accent-3);
  border-radius: 14px 0 0 14px;
  padding: 0.6rem 1rem;
  margin: 0.4rem 0;
  text-align: right;
  font-size: 0.88rem;
}
.chat-ai {
  background: rgba(0,229,255,0.07);
  border-left: 3px solid var(--accent-1);
  border-radius: 0 14px 14px 0;
  padding: 0.6rem 1rem;
  margin: 0.4rem 0;
  font-size: 0.88rem;
}
.chat-meta {
  font-size: 0.68rem;
  color: var(--text-dim);
  font-family: var(--font-mono);
}

/* ── Consciousness stream ─────────────────────────────── */
.thought-bubble {
  background: rgba(124,58,237,0.1);
  border-left: 3px solid var(--accent-2);
  border-radius: 0 8px 8px 0;
  padding: 0.8rem 1rem;
  font-style: italic;
  font-size: 0.85rem;
  color: #c4b5fd;
  margin: 0.5rem 0;
}

/* ── Metrics ──────────────────────────────────────────── */
div[data-testid="stMetric"] {
  background: var(--bg-glass);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.6rem 0.8rem;
  box-shadow: var(--glow);
}
div[data-testid="stMetricLabel"] { color: var(--text-dim) !important; font-size: 0.7rem; }
div[data-testid="stMetricValue"] { color: var(--accent-1) !important; font-family: var(--font-mono); }

/* ── Buttons ──────────────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, rgba(0,229,255,0.15), rgba(124,58,237,0.2));
  color: var(--accent-1);
  border: 1px solid rgba(0,229,255,0.3);
  border-radius: 8px;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  letter-spacing: 0.1em;
  transition: all 0.2s ease;
}
.stButton > button:hover {
  background: linear-gradient(135deg, rgba(0,229,255,0.3), rgba(124,58,237,0.35));
  box-shadow: 0 0 15px rgba(0,229,255,0.35);
  transform: translateY(-1px);
}

/* ── Sidebar ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a0d1e 0%, #060a18 100%);
  border-right: 1px solid var(--border);
}

/* ── Progress bars ────────────────────────────────────── */
.stProgress > div > div > div { background: linear-gradient(90deg, #00e5ff, #7c3aed) !important; }

/* ── Tabs ─────────────────────────────────────────────── */
.stTabs [data-baseweb="tab"] {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  letter-spacing: 0.1em;
  color: var(--text-dim);
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
  color: var(--accent-1);
  border-bottom: 2px solid var(--accent-1);
}

/* ── Scrollbar ────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--accent-2); border-radius: 3px; }

/* ── Text & code ──────────────────────────────────────── */
.stTextArea textarea { 
  font-family: var(--font-mono) !important; 
  font-size: 0.78rem !important;
  background: #030509 !important;
  border: 1px solid var(--border) !important;
  color: var(--text-main) !important;
}
code, pre { font-family: var(--font-mono) !important; }

@keyframes pulse-glow {
  0%, 100% { box-shadow: 0 0 8px rgba(0,229,255,0.2); }
  50% { box-shadow: 0 0 20px rgba(0,229,255,0.5); }
}
.alive-pulse { animation: pulse-glow 2.5s ease-in-out infinite; }
</style>

<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════
def _build_config() -> dict:
    return {
        # RL Brain
        'h1': 256, 'h2': 128, 'h3': 64,
        'buffer_size': 50_000, 'batch_size': 64,
        'gamma': 0.99, 'n_steps': 3,
        'lr': 0.001, 'tau': 0.005,
        'epsilon_min': 0.04, 'epsilon_decay': 0.997,
        'icm_beta': 0.05,
        # Environment
        'maze_h': 11, 'maze_w': 13,
        'algorithm': 'backtracker',
        'fog': False, 'dynamic': False, 'portals': False,
        # Simulation
        'sim_speed': 0.08,
        'auto_curriculum': True,
        'steps_per_tick': 1,
    }


def init_state():
    cfg = _build_config()
    st.session_state.config = cfg

    env = MazeEnvironment(cfg)
    st.session_state.env = env

    brain = AgentBrain(
        state_size  = MazeEnvironment.STATE_SIZE,
        action_size = MazeEnvironment.ACTION_SIZE,
        config      = cfg
    )
    st.session_state.brain = brain

    soul = SoulCore(name="Nik", seed=7)
    st.session_state.soul = soul

    dashboard = PerformanceDashboard()
    st.session_state.dash = dashboard

    memory = MemoryPalace()
    st.session_state.mem = memory

    # Simulation state
    st.session_state.auto_mode     = False
    st.session_state.current_state = env.reset()
    st.session_state.episode_done  = False
    st.session_state.total_reward  = 0.0
    st.session_state.tick          = 0

    # Try loading saved state
    saved = memory.load_all()
    if saved and 'brain_weights' in saved:
        try:
            brain.set_weights(saved['brain_weights'])
            # Restore relationship
            soul.relationship.score = saved.get('soul', {}).get('relationship', 50)
            st.toast("💾 Memory loaded from previous session!", icon="🧠")
        except Exception:
            pass


if 'brain' not in st.session_state:
    init_state()


# ═══════════════════════════════════════════════════════════
#  SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════
def do_reset(new_config: dict = None):
    """Reset environment for a new episode."""
    env:   MazeEnvironment    = st.session_state.env
    brain: AgentBrain         = st.session_state.brain
    mem:   MemoryPalace       = st.session_state.mem
    cfg = st.session_state.config

    # Apply curriculum if auto
    if cfg.get('auto_curriculum', True):
        cur_cfg = brain.curriculum.config
        env_cfg = dict(cur_cfg)
    else:
        env_cfg = {
            'maze_h': cfg['maze_h'], 'maze_w': cfg['maze_w'],
            'algorithm': cfg['algorithm'],
            'fog': cfg['fog'], 'dynamic': cfg['dynamic'], 'portals': cfg['portals'],
        }
    if new_config:
        env_cfg.update(new_config)

    state = env.reset(config=env_cfg, seed=random.randint(0, 2**30))
    st.session_state.current_state = state
    st.session_state.episode_done  = False
    st.session_state.total_reward  = 0.0

    mem.start_episode(
        maze_seed  = env.seed,
        maze_alg   = env.algorithm,
        maze_h     = env.maze_h,
        maze_w     = env.maze_w,
        level      = brain.curriculum.level,
        epsilon    = brain.epsilon,
        max_steps  = env.max_steps,
    )


def do_step():
    """Execute one RL step."""
    env:   MazeEnvironment    = st.session_state.env
    brain: AgentBrain         = st.session_state.brain
    soul:  SoulCore            = st.session_state.soul
    dash:  PerformanceDashboard = st.session_state.dash
    mem:   MemoryPalace       = st.session_state.mem

    if st.session_state.episode_done:
        do_reset()
        return

    state = st.session_state.current_state
    action = brain.act(state)
    next_state, reward, done, info = env.step(action)

    loss, td_err = brain.step(state, action, reward, next_state, done)

    st.session_state.current_state = next_state
    st.session_state.total_reward += reward
    st.session_state.tick += 1

    # Dashboard
    dash.record_step(env.agent_r, env.agent_c, loss, td_err, brain.epsilon)

    # Memory
    mem.record_transition(state, action, reward, next_state, done)

    # Soul update
    env_info_for_soul = {
        'reward':       reward,
        'reached':      info.get('reached', False),
        'timeout':      info.get('timeout', False),
        'trap_hit':     info.get('trap_hit', False),
        'trap_nearby':  any(abs(t.r - env.agent_r) + abs(t.c - env.agent_c) < 4 for t in env.traps),
        'is_new_cell':  env.visit_grid[env.agent_r, env.agent_c] == 1.0,
        'success_rate': env.success_count / max(env.total_episodes, 1),
        'cells_visited': len(env.cells_visited),
        'maze_size':    f"{env.maze_h}×{env.maze_w}",
    }
    soul.update_from_rl(brain.get_stats(), env_info_for_soul)

    if done:
        st.session_state.episode_done = True
        success = info.get('reached', False)

        # Curriculum update
        brain.curriculum.record(success, env.step_count, env.max_steps, st.session_state.total_reward)

        # Dashboard episode record
        ep_info = {
            'optimality': info.get('optimality', 0.0),
            'fog_coverage': info.get('fog_coverage', 1.0),
            'level': brain.curriculum.level,
            'success_count': env.success_count,
            'cells_visited': len(env.cells_visited),
        }
        dash.record_episode(
            st.session_state.total_reward, env.step_count, success,
            ep_info, brain.curriculum.level, env.maze_h, env.maze_w, env.maze
        )

        # Memory episode end
        mem.end_episode(
            total_reward  = st.session_state.total_reward,
            steps         = env.step_count,
            success       = success,
            cells_visited = len(env.cells_visited),
            astar_optimal = env.astar_optimal,
            fog           = env.use_fog,
            traps         = env.use_dynamic,
            td_error      = brain.avg_td_error,
            epsilon       = brain.epsilon,
        )


# ═══════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="nexus-header">▸ A.L.I.V.E. NEXUS</div>', unsafe_allow_html=True)
st.markdown('<div class="nexus-sub">ADAPTIVE · LEARNING · INTELLIGENCE · VIA · EVOLUTION  |  v2.0</div>',
            unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🎛️ Control Matrix")

    brain: AgentBrain = st.session_state.brain
    env:   MazeEnvironment = st.session_state.env
    cfg = st.session_state.config

    # ── Live Stats ───────────────────────────────────────
    st.markdown("**LIVE METRICS**")
    cur_stats = brain.curriculum.get_stats()
    c1, c2 = st.columns(2)
    c1.metric("Level",     f"{cur_stats['level']}/10")
    c2.metric("ε-explore", f"{brain.epsilon:.3f}")
    c1.metric("Wins",      env.success_count)
    c2.metric("Episodes",  env.total_episodes)

    st.progress(cur_stats['zpd_progress'], text=f"Next Level: {cur_stats['zpd_progress']*100:.0f}%")

    st.divider()

    # ── Simulation Control ───────────────────────────────
    st.markdown("**SIMULATION**")
    auto = st.toggle("▶ Auto-Run", value=st.session_state.auto_mode)
    st.session_state.auto_mode = auto

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⏭ Step"):
            do_step()
            st.rerun()
    with col_b:
        if st.button("↺ Reset"):
            do_reset()
            st.rerun()

    steps_per_tick = st.slider("Steps/Tick", 1, 20, cfg.get('steps_per_tick', 1))
    cfg['steps_per_tick'] = steps_per_tick
    sim_speed = st.slider("Speed (s/tick)", 0.01, 0.5, cfg.get('sim_speed', 0.08), step=0.01)
    cfg['sim_speed'] = sim_speed

    st.divider()

    # ── Curriculum ───────────────────────────────────────
    st.markdown("**CURRICULUM**")
    auto_curr = st.toggle("Auto-Curriculum (ZPD)", value=cfg.get('auto_curriculum', True))
    cfg['auto_curriculum'] = auto_curr

    if not auto_curr:
        alg = st.selectbox("Algorithm", ['backtracker', 'prim', 'wilson', 'hybrid'],
                           index=['backtracker','prim','wilson','hybrid'].index(cfg['algorithm']))
        cfg['algorithm'] = alg
        maze_h = st.slider("Maze Height", 7, 41, cfg['maze_h'], step=2)
        maze_w = st.slider("Maze Width",  9, 45, cfg['maze_w'], step=2)
        cfg['maze_h'] = maze_h
        cfg['maze_w'] = maze_w
        cfg['fog']     = st.toggle("Fog of War",        value=cfg['fog'])
        cfg['dynamic'] = st.toggle("Dynamic Traps",     value=cfg['dynamic'])
        cfg['portals'] = st.toggle("Teleport Portals",  value=cfg['portals'])

    st.divider()

    # ── Save / Load ──────────────────────────────────────
    st.markdown("**PERSISTENCE**")
    col_s, col_l = st.columns(2)
    with col_s:
        if st.button("💾 Save"):
            mem: MemoryPalace = st.session_state.mem
            soul: SoulCore    = st.session_state.soul
            dash: PerformanceDashboard = st.session_state.dash
            ok = mem.save_all(brain.get_weights(), dash.get_chart_data(50), soul.get_status())
            st.toast("✅ Saved!" if ok else "❌ Save failed")
    with col_l:
        if st.button("📂 Load"):
            mem: MemoryPalace = st.session_state.mem
            data = mem.load_all()
            if data and 'brain_weights' in data:
                brain.set_weights(data['brain_weights'])
                st.toast("✅ Weights loaded!")
            else:
                st.toast("No save found.")

    st.divider()
    soul_s = st.session_state.soul.get_status()
    st.markdown(f"**SOUL STATE**")
    st.markdown(f"`{soul_s['mood_emoji']}` **{soul_s['mood']}** | 💙 {soul_s['relationship']}")
    st.caption(f"*{soul_s['stage']}* — {soul_s['personality']}")

    st.markdown('<br>', unsafe_allow_html=True)
    st.caption("PROJECT A.L.I.V.E. NEXUS v2.0")


# ═══════════════════════════════════════════════════════════
#  MAIN TABS
# ═══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧿 SIMULATION",
    "🧠 BRAIN LAB",
    "💬 SOUL INTERFACE",
    "🏛️ MEMORY PALACE",
    "⚙️ CONFIG LAB"
])


# ═══════════════════════════════════════════════════════════
#  TAB 1 — SIMULATION
# ═══════════════════════════════════════════════════════════
with tab1:
    env   = st.session_state.env
    brain = st.session_state.brain
    dash  = st.session_state.dash
    soul  = st.session_state.soul

    col_maze, col_info = st.columns([3, 2], gap="medium")

    # ── Left: Maze Viewport ──────────────────────────────
    with col_maze:
        st.markdown('<div class="nexus-card-title">◈ WORLD VIEWPORT</div>', unsafe_allow_html=True)

        # Render maze
        maze_str, legend = env.render_ascii()
        st.markdown(f'<div class="maze-viewport alive-pulse"><pre>{maze_str}</pre></div>',
                    unsafe_allow_html=True)
        st.caption(legend)

        # Episode progress
        prog = env.step_count / max(env.max_steps, 1)
        color = "normal" if prog < 0.7 else ("off" if prog < 0.9 else "inverse")
        st.progress(min(prog, 1.0), text=f"Episode: {env.step_count}/{env.max_steps} steps")

        # Status pills
        cur_cfg = brain.curriculum.config
        pills_html = f"""
        <div style='margin-top:0.5rem;'>
          <span class='pill pill-cyan'>ALG: {cur_cfg['algorithm'].upper()}</span>
          <span class='pill pill-violet'>LVL: {brain.curriculum.level}/10</span>
          <span class='pill pill-orange'>ε: {brain.epsilon:.3f}</span>
          <span class='pill pill-green'>WINS: {env.success_count}</span>
          {'<span class="pill pill-red">🌫 FOG</span>' if cur_cfg['fog'] else ''}
          {'<span class="pill pill-red">💀 TRAPS</span>' if cur_cfg['dynamic'] else ''}
          {'<span class="pill pill-violet">🌀 PORTALS</span>' if cur_cfg['portals'] else ''}
        </div>"""
        st.markdown(pills_html, unsafe_allow_html=True)

    # ── Right: Brain & Soul Status ───────────────────────
    with col_info:
        # Soul thought
        st.markdown('<div class="nexus-card-title">◈ CONSCIOUSNESS STREAM</div>', unsafe_allow_html=True)
        soul_status = soul.get_status()
        st.markdown(f'<div class="thought-bubble">{soul_status["thought"]}</div>',
                    unsafe_allow_html=True)

        # Live metrics
        st.markdown('<div class="nexus-card-title">◈ BRAIN METRICS</div>', unsafe_allow_html=True)
        brain_stats = brain.get_stats()
        live_stats  = dash.get_live_stats()

        m1, m2, m3 = st.columns(3)
        m1.metric("Avg Reward",   f"{brain_stats['avg_reward']:.2f}")
        m2.metric("Avg Loss",     f"{brain_stats['avg_loss']:.4f}")
        m3.metric("TD-Error",     f"{brain_stats['avg_td_error']:.3f}")

        m4, m5, m6 = st.columns(3)
        m4.metric("Train Steps",  brain_stats['train_step'])
        m5.metric("Memory",       f"{brain_stats['memory_size']:,}")
        m6.metric("LR",           f"{brain_stats['lr']:.5f}")

        # Convergence status
        conv = live_stats['convergence']
        conv_icon = live_stats['convergence_icon']
        conv_desc = live_stats['convergence_desc']
        st.markdown(f"""
        <div class='nexus-card' style='margin-top:0.5rem;'>
          <div class='nexus-card-title'>◈ LEARNING STATUS</div>
          <b style='color:var(--accent-1);font-size:1.1rem;'>{conv_icon} {conv.upper()}</b>
          <br><span style='font-size:0.78rem;color:var(--text-dim);'>{conv_desc}</span>
        </div>""", unsafe_allow_html=True)

        # Curriculum ZPD bar
        st.markdown(f"**Curriculum Level {brain.curriculum.level}/10**")
        st.progress(brain.curriculum.zpd_progress,
                    text=f"ZPD Progress → promote at {brain.curriculum.promote_thresh*100:.0f}%")

        # Capability score
        cap = live_stats['capability']
        cap_trend = live_stats['capability_trend']
        st.metric("🎯 Capability Score", f"{cap:.1f}/100", delta=cap_trend,
                  help="Composite metric: success rate + optimality + exploration + convergence + level")

    # ── Chart strip ──────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="nexus-card-title">◈ PERFORMANCE TIMELINE</div>', unsafe_allow_html=True)
    chart_data = dash.get_chart_data(100)

    if chart_data['rewards']:
        df = pd.DataFrame({
            'Reward (EMA)': chart_data['ema_rewards'],
            'Raw Reward':   chart_data['rewards'],
        })
        st.line_chart(df, height=160, use_container_width=True)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if chart_data['steps']:
            st.line_chart(pd.DataFrame({'Steps/Episode': chart_data['steps']}), height=120)
    with col_c2:
        if chart_data['epsilons']:
            st.line_chart(pd.DataFrame({'Epsilon': chart_data['epsilons'][-200:]}), height=120)


# ═══════════════════════════════════════════════════════════
#  TAB 2 — BRAIN LAB
# ═══════════════════════════════════════════════════════════
with tab2:
    brain = st.session_state.brain
    dash  = st.session_state.dash

    st.markdown("### 🧠 Neural Network Deep Analysis")
    col_b1, col_b2 = st.columns([2, 1])

    with col_b1:
        chart_data = dash.get_chart_data(200)

        st.markdown("**Loss History**")
        if chart_data['losses']:
            loss_df = pd.DataFrame({'Loss': chart_data['losses'][-300:]})
            st.line_chart(loss_df, height=150)

        st.markdown("**TD-Error History**")
        if chart_data['td_errors']:
            td_df = pd.DataFrame({'TD-Error': chart_data['td_errors'][-300:]})
            st.line_chart(td_df, height=150)

        st.markdown("**Reward Distribution (last 100 episodes)**")
        if len(chart_data['rewards']) >= 2:
            reward_df = pd.DataFrame({'Reward': chart_data['rewards'][-100:]})
            st.bar_chart(reward_df, height=150)

    with col_b2:
        st.markdown("**Brain Architecture**")
        st.markdown(f"""
        <div class='nexus-card'>
          <div class='nexus-card-title'>NETWORK TOPOLOGY</div>
          <pre style='font-size:0.7rem;color:var(--accent-1);'>
Input   [{brain.state_size}]
   ↓ ReLU
H1      [{brain.online_net.W2.shape[0]}]  He init
   ↓ ReLU  
H2      [{brain.online_net.W3.shape[0]}]  Adam
   ↓ ReLU
H3      [{brain.online_net.W_val.shape[0]}]  clip±10
   ↓ Dueling
V(s)    [1]   A(s,a)  [{brain.action_size}]
   ↓     ↓
Q(s,a) = V + A - mean(A)
  [{brain.action_size} actions]</pre>
          <br>
          <span class='pill pill-cyan'>DDQN</span>
          <span class='pill pill-violet'>PER</span>
          <span class='pill pill-orange'>N-Step={brain.n_step.n}</span>
          <span class='pill pill-green'>ICM</span>
          <br><br>
          <b style='color:var(--text-dim);font-size:0.7rem;'>Soft update τ={brain.tau}</b><br>
          <b style='color:var(--text-dim);font-size:0.7rem;'>PER α=0.6, β→1.0</b><br>
          <b style='color:var(--text-dim);font-size:0.7rem;'>Adam β₁=0.9, β₂=0.999</b>
        </div>""", unsafe_allow_html=True)

        st.markdown("**Intrinsic Curiosity**")
        uniq = brain.curiosity.coverage()
        st.metric("Unique States Discovered", uniq)
        st.metric("Exploration Bonus β", brain.curiosity.beta)

        st.markdown("**Replay Buffer**")
        buf_fill = len(brain.memory) / brain.memory.capacity
        st.progress(buf_fill, text=f"{len(brain.memory):,} / {brain.memory.capacity:,}")
        st.metric("Buffer β (IS weight)", f"{brain.memory.beta:.3f}")

        st.markdown("**Network Weights (Layer 1 sample)**")
        w1 = brain.online_net.W1[:, :8].flatten()[:20]
        weight_df = pd.DataFrame({'W1 weights': w1})
        st.bar_chart(weight_df, height=100)

    # Curriculum history
    st.markdown("---")
    st.markdown("**Curriculum Learning History**")
    if brain.curriculum.history:
        n = min(len(brain.curriculum.history), 100)
        h = brain.curriculum.history[-n:]
        cur_df = pd.DataFrame({
            'Score':         [e['score'] for e in h],
            'Level':         [e['level'] for e in h],
        })
        st.line_chart(cur_df, height=150)
    else:
        st.info("Play episodes to see curriculum history.")

    col_c1, col_c2, col_c3 = st.columns(3)
    col_c1.metric("Promotions", brain.curriculum.promotions)
    col_c2.metric("Demotions",  brain.curriculum.demotions)
    col_c3.metric("Avg Score",  f"{brain.curriculum.avg_score:.3f}")


# ═══════════════════════════════════════════════════════════
#  TAB 3 — SOUL INTERFACE
# ═══════════════════════════════════════════════════════════
with tab3:
    soul: SoulCore = st.session_state.soul

    col_soul, col_chat = st.columns([1, 2])

    with col_soul:
        st.markdown("### 🌀 A.L.I.V.E. Identity Core")
        status = soul.get_status()

        # Emotion display
        v = status['valence']
        a = status['arousal']
        st.markdown(f"""
        <div class='nexus-card alive-pulse'>
          <div class='nexus-card-title'>◈ EMOTION STATE</div>
          <div style='text-align:center;font-size:2.5rem;margin:0.5rem 0;'>{status['mood_emoji']}</div>
          <div style='text-align:center;font-size:1rem;color:var(--accent-1);font-family:var(--font-mono);'>
            {status['mood'].upper()}
          </div>
          <br>
          <b>Valence</b> (pleasure): <code>{v:+.3f}</code><br>
          <b>Arousal</b> (energy):   <code>{a:+.3f}</code><br>
          <b>Intensity</b>:          <code>{status['intensity']:.3f}</code>
        </div>""", unsafe_allow_html=True)

        # Relationship
        st.markdown(f"""
        <div class='nexus-card'>
          <div class='nexus-card-title'>◈ RELATIONSHIP</div>
          <b style='color:var(--accent-1);'>{status['stage']}</b><br>
          <span style='font-size:0.78rem;color:var(--text-dim);'>{status['stage_desc']}</span><br><br>
          Bond score: <code>{status['relationship']}</code>
        </div>""", unsafe_allow_html=True)
        st.progress(status['relationship'] / 100)

        # Big Five
        st.markdown("**Personality Traits (OCEAN)**")
        ocean = pd.DataFrame({
            'Trait': ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'],
            'Score': [status['O'], status['C'], status['E'], status['A'], status['N']]
        })
        st.bar_chart(ocean.set_index('Trait'), height=160)

        # Memories
        st.markdown("**Emotional Memory Highlights**")
        for m in status.get('strongest_memories', []):
            st.caption(f"• {m}")

    with col_chat:
        st.markdown("### 💬 Cognitive Interface")

        # Inner monologue
        st.markdown(f'<div class="thought-bubble">💭 {status["thought"]}</div>',
                    unsafe_allow_html=True)

        # Chat history
        chat_container = st.container(height=400)
        with chat_container:
            for msg in soul.get_chat_history():
                if msg['role'] == 'user':
                    st.markdown(f"""
                    <div class='chat-user'>
                      <b>YOU</b> <span class='chat-meta'>intent: {msg.get('intent','?')}</span><br>
                      {msg['text']}
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='chat-ai'>
                      <b>A.L.I.V.E.</b> <span class='chat-meta'>mood: {msg.get('emotion','?')}</span><br>
                      {msg['text']}
                    </div>""", unsafe_allow_html=True)

        # Input
        user_input = st.chat_input("Speak to A.L.I.V.E. ...")
        if user_input:
            response = soul.chat(user_input)
            st.rerun()

        # Quick-fire prompts
        st.markdown("**Quick Prompts**")
        q1, q2, q3, q4 = st.columns(4)
        prompts = ["Hello!", "How do you feel?", "What are you learning?", "Who are you?"]
        for col, prompt in zip([q1, q2, q3, q4], prompts):
            if col.button(prompt, use_container_width=True):
                soul.chat(prompt)
                st.rerun()


# ═══════════════════════════════════════════════════════════
#  TAB 4 — MEMORY PALACE
# ═══════════════════════════════════════════════════════════
with tab4:
    mem: MemoryPalace = st.session_state.mem

    st.markdown("### 🏛️ Memory Palace")
    full_status = mem.get_full_status()

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("**📖 Episodic Memory**")
        ep_stats = full_status['episodic_stats']
        if ep_stats:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Episodes", ep_stats.get('total_stored', 0))
            m2.metric("Success Rate",   f"{ep_stats.get('success_rate',0)*100:.1f}%")
            m3.metric("Landmarks",      ep_stats.get('landmarks', 0))
            m4, m5 = st.columns(2)
            m4.metric("Max Level",      ep_stats.get('max_level_reached', 1))
            m5.metric("Best Efficiency",f"{ep_stats.get('avg_efficiency',0)*100:.1f}%")

        st.markdown("**Recent Episodes**")
        recent = full_status.get('episodic_recent', [])
        if recent:
            recent_df = pd.DataFrame(recent)[['episode_id','curriculum_level','total_reward','success','steps','efficiency','maze_alg']].tail(8)
            recent_df.columns = ['EP#','LVL','REWARD','WIN','STEPS','EFFIC','ALG']
            recent_df['WIN'] = recent_df['WIN'].map({True: '✅', False: '❌'})
            st.dataframe(recent_df, use_container_width=True, hide_index=True)
        else:
            st.info("No episodes recorded yet. Start simulation!")

        st.markdown("**🌟 Landmark Episodes**")
        landmarks = full_status.get('landmark_episodes', [])
        for lm in landmarks:
            st.markdown(f"""
            <div class='nexus-card'>
              <b>EP#{lm['episode_id']}</b> — Level {lm['curriculum_level']} — {lm['maze_alg']} maze<br>
              Reward: <code>{lm['total_reward']}</code> | Efficiency: <code>{lm['efficiency']:.2%}</code> |
              {'✅ SUCCESS' if lm['success'] else '❌ FAILED'}
            </div>""", unsafe_allow_html=True)

    with col_m2:
        st.markdown("**🧬 Semantic Memory (World Model)**")
        facts = full_status.get('semantic_facts', [])
        if facts:
            facts_df = pd.DataFrame(facts)[['key','value','confidence','source','accesses']].head(12)
            facts_df.columns = ['FACT','VALUE','CONF','SOURCE','RECALLS']
            st.dataframe(facts_df, use_container_width=True, hide_index=True)
        else:
            st.info("Semantic memory empty. Play more episodes to generate inferences.")

        st.markdown("**💡 Insights**")
        for insight in full_status.get('insights', []):
            st.markdown(f"• {insight}")

        st.markdown("**💾 Persistence**")
        st.markdown(f"""
        <div class='nexus-card'>
          <div class='nexus-card-title'>◈ STORAGE</div>
          Path: <code>{full_status['save_path']}</code><br>
          Size: <code>{full_status['save_size_kb']:.1f} KB</code><br>
          Loaded from disk: <code>{'YES' if full_status['loaded_from_disk'] else 'NO'}</code><br>
          Episode count: <code>{full_status['total_episodes']}</code>
        </div>""", unsafe_allow_html=True)

        st.markdown("**📊 Session Report**")
        report = st.session_state.dash.get_session_report()
        st.text(report)

        col_e1, col_e2 = st.columns(2)
        if col_e1.button("📋 Export JSON"):
            export = st.session_state.dash.export_json()
            st.download_button("⬇ Download", export, "alive_session.json", "application/json")
        if col_e2.button("🗑 Clear Memory"):
            mem.episodic.episodes.clear()
            mem.semantic.facts.clear()
            st.toast("Memory cleared.")
            st.rerun()


# ═══════════════════════════════════════════════════════════
#  TAB 5 — CONFIG LAB
# ═══════════════════════════════════════════════════════════
with tab5:
    st.markdown("### ⚙️ Configuration Laboratory")
    cfg = st.session_state.config
    brain = st.session_state.brain

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

    with col_cfg1:
        st.markdown("**🧠 Neural Network**")
        new_h1 = st.slider("Hidden Layer 1",  64, 512, cfg['h1'], step=64)
        new_h2 = st.slider("Hidden Layer 2",  32, 256, cfg['h2'], step=32)
        new_h3 = st.slider("Hidden Layer 3",  16, 128, cfg['h3'], step=16)
        cfg['h1'], cfg['h2'], cfg['h3'] = new_h1, new_h2, new_h3

        st.markdown("**🎯 Training**")
        cfg['lr']           = st.number_input("Learning Rate", 1e-5, 0.01, cfg['lr'], format="%.5f")
        cfg['gamma']        = st.slider("Gamma (γ)", 0.8, 0.999, cfg['gamma'], step=0.001)
        cfg['tau']          = st.number_input("Soft Update τ", 0.001, 0.1, cfg['tau'], format="%.4f")
        cfg['batch_size']   = st.selectbox("Batch Size", [32, 64, 128, 256], index=1)
        cfg['n_steps']      = st.slider("N-Step Returns", 1, 10, cfg['n_steps'])

    with col_cfg2:
        st.markdown("**🎲 Exploration**")
        cfg['epsilon_min']   = st.slider("Epsilon Min",   0.01, 0.2,  cfg['epsilon_min'],  step=0.01)
        cfg['epsilon_decay'] = st.slider("Epsilon Decay", 0.99, 0.9999, cfg['epsilon_decay'], step=0.0001)
        cfg['icm_beta']      = st.slider("ICM Beta (curiosity)", 0.0, 0.5, cfg['icm_beta'], step=0.01)

        st.markdown("**🗂 Memory**")
        cfg['buffer_size']  = st.selectbox("Buffer Capacity", [10_000, 25_000, 50_000, 100_000], index=2)

        st.markdown("**⚡ Apply Config**")
        if st.button("🔄 Rebuild Brain (new config)", use_container_width=True):
            st.session_state.brain = AgentBrain(
                state_size  = MazeEnvironment.STATE_SIZE,
                action_size = MazeEnvironment.ACTION_SIZE,
                config      = cfg
            )
            do_reset()
            st.success("Brain rebuilt with new architecture!")
            st.rerun()

    with col_cfg3:
        st.markdown("**📐 Current Architecture Summary**")
        bn = st.session_state.brain.online_net
        total_params = sum(
            np.prod(getattr(bn, p).shape)
            for p in ['W1','b1','W2','b2','W3','b3','W_val','b_val','W_adv','b_adv']
        )
        st.markdown(f"""
        <div class='nexus-card'>
          <div class='nexus-card-title'>◈ PARAMETER COUNT</div>
          <code style='font-size:1.2rem;color:var(--accent-1);'>{total_params:,}</code>
          <br><br>
          <b>Input:  </b><code>{brain.state_size}</code><br>
          <b>H1:     </b><code>{bn.W2.shape[0]}</code> neurons<br>
          <b>H2:     </b><code>{bn.W3.shape[0]}</code> neurons<br>
          <b>H3:     </b><code>{bn.W_val.shape[0]}</code> neurons<br>
          <b>Actions:</b><code>{brain.action_size}</code><br>
          <br>
          <b>Optimizer:</b> Adam<br>
          <b>Activation:</b> Leaky ReLU<br>
          <b>Init:</b> He Normal<br>
          <b>Heads:</b> V(s) + A(s,a) Dueling
        </div>""", unsafe_allow_html=True)

        st.markdown("**State Space Breakdown**")
        state_desc = pd.DataFrame({
            'Component':    ['Local vision (3×3)', 'Agent pos (r,c)', 'Target pos (r,c)',
                            'Manhattan dist', 'Trap dist', 'Fog coverage', 'Time pressure'],
            'Dimensions': [9, 2, 2, 1, 1, 1, 1]
        })
        st.dataframe(state_desc, hide_index=True, use_container_width=True)
        st.caption(f"**Total state size: {brain.state_size}**")


# ═══════════════════════════════════════════════════════════
#  AUTO-RUN LOOP (bottom of script — always runs)
# ═══════════════════════════════════════════════════════════
if st.session_state.auto_mode:
    steps = st.session_state.config.get('steps_per_tick', 1)
    for _ in range(steps):
        do_step()
    time.sleep(st.session_state.config.get('sim_speed', 0.08))
    st.rerun()
