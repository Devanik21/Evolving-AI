# streamlit_ai_friend_app.py
# A Streamlit app that embeds an HTML/JS animated AI friend (green smiley) which follows the mouse,
# supports three behavior states (idle, following, playing), logs episodes in JSONL, and provides
# feedback buttons to teach the friend (Good!/Bad!).
#
# Usage:
#   pip install streamlit
#   (optional) pip install fastapi uvicorn
#   streamlit run streamlit_ai_friend_app.py
#
# Notes:
# - The interactive animation and learning live inside the embedded HTML (canvas + JS).
# - Episodes can be saved to the Streamlit-side file data/episodes.jsonl either via postMessage
#   (when using the "Save to Streamlit" button inside the component) or via a local API
#   if FastAPI/uvicorn are installed and started. The embedded app will attempt to POST to
#   http://localhost:8001/api/episodes when the API is available.

import streamlit as st
import json
import os
from pathlib import Path
from typing import List

st.set_page_config(page_title="AI Friend — Streamlit", layout="wide")

DATA_DIR = Path("data")
EPISODES_FILE = DATA_DIR / "episodes.jsonl"
DATA_DIR.mkdir(exist_ok=True)

# Try to start a small FastAPI server to expose endpoints if FastAPI & uvicorn are available.
use_api = False
api_server_thread = None

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, PlainTextResponse
    import uvicorn
    from threading import Thread

    app = FastAPI()

    def read_episodes_file() -> List[dict]:
        if not EPISODES_FILE.exists():
            return []
        with EPISODES_FILE.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return [json.loads(l) for l in lines]

    @app.get("/api/episodes")
    async def get_episodes():
        return read_episodes_file()

    @app.get("/api/stats")
    async def get_stats():
        eps = read_episodes_file()
        # crude stats: number of episodes and last feedback counts
        total = len(eps)
        good = sum(1 for e in eps if e.get("feedback") == "good")
        bad = sum(1 for e in eps if e.get("feedback") == "bad")
        return {"total_episodes": total, "good": good, "bad": bad}

    @app.delete("/api/episodes")
    async def delete_episodes():
        if EPISODES_FILE.exists():
            EPISODES_FILE.unlink()
        return {"ok": True}

    @app.post("/api/episodes")
    async def post_episodes(request: Request):
        payload = await request.json()
        incoming = payload.get("episodes")
        if not incoming:
            return JSONResponse({"error": "no episodes provided"}, status_code=400)
        # incoming is expected to be a list of episode objects
        with EPISODES_FILE.open("a", encoding="utf-8") as f:
            for ep in incoming:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")
        return {"saved": len(incoming)}

    def run_api():
        uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")

    # start uvicorn in a background thread
    api_server_thread = Thread(target=run_api, daemon=True)
    api_server_thread.start()
    use_api = True
except Exception:
    use_api = False


st.title("AI Friend — Streamlit (Green Smiley)")

col1, col2 = st.columns([3, 1])

# HTML/JS component that contains the full interactive canvas and learning logic.
# It will attempt to POST to the local API (if available). It also supports sending episodes to Streamlit
# via window.parent.postMessage so the Python side can capture and save them.

html_code = f'''
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    html,body {{ height:100%; margin:0; background:transparent; }}
    #stage {{ width:100%; height:100%; touch-action:none; }}
    .controls {{ position: absolute; right: 12px; top: 12px; z-index:50; display:flex; gap:8px; }}
    .btn {{ padding:8px 12px; border-radius:8px; border:none; font-weight:600; cursor:pointer; }}
    .good {{ background:#2ecc71; color:white; }}
    .bad {{ background:#e74c3c; color:white; }}
    .save {{ background:#3498db; color:white; }}
    .clear {{ background:#95a5a6; color:white; }}
    .statsBox {{ position: absolute; left: 12px; top:12px; z-index:50; background: rgba(255,255,255,0.9); padding:8px 10px; border-radius:8px; font-family: Inter, Roboto, sans-serif; font-size:13px; }}
  </style>
</head>
<body>
  <canvas id="stage"></canvas>
  <div class="statsBox" id="statsBox">State: <span id="state">idle</span><br>Episodes: <span id="epsCount">0</span></div>
  <div class="controls">
    <button class="btn good" id="goodBtn">Good!</button>
    <button class="btn bad" id="badBtn">Bad!</button>
    <button class="btn save" id="saveBtn">Save to Streamlit</button>
    <button class="btn clear" id="clearBtn">Clear</button>
  </div>

<script>
(function(){
  const canvas = document.getElementById('stage');
  const ctx = canvas.getContext('2d');
  let w = canvas.width = window.innerWidth * 0.98;
  let h = canvas.height = window.innerHeight * 0.85;

  window.addEventListener('resize', () => {
    w = canvas.width = window.innerWidth * 0.98;
    h = canvas.height = window.innerHeight * 0.85;
  });

  // Agent state
  const agent = { x: w/2, y: h/2, vx:0, vy:0, r:28 };
  let cursor = { x: w/2, y: h/2 };
  let lastTime = performance.now();
  let dtAccum = 0;

  // Behavior parameters (learned weights)
  const params = { followWeight: 1.0, smoothness: 0.85, idleWander: 0.6 };

  // Episode logging
  const episodes = [];
  let currentEpisode = [];
  let tick = 0;

  const STATE = { IDLE: 'idle', FOLLOWING: 'following', PLAYING: 'playing' };
  let state = STATE.IDLE;

  // Mouse capture
  window.addEventListener('mousemove', (e) => {
    cursor.x = e.clientX;
    cursor.y = e.clientY;
  });

  // For touch devices
  window.addEventListener('touchmove', (e) => {
    const t = e.touches[0];
    if (t) { cursor.x = t.clientX; cursor.y = t.clientY; }
  }, { passive:true });

  function distance(a,b){ const dx=a.x-b.x, dy=a.y-b.y; return Math.hypot(dx,dy); }

  function sampleObservation(){
    return {
      t: Date.now(),
      tick,
      state,
      agent: { x: agent.x, y: agent.y, vx: agent.vx, vy: agent.vy },
      cursor: { x: cursor.x, y: cursor.y },
      params: { ...params }
    };
  }

  function computeReward(obs, action){
    // reward components: improvement in distance (positive if got closer), smoothness penalty on jerk
    const prevDist = obs.prevDist ?? null;
    const nowDist = distance(agent, cursor);
    let reward = 0;
    if (prevDist !== null) reward += Math.max(0, prevDist - nowDist); // positive if moved closer
    const speed = Math.hypot(agent.vx, agent.vy);
    reward -= 0.01 * Math.abs(action.ax) + 0.01 * Math.abs(action.ay); // small penalty on big accelerations
    // normalize roughly
    return reward;
  }

  function stepPhysics(dt){
    // Decide behavior state based on distance
    const d = distance(agent, cursor);
    const prevState = state;
    if (d < 60) state = STATE.PLAYING;
    else if (d < 250) state = STATE.FOLLOWING;
    else state = STATE.IDLE;
    document.getElementById('state').innerText = state;

    // Action decision
    let ax = 0, ay = 0;
    if (state === STATE.FOLLOWING){
      // accelerate towards cursor with smoothing
      const dx = cursor.x - agent.x;
      const dy = cursor.y - agent.y;
      ax = dx * (0.002 * params.followWeight);
      ay = dy * (0.002 * params.followWeight);
    } else if (state === STATE.PLAYING){
      // dancing: small oscillations
      ax = Math.sin(performance.now()/120) * 0.12;
      ay = Math.cos(performance.now()/100) * 0.12;
    } else {
      // idle gentle wandering
      ax = (Math.random()-0.5) * 0.06 * params.idleWander;
      ay = (Math.random()-0.5) * 0.06 * params.idleWander;
    }

    // integrate velocity with smoothness
    agent.vx = agent.vx * params.smoothness + ax * (1 - params.smoothness) * 60 * dt;
    agent.vy = agent.vy * params.smoothness + ay * (1 - params.smoothness) * 60 * dt;

    agent.x += agent.vx * dt * 60;
    agent.y += agent.vy * dt * 60;

    // keep inside bounds
    agent.x = Math.max(agent.r, Math.min(w-agent.r, agent.x));
    agent.y = Math.max(agent.r, Math.min(h-agent.r, agent.y));

    // logging action + reward
    const action = { ax, ay, vx: agent.vx, vy: agent.vy };
    return { action };
  }

  function drawAgent(){
    ctx.clearRect(0,0,w,h);
    // subtle shadow
    ctx.save();
    ctx.fillStyle = '#2ecc71';
    ctx.beginPath(); ctx.arc(agent.x, agent.y, agent.r, 0, Math.PI*2); ctx.fill();
    // eyes
    ctx.fillStyle = '#093';
    ctx.beginPath(); ctx.arc(agent.x-9, agent.y-6, 4, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(agent.x+9, agent.y-6, 4, 0, Math.PI*2); ctx.fill();
    // smile
    ctx.strokeStyle = '#062';
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(agent.x, agent.y+2, 12, 0.2*Math.PI, 0.8*Math.PI); ctx.stroke();
    ctx.restore();

    // optional cursor indicator
    ctx.beginPath(); ctx.strokeStyle = 'rgba(0,0,0,0.08)'; ctx.arc(cursor.x, cursor.y, 6, 0, Math.PI*2); ctx.stroke();
  }

  function recordStep(obs, action, reward){
    const rec = { obs, action, reward, feedback: null };
    currentEpisode.push(rec);
  }

  function pushEpisode(){
    if (currentEpisode.length === 0) return;
    const episode = { id: Date.now() + '-' + Math.random().toString(36).slice(2,8), steps: currentEpisode };
    episodes.push(episode);
    currentEpisode = [];
    document.getElementById('epsCount').innerText = episodes.length;
  }

  // feedback buttons
  document.getElementById('goodBtn').addEventListener('click', ()=>{
    // when good pressed, nudge params to prefer recent behavior
    params.followWeight = Math.min(3.0, params.followWeight + 0.08);
    params.smoothness = Math.max(0.4, params.smoothness - 0.02); // a bit more responsive
    // label last episode with good feedback
    if (episodes.length) episodes[episodes.length-1].feedback = 'good';
    else if (currentEpisode.length) currentEpisode[currentEpisode.length-1].feedback = 'good';
    announceStats();
  });
  document.getElementById('badBtn').addEventListener('click', ()=>{
    // when bad pressed, nudge away
    params.followWeight = Math.max(0.1, params.followWeight - 0.12);
    params.smoothness = Math.min(0.98, params.smoothness + 0.03);
    if (episodes.length) episodes[episodes.length-1].feedback = 'bad';
    else if (currentEpisode.length) currentEpisode[currentEpisode.length-1].feedback = 'bad';
    announceStats();
  });

  document.getElementById('clearBtn').addEventListener('click', ()=>{
    episodes.length = 0; currentEpisode.length = 0; localStorage.removeItem('ai_friend_episodes');
    document.getElementById('epsCount').innerText = episodes.length; announceStats();
  });

  document.getElementById('saveBtn').addEventListener('click', ()=>{
    // try to POST to local API first. If that fails, ship to parent Streamlit via postMessage
    const payload = { episodes: episodes };
    const apiUrl = 'http://127.0.0.1:8001/api/episodes';
    fetch(apiUrl, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)})
      .then(r=>{
        if (!r.ok) throw new Error('api save failed');
        return r.json();
      })
      .then(data=>{
        alert('Saved to local API: ' + (data.saved||0) + ' steps');
        // persist to localStorage as well
        localStorage.setItem('ai_friend_episodes', JSON.stringify(episodes));
      })
      .catch(err=>{
        // fallback: send to Streamlit via postMessage
        try {
          window.parent.postMessage({isStreamlitMessage:true, type:'EPISODES', episodes: episodes}, '*');
          alert('Sent episodes to Streamlit for saving (fallback)');
          localStorage.setItem('ai_friend_episodes', JSON.stringify(episodes));
        } catch(e){
          alert('Could not save episodes: ' + e);
        }
      });
  });

  // announce stats to parent occasionally
  function announceStats(){
    const s = { params: params, episodes: episodes.length };
    try { window.parent.postMessage({isStreamlitMessage:true, type:'STATS', stats:s}, '*'); } catch(e){}
  }

  // periodic simulation
  function loop(now){
    const dt = Math.min((now - lastTime) / 1000, 0.1); lastTime = now; tick++;
    // sample observation
    const obs = sampleObservation();
    obs.prevDist = (obs.prevDist === undefined) ? distance(agent,cursor) : obs.prevDist;

    const { action } = stepPhysics(dt);
    const reward = computeReward(obs, action);
    recordStep(obs, action, reward);

    drawAgent();

    // periodically push episode (every 2 seconds)
    if (tick % 40 === 0){ // roughly 40 ticks ~ 2 seconds if 50ms per tick
      pushEpisode();
    }

    // persist to localStorage every 5 seconds
    if (tick % 100 === 0){ localStorage.setItem('ai_friend_episodes', JSON.stringify(episodes)); announceStats(); }

    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  // on load: try to restore saved episodes from localStorage and notify
  try{
    const saved = JSON.parse(localStorage.getItem('ai_friend_episodes') || '[]');
    if (Array.isArray(saved) && saved.length) { episodes.push(...saved); document.getElementById('epsCount').innerText = episodes.length; }
  }catch(e){}

  // listen for messages from parent (optional future control)
  window.addEventListener('message', (ev)=>{
    // nothing for now
  });

})();
</script>
</body>
</html>
'''

import streamlit.components.v1 as components

with col1:
    st.markdown("Move your mouse inside the canvas area — the green smiley will follow. Use Good!/Bad! to teach it.")
    comp_value = components.html(html_code, height=650, scrolling=True)

# If the embedded component sent episodes back via postMessage, components.html may return a value.
# The HTML uses window.parent.postMessage({isStreamlitMessage:true, type:'EPISODES', episodes: episodes}, '*')
# In practice, components.html returns whatever the component posted as a message. We'll try to capture it.

# Helper functions for reading and writing episodes file from Streamlit side.

def append_episodes_to_file(episodes_list):
    if not episodes_list:
        return 0
    EPISODES_FILE.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with EPISODES_FILE.open('a', encoding='utf-8') as f:
        for ep in episodes_list:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
            count += 1
    return count

def read_episodes_file():
    if not EPISODES_FILE.exists():
        return []
    with EPISODES_FILE.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

# If comp_value is truthy and carries episodes, save them.
if comp_value:
    try:
        # sometimes the component returns the whole message object
        if isinstance(comp_value, dict) and 'episodes' in comp_value:
            saved = append_episodes_to_file(comp_value['episodes'])
            st.success(f"Saved {saved} episodes to {EPISODES_FILE}")
        elif isinstance(comp_value, dict) and comp_value.get('type') == 'EPISODES' and 'episodes' in comp_value:
            saved = append_episodes_to_file(comp_value['episodes'])
            st.success(f"Saved {saved} episodes to {EPISODES_FILE}")
        elif isinstance(comp_value, dict) and comp_value.get('type') == 'STATS' and 'stats' in comp_value:
            st.info("Component posted stats")
        else:
            # unknown payload
            pass
    except Exception as e:
        st.error(f"Could not save episodes from component: {e}")

with col2:
    st.header("Controls & Data")
    eps = read_episodes_file()
    st.write(f"Local episodes logged: {len(eps)}")

    if st.button("Download episodes (JSONL)"):
        eps = read_episodes_file()
        if eps:
            st.download_button("Download JSONL", data='\n'.join(json.dumps(x) for x in eps), file_name="episodes.jsonl")
        else:
            st.warning("No episodes yet.")

    if st.button("Clear episodes (Streamlit)"):
        if EPISODES_FILE.exists():
            EPISODES_FILE.unlink()
        st.success("Cleared episodes file.")

    st.markdown("---")
    st.write("API server available:" , use_api)
    if use_api:
        st.write("GET  /api/episodes  — returns saved episodes")
        st.write("GET  /api/stats     — returns simple stats")
        st.write("DELETE /api/episodes — clear saved episodes (use with caution)")

st.markdown("---")
st.caption("Technical: The embedded component runs a 50ms-ish tick loop, logs observations/actions/rewards, and stores episodes in localStorage. Use Good!/Bad! to influence followWeight and smoothness. Click Save to push episodes to the Streamlit backend or to the local API if available.")
