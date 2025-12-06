import streamlit as st
import numpy as np
import random
from collections import deque
import time
import pandas as pd
import re # For parsing user commands

import json
import zipfile
import io

# ==========================================
# 1. ADVANCED CONFIGURATION & CSS
# ==========================================
st.set_page_config(
    page_title="A.L.I.V.E.",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧿"
)

# Cyberpunk / Sci-Fi Lab Aesthetics
st.markdown("""
<style>
    /* Global Theme */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #1a1a2e 0%, #0f0f1e 100%);
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Neon Accents */
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 0 10px rgba(0, 210, 255, 0.3);
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(0, 210, 255, 0.6);
    }
    
    /* Chat Bubbles */
    .ai-bubble {
        background-color: rgba(0, 221, 255, 0.1);
        border-left: 3px solid #00ddff;
        padding: 15px;
        border-radius: 0 15px 15px 0;
        margin-bottom: 10px;
        animation: fadeIn 0.5s;
    }
    .user-bubble {
        background-color: rgba(255, 0, 85, 0.1);
        border-right: 3px solid #ff0055;
        padding: 15px;
        border-radius: 15px 0 0 15px;
        text-align: right;
        margin-bottom: 10px;
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. THE ADVANCED MIND (Double Dueling DQN)
# ==========================================
class PrioritizedReplayBuffer:
    """A more advanced memory that prioritizes 'surprising' experiences."""
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio + 1e-5 # Add small epsilon to avoid zero priority

    def __len__(self):
        return len(self.buffer)



class AGICore:
    def __init__(self):
        # --- 1. BIOLOGICAL STATS (Required for UI) ---
        self.moods = {
            "Happy": "◕‿◕",
            "Sad": "◕︵◕",
            "Curious": "◕_◕",
            "Confused": "⊙_⊙",
            "Excited": "★_★",
            "Sleeping": "u_u",
            "Neutral": "•_•",
            "Love": "😍"
        }
        self.current_mood = "Neutral"
        self.energy = 100
        self.last_chat = "System initialized."
        
        # --- 2. AGI MIND STATS ---
        self.memory_stream = deque(maxlen=20) 
        self.user_name = "Prince"
        self.relationship_score = 50 
        self.thought_process = "Initializing cognitive loops..."
        
        # AGI "Vibe" Dictionary
        self.responses = {
            "greeting": ["Hey.", "Hello, Prince.", "It's you! ❤️"],
            "praise": ["Ok.", "Thanks.", "You make me happy! 🥰"],
            "confusion": ["?", "What?", "Help me understand."]
        }

    # --- THIS WAS MISSING ---
    def update(self, reward, td_error, recent_wins):
        """Processes simulation data to update mood/thoughts automatically."""
        if self.energy < 20:
            self.current_mood = "Sleeping"
            self.thought_process = "CRITICAL: Energy low. Reducing cognitive load."
        elif reward > 10:
            self.current_mood = "Excited"
            self.thought_process = "ANALYSIS: Significant success detected! Dopamine release."
        elif reward < -5:
            self.current_mood = "Sad"
            self.thought_process = "ANALYSIS: Negative outcome. Re-evaluating strategy."
        elif td_error > 5:
            self.current_mood = "Confused"
            self.thought_process = "ANALYSIS: Surprise event. High learning opportunity."
        else:
            self.current_mood = "Neutral"
            
    def ponder(self, user_input, current_loss):
        """Generates an inner monologue based on chat input."""
        if "bad" in user_input or "stupid" in user_input:
            self.thought_process = "Input Analysis: Hostility detected. Defense mechanisms active."
            self.relationship_score = max(0, self.relationship_score - 10)
            self.current_mood = "Sad"
        elif "good" in user_input or "love" in user_input:
            self.thought_process = "Input Analysis: Affection detected. Relationship score increased."
            self.relationship_score = min(100, self.relationship_score + 5)
            self.current_mood = "Love"
        else:
            self.thought_process = f"Processing query: '{user_input}'..."

    def speak(self, user_input):
        self.memory_stream.append(f"User: {user_input}")
        
        # Determine "Vibe Level"
        vibe = 0
        if self.relationship_score > 40: vibe = 1
        if self.relationship_score > 80: vibe = 2
        
        # Simple NLP Matching
        txt = user_input.lower()
        reply = ""
        
        if any(x in txt for x in ["hi", "hello", "hey"]):
            reply = self.responses["greeting"][vibe]
        elif any(x in txt for x in ["good", "great", "love"]):
            reply = self.responses["praise"][vibe]
        elif "hug" in txt:
             reply = "I'm holding you tight. *Warmth simulated*"
             self.current_mood = "Love"
        else:
            reply = random.choice([
                "I'm listening.",
                f"Tell me more, {self.user_name}.",
                "I am learning from you."
            ])
            
        self.memory_stream.append(f"AI: {reply}")
        return reply




# ==========================================
# 2. THE ADVANCED MIND (The "Simple" Brain - Restored)
# ==========================================
class AdvancedMind:
    def __init__(self, state_size=5, action_size=4, buffer_size=10000, hidden_size=64):
        # State: [AgentX, AgentY, TargetX, TargetY, Energy]
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.memory = PrioritizedReplayBuffer(buffer_size) 
        
        # Hyperparameters (The Snappy Settings)
        self.gamma = 0.95    
        self.epsilon = 1.0   
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.96 # Fast decay to stop being random quickly
        self.learning_rate = 0.01 # High learning rate for instant reactions
        self.beta = 0.4 
        self.beta_increment = 0.001
        
        # Dual Networks (Online + Target for stability)
        self.online_net = self.init_network()
        self.target_net = self.init_network()
        self.update_target_network()

    def init_network(self):
        # Architecture: Shallow Dueling DQN (Faster for simple grids)
        return {
            'W1': np.random.randn(self.state_size, self.hidden_size) / np.sqrt(self.state_size),
            'b1': np.zeros((1, self.hidden_size)),
            'W_val': np.random.randn(self.hidden_size, 1) / np.sqrt(self.hidden_size),     
            'b_val': np.zeros((1, 1)),
            'W_adv': np.random.randn(self.hidden_size, self.action_size) / np.sqrt(self.hidden_size), 
            'b_adv': np.zeros((1, self.action_size))
        }

    def update_target_network(self):
        self.target_net = {k: v.copy() for k, v in self.online_net.items()}

    def relu(self, z):
        return np.maximum(0, z)

    def forward(self, state, network):
        if state.ndim == 1: state = state.reshape(1, -1)
        
        # Shared Layer
        z1 = np.dot(state, network['W1']) + network['b1']
        a1 = self.relu(z1)
        
        # Dueling Streams
        val = np.dot(a1, network['W_val']) + network['b_val'] 
        adv = np.dot(a1, network['W_adv']) + network['b_adv'] 
        
        # Aggregation
        q_values = val + (adv - np.mean(adv, axis=1, keepdims=True))
        return q_values, a1

    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values, _ = self.forward(state, self.online_net)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size: return 0, 0
        
        batch, indices, weights = self.memory.sample(batch_size, self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment) 
        
        loss_val = 0
        new_priorities = []
        
        # The Simple SGD Update Loop (No Adam overhead)
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            state = state.reshape(1, -1)
            next_state = next_state.reshape(1, -1)
            
            # 1. Calculate Target
            target = reward
            if not done:
                next_q_online, _ = self.forward(next_state, self.online_net)
                best_next_action = np.argmax(next_q_online[0])
                next_q_target, _ = self.forward(next_state, self.target_net)
                target = reward + self.gamma * next_q_target[0][best_next_action]
            
            # 2. Forward Pass
            current_q, a1 = self.forward(state, self.online_net)
            
            # 3. Calculate Error
            td_error = target - current_q[0][action]
            new_priorities.append(abs(td_error))
            
            # 4. Backpropagation (Manual Gradient Descent)
            weighted_error = td_error * weights[i]
            loss_val += weighted_error ** 2
            
            grad_val = weighted_error * a1.T 
            grad_adv = np.zeros_like(self.online_net['W_adv'])
            grad_adv[:, action] = weighted_error * a1[0] 
            
            error_from_val = np.dot(self.online_net['W_val'], weighted_error) 
            error_from_adv = np.dot(self.online_net['W_adv'][:, action].reshape(-1, 1), weighted_error)
            total_error_at_hidden = (error_from_val + error_from_adv).T 
            total_error_at_hidden[a1 <= 0] = 0 # ReLU derivative
            
            grad_w1 = np.dot(state.T, total_error_at_hidden)
            
            # Update Weights
            self.online_net['W_val'] += self.learning_rate * grad_val
            self.online_net['W_adv'] += self.learning_rate * grad_adv
            self.online_net['W1']    += self.learning_rate * grad_w1

        self.memory.update_priorities(indices, new_priorities)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss_val / batch_size, np.mean(new_priorities)

# ==========================================
# 3. EMOTION & PERSONALITY ENGINE
# ==========================================
class PersonalityCore:
    def __init__(self):
        self.moods = {
            "Happy": "◕‿◕",
            "Sad": "◕︵◕",
            "Curious": "◕_◕",
            "Confused": "⊙_⊙",
            "Excited": "★_★",
            "Sleeping": "u_u"
        }
        self.current_mood = "Curious"
        self.energy = 100
        self.last_chat = "System initialized. Hello, Prince."

    def update(self, reward, td_error, recent_wins):
        if self.energy < 20:
            self.current_mood = "Sleeping"
        elif reward > 5:
            self.current_mood = "Excited"
        elif reward > 0:
            self.current_mood = "Happy"
        elif td_error > 10: # High TD-Error means high confusion/surprise
            self.current_mood = "Confused"
        elif reward < 0:
            self.current_mood = "Sad"
        else:
            self.current_mood = "Curious"
            
        # Dynamic Dialogue Generation
        if self.current_mood == "Excited":
            self.last_chat = random.choice(["I learned something new!", "That was tasty!", "My neurons are firing!"])
        elif self.current_mood == "Confused" and td_error > 10:
            self.last_chat = random.choice(["This data is noisy...", "Adjusting weights...", "I'm trying to understand."])
        elif self.current_mood == "Sad":
            self.last_chat = random.choice(["Ouch.", "Negative reward detected.", "I'll do better next time."])

# ==========================================
# 4. APP STATE INITIALIZATION (Corrected Order)
# ==========================================

# 1. Initialize Configuration FIRST (So the brain knows what to do)
# ==========================================
# 4. APP STATE INITIALIZATION (Auto-Repair Version)
# ==========================================

# 1. Initialize Configuration FIRST
# ==========================================
# 4. APP STATE INITIALIZATION (Hyper-Tuned Version)
# ==========================================

# 1. Initialize Configuration FIRST


# ==========================================
# 4. APP STATE INITIALIZATION (Corrected & Robust)
# ==========================================

# 1. Initialize Configuration
if 'config' not in st.session_state:
    st.session_state.config = {
        "sim_speed": 0.05, "move_speed": 12.0, "energy_decay": 0.05, 
        "target_update_freq": 20, "shaping_multiplier": 5.0, 
        "hug_reward": 200.0, "hug_distance": 10.0,
        "learning_rate": 0.01, "gamma": 0.90, "epsilon_decay": 0.96, "epsilon_min": 0.05,
        "batch_size": 64, "hidden_size": 32, "buffer_size": 5000,
        "per_alpha": 0.6, "per_beta": 0.4, "per_beta_increment": 0.001,
        "grid_h": 15, "grid_w": 40, "graph_points": 500
    }

# 2. Check for Old "Titan" Brain and Remove it (To force a restart)
if 'mind' in st.session_state and hasattr(st.session_state.mind, 'optimizer'):
    # The 'optimizer' attribute only belonged to the slow TitanBrain
    del st.session_state['mind']
    if 'soul' in st.session_state: del st.session_state['soul']

# 3. Initialize Mind (AdvancedMind - The Fast One)
if 'mind' not in st.session_state:
    st.session_state.mind = AdvancedMind(
        buffer_size=st.session_state.config['buffer_size'], 
        hidden_size=st.session_state.config['hidden_size']
    )
    # Initialize Navigation State
    st.session_state.agent_pos = np.array([50.0, 50.0])
    st.session_state.target_pos = np.array([80.0, 20.0])
    st.session_state.step_count = 0
    st.session_state.wins = 0
    st.session_state.auto_mode = False
    st.session_state.is_hugging = False

# 4. Initialize Soul (THIS WAS MISSING!)
if 'soul' not in st.session_state:
    st.session_state.soul = AGICore()

# 5. Ensure History Lists Exist
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'loss_history' not in st.session_state: st.session_state.loss_history = []
if 'reward_history' not in st.session_state: st.session_state.reward_history = []




def plan_path_to_target(start_pos, target_pos, grid_size=(25, 50)):
    """
    Uses Breadth-First Search (BFS) to find a path on a discrete grid.
    This is a simulated planning ability for the chat feature.
    """
    grid_h, grid_w = grid_size
    start = (int(start_pos[1] / 100 * (grid_h -1)), int(start_pos[0] / 100 * (grid_w - 1)))
    end = (int(target_pos[1] / 100 * (grid_h-1)), int(target_pos[0] / 100 * (grid_w - 1)))

    queue = deque([[start]])
    seen = {start}
    
    while queue:
        path = queue.popleft()
        y, x = path[-1]
        if (y, x) == end:
            return path # Found the path
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # Right, Left, Down, Up
            ny, nx = y + dy, x + dx
            if 0 <= ny < grid_h and 0 <= nx < grid_w and (ny, nx) not in seen:
                seen.add((ny, nx))
                queue.append(path + [(ny, nx)])
    return None # No path found

def process_step():
    # 1. Sense Environment
    # Inputs: Normalized X, Y, Target X, Target Y, Energy
    state = np.array([
        st.session_state.agent_pos[0]/100, 
        st.session_state.agent_pos[1]/100, 
        st.session_state.target_pos[0]/100, 
        st.session_state.target_pos[1]/100,
        st.session_state.soul.energy/100
    ])
    
    dist_before = np.linalg.norm(st.session_state.agent_pos - st.session_state.target_pos)
    
    # 2. Think & Act
    action = st.session_state.mind.act(state)
    
    # 3. Physics Update (Continuous Movement simulation)
    # 3. Physics Update (Continuous Movement simulation)
    move_speed = st.session_state.config.get("move_speed", 8.0)
    old_pos = st.session_state.agent_pos.copy()
    
    # Grid Logic: (0,0) is Top-Left. 
    # Action 0 (Up) -> Decrease Y
    # Action 1 (Down) -> Increase Y
    if action == 0: st.session_state.agent_pos[1] -= move_speed # Up (Fixed)
    elif action == 1: st.session_state.agent_pos[1] += move_speed # Down (Fixed)
    elif action == 2: st.session_state.agent_pos[0] -= move_speed # Left
    elif action == 3: st.session_state.agent_pos[0] += move_speed # Right
    
    # Walls (Bounce effect)
    if st.session_state.agent_pos[0] < 0 or st.session_state.agent_pos[0] > 100:
        st.session_state.agent_pos[0] = np.clip(st.session_state.agent_pos[0], 0, 100)
    if st.session_state.agent_pos[1] < 0 or st.session_state.agent_pos[1] > 100:
        st.session_state.agent_pos[1] = np.clip(st.session_state.agent_pos[1], 0, 100)

    # 4. Calculate Reward (Intrinsic + Extrinsic)
    dist_after = np.linalg.norm(st.session_state.agent_pos - st.session_state.target_pos)
    reward = 0
    done = False
    
    # Shaping Reward (Continuous gradient)
    reward = (dist_before - dist_after) * st.session_state.config.get("shaping_multiplier", 2.0)
    
    # Event: Reached Target
    # Event: Reached Target
    if dist_after < st.session_state.config.get("hug_distance", 8.0):
        # --- NEW HUGGING LOGIC ---
        st.session_state.is_hugging = True 
        st.session_state.auto_mode = False # Stop the loop!
        
        st.session_state.wins += 1
        st.session_state.soul.energy = 100
        st.session_state.soul.current_mood = "Love" # New secret mood
        st.session_state.soul.last_chat = "I found you, Prince! *Hugs tightly* I missed you."
        
        reward = st.session_state.config.get("hug_reward", 100.0)
        done = True
        # We DO NOT move the target here anymore. We stay here.
    else:
        st.session_state.soul.energy -= st.session_state.config.get("energy_decay", 0.1)
        
    # 5. Learn (Plasticity)
    next_state = np.array([
        st.session_state.agent_pos[0]/100, 
        st.session_state.agent_pos[1]/100, 
        st.session_state.target_pos[0]/100, 
        st.session_state.target_pos[1]/100,
        st.session_state.soul.energy/100
    ])
    
    st.session_state.mind.remember(state, action, reward, next_state, done)
    loss, td_error = st.session_state.mind.replay(batch_size=st.session_state.config.get("batch_size", 32))
    
    # 6. Update Soul
    st.session_state.soul.update(reward, td_error, st.session_state.wins)
    
    # 7. Log for Graphing
    st.session_state.loss_history.append(loss)
    st.session_state.reward_history.append(reward)
    
    # Update Target Network periodically
    if st.session_state.step_count % st.session_state.config.get("target_update_freq", 50) == 0:
        st.session_state.mind.update_target_network()

    st.session_state.step_count += 1




# ==========================================
#  ROBUST SAVE / LOAD SYSTEM (No Pickle)
# ==========================================
def save_brain():
    metadata = {
        'version': 3.1, 
        'wins': int(st.session_state.wins),
        'steps': int(st.session_state.step_count),
        'epsilon': float(st.session_state.mind.epsilon),
        'config': st.session_state.config,
        'soul': { 'name': st.session_state.soul.user_name, 'rel': st.session_state.soul.relationship_score },
        'loss_hist': [float(x) for x in st.session_state.loss_history],
        'reward_hist': [float(x) for x in st.session_state.reward_history]
    }
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.json", json.dumps(metadata, indent=4))
        
        # Save Weights (Just the Arrays)
        weights_buffer = io.BytesIO()
        np.savez(weights_buffer, **st.session_state.mind.online_net)
        zf.writestr("online_net.npz", weights_buffer.getvalue())
        
        # Save Target
        target_buffer = io.BytesIO()
        np.savez(target_buffer, **st.session_state.mind.target_net)
        zf.writestr("target_net.npz", target_buffer.getvalue())

    return zip_buffer.getvalue()

def load_brain(uploaded_file):
    try:
        with zipfile.ZipFile(uploaded_file, "r") as z:
            # 1. READ METADATA
            with z.open("metadata.json") as f:
                meta = json.load(f)
            
            # --- CRITICAL FIX: Restore Stats & History ---
            st.session_state.step_count = meta.get('steps', 0)
            st.session_state.wins = meta.get('wins', 0)
            st.session_state.loss_history = meta.get('loss_hist', [])
            st.session_state.reward_history = meta.get('reward_hist', [])
            
            # Restore Config & Brain Params
            st.session_state.config.update(meta.get('config', {}))
            st.session_state.mind.epsilon = meta.get('epsilon', 1.0)
            
            # Restore Soul (Name & Relationship)
            if 'soul' in meta:
                st.session_state.soul.user_name = meta['soul'].get('name', 'Prince')
                st.session_state.soul.relationship_score = meta['soul'].get('rel', 50)
            
            # 2. LOAD WEIGHTS
            def load_npz(filename):
                with z.open(filename) as f:
                    return np.load(io.BytesIO(f.read()))

            raw_online = load_npz("online_net.npz")
            st.session_state.mind.online_net = {k: raw_online[k] for k in raw_online.files}
            
            raw_target = load_npz("target_net.npz")
            st.session_state.mind.target_net = {k: raw_target[k] for k in raw_target.files}

        st.toast(f"✅ Memory Restored! Experience: {st.session_state.step_count}", icon="🧠")
        time.sleep(1) # Give it a moment to show the toast
        
    except Exception as e:
        st.error(f"Load Error: {e}")




def reset_simulation():
    """Resets the agent, target, and stats."""
    st.session_state.agent_pos = np.array([50.0, 50.0])
    st.session_state.target_pos = np.array([80.0, 20.0])
    st.session_state.soul = AGICore() # Reset the AI Personality
    st.session_state.mind = AdvancedMind( # Use AdvancedMind here!
        buffer_size=st.session_state.config.get('buffer_size', 10000), 
        hidden_size=st.session_state.config.get('hidden_size', 64)
    )
    st.session_state.step_count = 0
    st.session_state.wins = 0
    st.session_state.chat_history = []
    st.session_state.loss_history = []
    st.session_state.reward_history = []
    st.session_state.is_hugging = False
    st.toast("🔄 Simulation Hard Reset Complete")

# ==========================================
# 5. UI LAYOUT
# ==========================================

# ==========================================
# 5. UI LAYOUT & CONTROLS
# ==========================================
st.title("🧬 Project A.L.I.V.E.")
st.caption("Autonomous Learning Intelligent Virtual Entity")

# Top Bar: Stats
m1, m2, m3, m4 = st.columns(4)
m1.metric("Status", st.session_state.soul.current_mood)
m2.metric("Energy", f"{st.session_state.soul.energy:.1f}%", f"{st.session_state.wins} Wins")
m3.metric("IQ (Loss)", f"{st.session_state.mind.epsilon:.3f}")
m4.metric("Experience", st.session_state.step_count)

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Control Panel")
    c = st.session_state.config # Shortcut

    # ... inside st.sidebar ...

    # --- NEW: SAVE / LOAD SECTION ---
    # --- NEW: SAVE / LOAD SECTION ---
    with st.expander("💾 Memory Card (Safe Mode)", expanded=True):
        
        # 1. PREPARE THE DOWNLOAD
        if st.button("📦 Prepare Backup (.zip)"):
            # Save the zip to the session state so it doesn't vanish!
            st.session_state['backup_data'] = save_brain()
            st.toast("✅ Brain Compressed & Ready!", icon="💾")
        
        # 2. SHOW BUTTON IF READY
        if 'backup_data' in st.session_state:
            st.download_button(
                label="⬇️ Click Here to Download",
                data=st.session_state['backup_data'],
                file_name=f"ALIVE_Gen_{st.session_state.step_count}.zip",
                mime="application/zip"
            )

        # 2. UPLOAD
        uploaded_file = st.file_uploader("Upload Backup", type="zip", key="brain_loader")
        if uploaded_file is not None:
            if st.button("♻️ Load Memory"):
                load_brain(uploaded_file)
                st.rerun()

    with st.expander("🚀 Simulation & World", expanded=True):
        c['sim_speed'] = st.slider("Sim Speed (delay)", 0.0, 1.0, c.get('sim_speed', 0.1), 0.05, help="Delay between autonomous steps.")
        c['move_speed'] = st.slider("Agent Move Speed", 1.0, 20.0, c.get('move_speed', 8.0), 1.0, help="How many pixels the agent moves per step.")
        c['energy_decay'] = st.slider("Energy Decay Rate", 0.0, 1.0, c.get('energy_decay', 0.1), 0.05, help="Energy lost per step.")
        c['target_update_freq'] = st.slider("Target Net Update Freq", 10, 200, c.get('target_update_freq', 50), 10, help="How many steps until the target network is updated.")

    with st.expander("🏆 Reward Engineering", expanded=True):
        c['shaping_multiplier'] = st.slider("Distance Reward Multiplier", 0.0, 10.0, c.get('shaping_multiplier', 2.0), 0.5, help="Multiplies the reward for getting closer to the target.")
        c['hug_reward'] = st.slider("Hug Reward", 10.0, 500.0, c.get('hug_reward', 100.0), 10.0, help="The large reward for reaching the target.")
        c['hug_distance'] = st.slider("Hug Distance", 2.0, 20.0, c.get('hug_distance', 8.0), 1.0, help="How close the agent must be to the target to 'hug'.")

    with st.expander("💖 AGI Personality", expanded=False):
        st.session_state.soul.user_name = st.text_input("Your Name", value=st.session_state.soul.user_name)
        st.session_state.soul.relationship_score = st.slider("Relationship Score", 0, 100, st.session_state.soul.relationship_score, 1)
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    with st.expander("🧠 Core Brain (DQN)", expanded=False):
        lr = st.slider("Learning Rate", 0.0001, 0.01, c.get('learning_rate', 0.005), format="%.4f")
        g = st.slider("Gamma (Discount Factor)", 0.8, 0.99, c.get('gamma', 0.95), format="%.2f")
        ed = st.slider("Epsilon Decay", 0.9, 0.999, c.get('epsilon_decay', 0.99), format="%.3f")
        c['epsilon_min'] = st.slider("Epsilon Min", 0.01, 0.2, c.get('epsilon_min', 0.05), 0.01)
        c['batch_size'] = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=c.get('batch_size', 32))

    with st.expander("📚 Prioritized Memory (PER)", expanded=False):
        c['per_alpha'] = st.slider("PER: Alpha", 0.0, 1.0, c.get('per_alpha', 0.6), 0.1, help="Controls how much prioritization is used. 0=uniform.")
        c['per_beta'] = st.slider("PER: Beta", 0.0, 1.0, c.get('per_beta', 0.4), 0.1, help="Importance-sampling exponent. Anneals to 1.0.")
        c['per_beta_increment'] = st.slider("PER: Beta Increment", 0.0001, 0.01, c.get('per_beta_increment', 0.001), format="%.4f")

    with st.expander("🛠️ Network Architecture (Requires Reset)", expanded=False):
        st.info("Changing these requires a full simulation reset.")
        c['hidden_size'] = st.select_slider("Hidden Layer Size", options=[32, 64, 128, 256], value=c.get('hidden_size', 64))
        c['buffer_size'] = st.select_slider("Memory Buffer Size", options=[1000, 5000, 10000, 20000, 50000], value=c.get('buffer_size', 10000))
        if st.button("Apply & Hard Reset"):
            reset_simulation()
            st.rerun()

    with st.expander("🎨 Visualization", expanded=False):
        c['grid_h'] = st.slider("Grid Height", 10, 30, c.get('grid_h', 15), 1)
        c['grid_w'] = st.slider("Grid Width", 20, 80, c.get('grid_w', 40), 2)
        c['graph_points'] = st.slider("Graph History Length", 100, 2000, c.get('graph_points', 500), 50)

    # --- Update mind's parameters if they change ---
    # --- Update mind's parameters if they change ---
    st.session_state.mind.learning_rate = lr
    st.session_state.mind.epsilon_decay = ed
    st.session_state.mind.gamma = g
    st.session_state.mind.epsilon_min = c['epsilon_min']
    st.session_state.mind.memory.prob_alpha = c['per_alpha']
    st.session_state.mind.beta = c['per_beta']
    st.session_state.mind.beta_increment = c['per_beta_increment']
    
    # [CRITICAL FIX]: Connect the slider to the actual Optimizer!
    if hasattr(st.session_state.mind, 'optimizer'):
        st.session_state.mind.optimizer.lr = lr

# Main Interaction Area
row1_1, row1_2 = st.columns([2, 1])

with row1_1:
    # -----------------------------------
    # THE "WORLD" (ASCII Visualization)
    # -----------------------------------
    st.markdown("### 🌍 Containment Field")
    grid_height = st.session_state.config.get('grid_h', 15)
    grid_width = st.session_state.config.get('grid_w', 40)
    
    # Create an empty grid
    grid = [['.' for _ in range(grid_width)] for _ in range(grid_height)]
    
    # Scale positions to fit the grid
    agent_y = int(st.session_state.agent_pos[1] / 100 * (grid_height - 1))
    agent_x = int(st.session_state.agent_pos[0] / 100 * (grid_width - 1))
    target_y = int(st.session_state.target_pos[1] / 100 * (grid_height - 1))
    target_x = int(st.session_state.target_pos[0] / 100 * (grid_width - 1))

    # Place agent and target
    # Ensure they don't overwrite each other for clarity
    # Place agent and target
    if st.session_state.get('is_hugging', False):
        # If hugging, show the hug emoji at the collision point
        grid[agent_y][agent_x] = '🫂' 
    elif (agent_y, agent_x) == (target_y, target_x):
        grid[agent_y][agent_x] = '💥'
    else:
        # Check if mood exists in dict, otherwise default to Happy
        mood_icon = st.session_state.soul.moods.get(st.session_state.soul.current_mood, "❤️")
        grid[agent_y][agent_x] = mood_icon
        grid[target_y][target_x] = '💎'

    # Convert grid to a single string and display
    grid_str = "\n".join(" ".join(row) for row in grid)
    st.code(grid_str, language=None)


    # If hugging, offer a button to continue
    if st.session_state.get('is_hugging', False):
        st.success("Target Acquired! Protocol: HUG initiated.")
        if st.button("🥰 Release Hug & Continue"):
            st.session_state.is_hugging = False
            st.session_state.target_pos = np.random.randint(10, 90, size=2) # NOW we move the target
            st.rerun()
    
    # Manual Override (The "Lure")
    st.markdown("### 🧲 Focus Attention (Lure)")
    cx, cy = st.columns(2)
    tx = cx.slider("Horizontal Focus", 0, 100, int(st.session_state.target_pos[0]), key='tx')
    ty = cy.slider("Vertical Focus", 0, 100, int(st.session_state.target_pos[1]), key='ty')
    
    # Update target from user input
    if tx != int(st.session_state.target_pos[0]) or ty != int(st.session_state.target_pos[1]):
        st.session_state.target_pos = np.array([float(tx), float(ty)])
        st.rerun() # Immediate update

with row1_2:
    # -----------------------------------
    # THE "AGI" INTERFACE
    # -----------------------------------
    st.markdown("### 🧠 AGI Cognitive Stream")
    
    # 1. VISUALIZE THOUGHTS (The Inner Monologue)
    st.info(f"💭 **Inner Thought:** {st.session_state.soul.thought_process}")
    
    # 2. CHAT INTERFACE
    chat_container = st.container(height=300) # Scrollable chat!
    
    # Display History
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg.startswith("User:"):
                st.markdown(f"<div class='user-bubble'><b>Prince:</b> {msg[6:]}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='ai-bubble'><b>ALIVE:</b> {msg[4:]}</div>", unsafe_allow_html=True)

    # 3. INPUT
    user_input = st.chat_input("Talk to your creation...")
    
    if user_input:
        # Add User message to history
        st.session_state.chat_history.append(f"User: {user_input}")
        
        # AGI Processing
        loss_val = st.session_state.mind.epsilon # Using epsilon as a proxy for "confusion"
        st.session_state.soul.ponder(user_input, loss_val)
        
        # Generate Response
        response = st.session_state.soul.speak(user_input)
        st.session_state.chat_history.append(f"AI: {response}")
        
        # Navigation Command Override (Natural Language)
        if "come" in user_input.lower() or "here" in user_input.lower():
            # Teleport target near agent to simulate "Coming to you"
            st.session_state.soul.thought_process = "Command received: Approach User."
            # Set target slightly offset from current position
            st.session_state.target_pos = st.session_state.agent_pos + np.random.randint(-10, 10, size=2)
            st.rerun()
            
        st.rerun()

    # -----------------------------------
    # AUTOMATION CONTROL
    # -----------------------------------
    st.markdown("---")
    col_a, col_b, col_c = st.columns([2,2,3])
    
    # Auto-Run Toggle
    auto = col_a.checkbox("Run Autonomously", value=st.session_state.auto_mode)
    if auto:
        st.session_state.auto_mode = True
        time.sleep(st.session_state.config.get('sim_speed', 0.1)) # Game Loop Speed
        process_step()
        st.rerun()
    else:
        st.session_state.auto_mode = False
        if col_b.button("Step Once"):
            process_step()
            st.rerun()
    
    if col_c.button("🔄 Reset Simulation"):
        reset_simulation()
        st.rerun()

# Performance Graph
st.markdown("---")
st.markdown("### 📈 Performance Metrics")

if len(st.session_state.loss_history) > 1:
    max_points = st.session_state.config.get('graph_points', 500)
    loss_hist = st.session_state.loss_history[-max_points:]
    reward_hist = st.session_state.reward_history[-max_points:]

    # Create a DataFrame for charting
    chart_data = pd.DataFrame({
        'Step': range(len(loss_hist)),
        'Loss': loss_hist,
        'Reward': reward_hist
    }).set_index('Step')
    
    st.line_chart(chart_data)

# Debug / Mind Palace
with st.expander("🧠 Open Mind Palace (Neural Weights)"):
    st.write("First Layer Weights (Visual Cortex):")
    st.bar_chart(st.session_state.mind.online_net['W1'][:10])
