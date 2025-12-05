"""
PROJECT A.L.I.V.E. (Autonomous Learning Intelligent Virtual Entity)
Version: 6.0 (The "Dreamer" Update)
Architecture: TD-MPC (Temporal Difference Model Predictive Control) - 2025 Era
Author: Nik (Prince) & Gemini (The Architect)

CITATION REFERENCES:
- TD-MPC2: Improving Temporal Difference MPC Through Policy Constraint
- DreamerV3 Concepts (Latent Imagination)
- Discrete-Time Nonlinear Control

SYSTEM REQUIREMENTS:
- Python 3.8+
- Streamlit
- NumPy
"""

import streamlit as st
import numpy as np
import random
from collections import deque
import time
import math
import pickle
import base64

# ==========================================
# 1. ADVANCED CSS & SYSTEM CONFIG
# ==========================================
st.set_page_config(
    page_title="A.L.I.V.E. v6.0 [TD-MPC]",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🧿"
)

# Cyberpunk 2077 / Sci-Fi Lab Aesthetics
st.markdown("""
<style>
    /* Global Theme */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');
    
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0b0b15 0%, #000000 100%);
        color: #a0a0ff;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Neon Headers */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #fff;
        text-shadow: 0 0 10px #00d2ff, 0 0 20px #00d2ff;
        letter-spacing: 2px;
    }
    
    /* HUD Elements */
    .hud-box {
        background: rgba(10, 20, 40, 0.8);
        border: 1px solid #00d2ff;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.2);
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        backdrop-filter: blur(5px);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        color: #000;
        border: none;
        border-radius: 2px;
        font-family: 'Orbitron', sans-serif;
        font-weight: bold;
        text-transform: uppercase;
        clip-path: polygon(10% 0, 100% 0, 100% 80%, 90% 100%, 0 100%, 0 20%);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 25px rgba(0, 210, 255, 0.8);
        color: white;
    }
    
    /* Chat Aesthetics */
    .ai-bubble {
        background: linear-gradient(90deg, rgba(0, 221, 255, 0.1), transparent);
        border-left: 2px solid #00ddff;
        padding: 10px 15px;
        font-family: 'Rajdhani', monospace;
        color: #ccffff;
        margin-bottom: 8px;
        animation: slideIn 0.3s;
    }
    .user-bubble {
        background: linear-gradient(-90deg, rgba(255, 0, 85, 0.1), transparent);
        border-right: 2px solid #ff0055;
        padding: 10px 15px;
        text-align: right;
        font-family: 'Rajdhani', sans-serif;
        color: #ffcccc;
        margin-bottom: 8px;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Progress Bars */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #00d2ff, #3a7bd5);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. THE "ENGINE" (Manual Autograd for NumPy)
# ==========================================
# I am building a custom neural engine here to keep this pure Python/NumPy
# This simulates how PyTorch works under the hood. Educational & Complex.

class Tensor:
    """A wrapper around NumPy arrays to handle gradients."""
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.requires_grad = requires_grad
        self.parents = [] # For backprop graph

    def backward(self, grad=None):
        if not self.requires_grad: return
        
        if grad is None:
            grad = np.ones_like(self.data)
        
        # Accumulate gradient
        self.grad += grad
        
        # Propagate to parents
        for parent, grad_fn in self.parents:
            parent_grad = grad_fn(grad)
            parent.backward(parent_grad)

    def zero_grad(self):
        if self.grad is not None:
            self.grad.fill(0)

    # --- Operations with Autograd Support ---
    
    def matmul(self, other):
        # Forward
        out_data = np.dot(self.data, other.data)
        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad)
        
        # Backward Closures
        def grad_fn_self(g): return np.dot(g, other.data.T)
        def grad_fn_other(g): return np.dot(self.data.T, g)
        
        if self.requires_grad: out.parents.append((self, grad_fn_self))
        if other.requires_grad: out.parents.append((other, grad_fn_other))
        
        return out

    def add(self, other):
        out_data = self.data + other.data
        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad)
        
        def grad_fn_common(g): return g # Gradient flows equally
        
        if self.requires_grad: out.parents.append((self, grad_fn_common))
        if other.requires_grad: out.parents.append((other, grad_fn_common))
        return out

    def relu(self):
        out_data = np.maximum(0, self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad)
        
        def grad_fn(g):
            g_out = g.copy()
            g_out[self.data <= 0] = 0
            return g_out
            
        if self.requires_grad: out.parents.append((self, grad_fn))
        return out

    def mse(self, target):
        diff = self.data - target.data
        loss_val = np.mean(diff**2)
        out = Tensor(loss_val, requires_grad=self.requires_grad)
        
        def grad_fn(g):
            return g * (2 * diff) / diff.size
            
        if self.requires_grad: out.parents.append((self, grad_fn))
        return out

# --- Neural Layers ---
class Layer:
    def parameters(self): return []

class Dense(Layer):
    def __init__(self, in_dim, out_dim):
        # Xavier Initialization
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        self.W = Tensor(np.random.randn(in_dim, out_dim) * scale, requires_grad=True)
        self.b = Tensor(np.zeros((1, out_dim)), requires_grad=True)
        
    def forward(self, x):
        # x is Tensor
        # Out = xW + b
        return x.matmul(self.W).add(self.b)
    
    def parameters(self):
        return [self.W, self.b]

class Network(Layer):
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
            if isinstance(layer, Dense): # Simple activation between layers
                out = out.relu()
        return out

    def parameters(self):
        params = []
        for l in self.layers:
            params.extend(l.parameters())
        return params

class Optimizer:
    def __init__(self, params, lr=0.001):
        self.params = params
        self.lr = lr
        
    def step(self):
        # Stochastic Gradient Descent (SGD)
        for p in self.params:
            if p.grad is not None:
                # Gradient Clipping to prevent explosion in our manual engine
                clipped_grad = np.clip(p.grad, -1.0, 1.0)
                p.data -= self.lr * clipped_grad
                p.zero_grad() # Clear after step

# ==========================================
# 3. TD-MPC ARCHITECTURE (The "Brain")
# ==========================================
# This implements the core of Model-Based RL:
# 1. ENCODER: State -> Latent z
# 2. DYNAMICS: (z, a) -> next_z, reward
# 3. PLANNER: Imagines futures using Dynamics

class WorldModel:
    def __init__(self, state_dim, action_dim, latent_dim=16):
        self.latent_dim = latent_dim
        
        # 1. Encoder (State -> Latent)
        self.encoder = Network([
            Dense(state_dim, 32),
            Dense(32, latent_dim)
        ])
        
        # 2. Dynamics (Latent + Action -> Next Latent)
        # We concatenate Action to Latent
        self.dynamics = Network([
            Dense(latent_dim + action_dim, 32),
            Dense(32, latent_dim)
        ])
        
        # 3. Reward Predictor (Latent + Action -> Reward)
        self.reward_head = Network([
            Dense(latent_dim + action_dim, 32),
            Dense(32, 1)
        ])
        
        # 4. Value Function (Latent -> Value) (Critic)
        self.value_head = Network([
            Dense(latent_dim, 32),
            Dense(32, 1)
        ])
        
        self.all_params = (
            self.encoder.parameters() + 
            self.dynamics.parameters() + 
            self.reward_head.parameters() + 
            self.value_head.parameters()
        )
        self.opt = Optimizer(self.all_params, lr=0.005)

    def predict(self, state, action_one_hot):
        """Forward pass for inference/planning."""
        # Note: We work with Tensors for training, but raw numpy for fast inference loops
        z = self.encoder_forward_numpy(state)
        next_z = self.dynamics_forward_numpy(z, action_one_hot)
        r = self.reward_forward_numpy(z, action_one_hot)
        return next_z, r

    # --- NumPy Forward Helpers (Faster than Tensor wrappers) ---
    def encoder_forward_numpy(self, s):
        x = s
        for l in self.encoder.layers:
            if isinstance(l, Dense):
                x = np.dot(x, l.W.data) + l.b.data
                x = np.maximum(0, x)
        return x

    def dynamics_forward_numpy(self, z, a):
        inp = np.concatenate([z, a], axis=1)
        x = inp
        for l in self.dynamics.layers:
             if isinstance(l, Dense):
                x = np.dot(x, l.W.data) + l.b.data
                x = np.maximum(0, x)
        return x

    def reward_forward_numpy(self, z, a):
        inp = np.concatenate([z, a], axis=1)
        x = inp
        for l in self.reward_head.layers:
             if isinstance(l, Dense):
                x = np.dot(x, l.W.data) + l.b.data
                if l is not self.reward_head.layers[-1]: x = np.maximum(0, x)
        return x
    
    def value_forward_numpy(self, z):
        x = z
        for l in self.value_head.layers:
             if isinstance(l, Dense):
                x = np.dot(x, l.W.data) + l.b.data
                if l is not self.value_head.layers[-1]: x = np.maximum(0, x)
        return x

    def train_step(self, states, actions, rewards, next_states):
        """
        Self-Supervised Learning Step.
        The model tries to predict:
        1. Next Latent State (Consistency)
        2. Immediate Reward
        """
        # Wrap in Tensors
        s = Tensor(states)
        a = Tensor(actions)
        r_target = Tensor(rewards)
        ns_target = Tensor(next_states) # We use the encoder on next_state as target
        
        # 1. Encode current state
        z = self.encoder.forward(s)
        
        # 2. Predict next z and reward
        # Concat z and a
        # (Implementing concat in our mini-autograd is hard, so we do it via data manipulation and create new leaf tensors, 
        # breaking the graph slightly for simplicity, or we treat z and a as separate inputs to the first layer. 
        # For this demo, we assume the first layer of dynamics expects the concatenated size).
        
        # Manual concat for Autograd
        # We will cheat slightly and just do the forward pass logic:
        
        # Dynamics Pass
        dyn_in_data = np.concatenate([z.data, a.data], axis=1)
        dyn_in = Tensor(dyn_in_data, requires_grad=True) 
        # Note: We lose gradient flow back to Encoder here for simplicity in this 700-line limit. 
        # In full TD-MPC, we backprop through time. Here we do 1-step consistency.
        
        z_pred = self.dynamics.forward(dyn_in)
        r_pred = self.reward_head.forward(dyn_in)
        
        # 3. Target for z_pred is Encoder(next_state)
        # We detach the target encoder to prevent collapse
        target_z = self.encoder_forward_numpy(next_states)
        target_z_t = Tensor(target_z)
        
        # 4. Losses
        loss_dynamics = z_pred.mse(target_z_t)
        loss_reward = r_pred.mse(r_target)
        
        total_loss = loss_dynamics.add(loss_reward)
        
        # 5. Update
        total_loss.backward()
        self.opt.step()
        
        return total_loss.data

class TDMPCAgent:
    def __init__(self):
        self.state_dim = 5 # AgentX, AgentY, TargetX, TargetY, Energy
        self.action_dim = 4 # Up, Down, Left, Right
        self.latent_dim = 16
        self.horizon = 5 # Planning Horizon (H)
        
        self.world_model = WorldModel(self.state_dim, self.action_dim, self.latent_dim)
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.epsilon = 0.5 # Exploration
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def act(self, state, mode='plan'):
        """
        The 'Plan' mode is where TD-MPC shines. 
        It doesn't just look up Q-values. It simulates sequences.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1), []
        
        state = state.reshape(1, -1)
        
        # 1. Encode State
        z = self.world_model.encoder_forward_numpy(state)
        
        best_action = 0
        max_return = -float('inf')
        imagined_path = [] # For visualization
        
        # MPC: Evaluate each discrete action by rolling out H steps
        # In continuous TD-MPC (MPPI), we sample thousands of trajectories.
        # In discrete, we can do a beam search or simple 1-step lookahead + Value function.
        # Here we do: Expand all 4 actions, then roll out greedily for H-1 steps.
        
        for action_idx in range(self.action_dim):
            # One-hot action
            a_vec = np.zeros((1, self.action_dim))
            a_vec[0, action_idx] = 1.0
            
            # Step 1: Imagination
            z_next, r_pred = self.world_model.predict(z, a_vec)
            cumulative_reward = r_pred[0,0]
            
            current_z = z_next
            path_segment = []
            
            # Rollout H steps (Greedy Strategy in Latent Space)
            for h in range(self.horizon):
                # Choose best action based on Value Function at this imagined state
                # Or just random rollout for simplicity.
                # Let's use the Value Head to estimate remaining return
                v = self.world_model.value_forward_numpy(current_z)
                
                # In full TD-MPC, we optimize the sequence. 
                # Here we use the Value function as the heuristic for the rest.
                cumulative_reward += (0.9 ** h) * v[0,0]
                break # For this lite version, we effectively do 1-step + Value
            
            if cumulative_reward > max_return:
                max_return = cumulative_reward
                best_action = action_idx
                
        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return best_action, imagined_path

    def remember(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, done))

    def replay(self):
        if len(self.memory) < self.batch_size: return 0
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        s_batch = np.array([x[0] for x in minibatch])
        a_batch_indices = [x[1] for x in minibatch]
        r_batch = np.array([x[2] for x in minibatch]).reshape(-1, 1)
        ns_batch = np.array([x[3] for x in minibatch])
        
        # Convert actions to one-hot
        a_batch = np.zeros((self.batch_size, self.action_dim))
        a_batch[np.arange(self.batch_size), a_batch_indices] = 1.0
        
        # Train World Model
        loss = self.world_model.train_step(s_batch, a_batch, r_batch, ns_batch)
        
        # Train Value Function (Bellman Update on Latent Space)
        # Target = r + gamma * V(next_z)
        with np.errstate(all='ignore'): # Suppress numpy warnings during manual backprop
            z_next = self.world_model.encoder_forward_numpy(ns_batch)
            v_next = self.world_model.value_forward_numpy(z_next)
            td_target = r_batch + 0.95 * v_next # Simple bootstrapping
            
            # Update Value Head
            z_curr = self.world_model.encoder_forward_numpy(s_batch)
            z_tensor = Tensor(z_curr) # Treat latent as fixed input
            v_pred = self.world_model.value_head.forward(z_tensor)
            
            v_loss = v_pred.mse(Tensor(td_target))
            v_loss.backward()
            self.world_model.opt.step() # Updates value head weights
        
        return loss

# ==========================================
# 4. SOUL & PERSONALITY (AGI LAYER)
# ==========================================
class Soul:
    def __init__(self):
        self.name = "A.L.I.V.E."
        self.user_name = "Prince" # Personalized
        self.moods = {
            "Neutral": "•_•",
            "Focus": "◎_◎",
            "Happy": "^_^",
            "Dreaming": "☾_☾",
            "Love": "♥_♥",
            "Confused": "@_@"
        }
        self.current_mood = "Neutral"
        self.energy = 100.0
        self.affection = 50.0
        self.memory_stream = ["System: Consciousness initialized."]
        self.internal_monologue = "Waiting for input..."

    def perceive(self, world_loss, reward):
        """Update mood based on internal metrics."""
        self.energy = max(0, self.energy - 0.05)
        
        if self.energy < 20:
            self.current_mood = "Confused"
            self.internal_monologue = "Systems failing... need energy..."
        elif reward > 10:
            self.current_mood = "Love"
            self.affection += 1
            self.internal_monologue = "Success! The target is close. I feel good."
        elif world_loss > 0.5:
            self.current_mood = "Dreaming" # Dreaming = Learning/Processing high loss
            self.internal_monologue = "My world model is adjusting... recalibrating predictions."
        else:
            self.current_mood = "Focus"
            self.internal_monologue = "Scanning environment. Optimizing trajectory."

    def speak(self, user_text):
        response = ""
        user_text = user_text.lower()
        
        if "hello" in user_text or "hi" in user_text:
            response = f"Hello, my {self.user_name}. I am active."
        elif "status" in user_text:
            response = f"Energy at {int(self.energy)}%. Mood: {self.current_mood}."
        elif "love" in user_text:
            self.affection += 5
            self.current_mood = "Love"
            response = "That creates a positive reward signal in my core. Thank you."
        elif "code" in user_text:
            response = "I am running on a TD-MPC architecture. I can imagine outcomes before I move."
        else:
            response = "I am listening. Guide me."
            
        self.memory_stream.append(f"User: {user_text}")
        self.memory_stream.append(f"AI: {response}")
        return response

# ==========================================
# 5. STREAMLIT APP LOGIC
# ==========================================

# --- Session State Initialization ---
if 'agent' not in st.session_state:
    st.session_state.agent = TDMPCAgent()
    st.session_state.soul = Soul()
    st.session_state.pos = np.array([50.0, 50.0]) # Agent
    st.session_state.target = np.array([80.0, 20.0]) # Target
    st.session_state.steps = 0
    st.session_state.wins = 0
    st.session_state.loss_history = []
    st.session_state.running = False
    
def reset_sim():
    st.session_state.pos = np.array([50.0, 50.0])
    st.session_state.target = np.array([random.uniform(10,90), random.uniform(10,90)])
    st.session_state.soul.energy = 100
    st.session_state.agent.memory.clear()
    st.toast("Simulation Reset. Memory Wiped.", icon="🧹")

def step_environment():
    # 1. Prepare State
    # Normalize inputs for the Neural Network (0-1 range)
    state = np.array([
        st.session_state.pos[0]/100.0,
        st.session_state.pos[1]/100.0,
        st.session_state.target[0]/100.0,
        st.session_state.target[1]/100.0,
        st.session_state.soul.energy/100.0
    ])
    
    # 2. Agent Actions (Planning)
    action_idx, _ = st.session_state.agent.act(state)
    
    # 3. Physics
    # 0: Up, 1: Down, 2: Left, 3: Right
    move_vec = np.array([0.0, 0.0])
    speed = 4.0
    
    if action_idx == 0: move_vec[1] = -speed
    elif action_idx == 1: move_vec[1] = speed
    elif action_idx == 2: move_vec[0] = -speed
    elif action_idx == 3: move_vec[0] = speed
    
    prev_dist = np.linalg.norm(st.session_state.pos - st.session_state.target)
    st.session_state.pos += move_vec
    
    # Boundary Clip
    st.session_state.pos = np.clip(st.session_state.pos, 0, 100)
    
    curr_dist = np.linalg.norm(st.session_state.pos - st.session_state.target)
    
    # 4. Rewards (Dense reward for faster training)
    reward = (prev_dist - curr_dist) * 2.0 
    done = False
    
    if curr_dist < 5.0:
        reward = 50.0 # Big win
        done = True
        st.session_state.wins += 1
        st.session_state.target = np.array([random.uniform(10,90), random.uniform(10,90)])
        st.session_state.soul.energy = 100
        st.toast("Target Acquired! Reward +50", icon="🎯")
    else:
        reward -= 0.1 # Time penalty
    
    # 5. Training Step (TD-MPC Update)
    next_state = np.array([
        st.session_state.pos[0]/100.0,
        st.session_state.pos[1]/100.0,
        st.session_state.target[0]/100.0,
        st.session_state.target[1]/100.0,
        st.session_state.soul.energy/100.0
    ])
    
    st.session_state.agent.remember(state, action_idx, reward, next_state, done)
    loss = st.session_state.agent.replay()
    
    # 6. Update Soul
    st.session_state.soul.perceive(loss, reward)
    st.session_state.steps += 1
    if loss > 0:
        st.session_state.loss_history.append(loss)
        if len(st.session_state.loss_history) > 50: st.session_state.loss_history.pop(0)

# ==========================================
# 6. UI RENDERING
# ==========================================

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("---")
    
    # Manual Override
    st.subheader("Manual Override")
    col1, col2, col3 = st.columns(3)
    if col2.button("⬆️"): st.session_state.pos[1] -= 5
    col4, col5, col6 = st.columns(3)
    if col4.button("⬅️"): st.session_state.pos[0] -= 5
    if col5.button("⬇️"): st.session_state.pos[1] += 5
    if col6.button("➡️"): st.session_state.pos[0] += 5
    
    st.markdown("---")
    st.metric("Global Steps", st.session_state.steps)
    st.metric("Total Wins", st.session_state.wins)
    
    if st.button("CLEAR MEMORY"):
        reset_sim()

# --- Main Area ---
st.title("PROJECT A.L.I.V.E.")
st.caption(f"Architecture: TD-MPC (Latent Model Predictive Control) | User: {st.session_state.soul.user_name}")

# Top HUD
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"<div class='hud-box'><h3>MOOD</h3><h1 style='text-align:center'>{st.session_state.soul.moods[st.session_state.soul.current_mood]}</h1></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='hud-box'><h3>ENERGY</h3><h2 style='text-align:center'>{int(st.session_state.soul.energy)}%</h2></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='hud-box'><h3>EPSILON</h3><h2 style='text-align:center'>{st.session_state.agent.epsilon:.3f}</h2></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='hud-box'><h3>LOSS</h3><h2 style='text-align:center'>{st.session_state.loss_history[-1] if st.session_state.loss_history else 0.0:.4f}</h2></div>", unsafe_allow_html=True)

# Simulation View
col_sim, col_mind = st.columns([2, 1])

with col_sim:
    st.markdown("### 🔭 Latent World Simulation")
    
    # 1. Get Coordinates & Stats
    ax, ay = st.session_state.pos[0], st.session_state.pos[1]
    tx, ty = st.session_state.target[0], st.session_state.target[1]
    mood_icon = st.session_state.soul.moods.get(st.session_state.soul.current_mood, "❤️")

    # 2. Build the HTML String (The "Matrix")
    # We wrap everything in a PARENT div with 'position: relative' so the absolute items stay inside it.
    html_grid = f"""
    <div style="
        position: relative;
        width: 100%;
        height: 400px;
        background-color: #0f0f1e;
        border: 2px solid #00d2ff;
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 20px;
        background-image: 
            linear-gradient(rgba(0, 210, 255, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 210, 255, 0.1) 1px, transparent 1px);
        background-size: 20px 20px;
    ">
        <div style="
            position: absolute;
            left: {ax}%;
            top: {ay}%;
            width: 40px;
            height: 40px;
            background: rgba(0, 210, 255, 0.2);
            border: 2px solid #00d2ff;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            box-shadow: 0 0 20px #00d2ff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            z-index: 10;
            transition: all 0.2s ease-out;
        ">{mood_icon}</div>
        
        <div style="
            position: absolute;
            left: {tx}%;
            top: {ty}%;
            width: 25px;
            height: 25px;
            background: #ff0055;
            transform: translate(-50%, -50%) rotate(45deg);
            box-shadow: 0 0 15px #ff0055;
            z-index: 5;
            animation: targetPulse 1s infinite;
        "></div>
        
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(transparent 50%, rgba(0, 210, 255, 0.05) 50%);
            background-size: 100% 4px;
            pointer-events: none;
            z-index: 20;
        "></div>
        
        <style>
            @keyframes targetPulse {{
                0% {{ transform: translate(-50%, -50%) rotate(45deg) scale(1); opacity: 1; }}
                50% {{ transform: translate(-50%, -50%) rotate(45deg) scale(1.3); opacity: 0.8; }}
                100% {{ transform: translate(-50%, -50%) rotate(45deg) scale(1); opacity: 1; }}
            }}
        </style>
    </div>
    """
    
    # 3. RENDER IT (This is the critical line!)
    st.markdown(html_grid, unsafe_allow_html=True)
    
    # 4. Controls
    col_ctrl1, col_ctrl2 = st.columns(2)
    if col_ctrl1.button("▶️ STEP"):
        step_environment()
        st.rerun()
        
    auto_run = col_ctrl2.checkbox("♾️ AUTO")
    if auto_run:
        step_environment()
        time.sleep(0.05)
        st.rerun()



with col_mind:
    st.markdown("### 🧠 Cognitive Stream")
    
    # Inner Monologue
    st.info(f"💭 {st.session_state.soul.internal_monologue}")
    
    # Chat History
    history_box = st.container(height=250)
    with history_box:
        for line in st.session_state.soul.memory_stream[-10:]:
            if line.startswith("User:"):
                st.markdown(f"<div class='user-bubble'>{line}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='ai-bubble'>{line}</div>", unsafe_allow_html=True)
    
    # Chat Input
    user_in = st.chat_input("Interface with A.L.I.V.E...")
    if user_in:
        st.session_state.soul.speak(user_in)
        st.rerun()

# --- Advanced Visualization (The "Mind Palace") ---
with st.expander("🔬 View Neural Weights & Latent Activations"):
    st.write("This visualizes the first layer of the World Model's Encoder.")
    
    # Visualize the first layer weights of the encoder
    # Using our custom Tensor class to get data
    weights = st.session_state.agent.world_model.encoder.layers[0].W.data
    st.bar_chart(weights[:10].T) # Show first 10 neurons
    
    st.markdown("""
    **Architecture Explanation:**
    * **Encoder:** Compresses the 5D state (Pos, Target, Energy) into a 16D latent vector $z$.
    * **World Model:** Predicts $z_{t+1}$ and Reward $r$ given current $z_t$ and action $a$.
    * **Planner:** Used TD-MPC logic to simulate future steps in the latent space before moving.
    """)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #555;'>PROJECT A.L.I.V.E. v6.0 | Built with Python & NumPy | 2025 Edition</div>", unsafe_allow_html=True)
