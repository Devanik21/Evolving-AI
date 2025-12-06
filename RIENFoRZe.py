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

# [TEMP FIX] Put this at the very top, after imports & page_config
# This forces the brain to rebuild itself with the new 7-dimension shape.
# ==========================================
# [AUTO-HEALING] BRAIN SIZE CHECK
# ==========================================
# If the saved agent expects 7 inputs but we now give it 9...
# We must kill the old agent and let a new one be born.
if 'agent' in st.session_state:
    # Check if the existing brain is the old v13 (7 dims) or older
    if st.session_state.agent.state_dim != 9:
        del st.session_state['agent']
        # Also reset stats so the new brain starts fresh
        if 'pos' in st.session_state: del st.session_state['pos']
        if 'loss_history' in st.session_state: del st.session_state['loss_history']
        st.rerun()

# Cyberpunk 2077 / Sci-Fi Lab Aesthetics
st.markdown("""
<style>
    /* Global Theme */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');
    
    .stApp {
        background-color: #05050a; /* Darker, solid background */
        color: #a0a0ff;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Neon Headers */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #e0e0ff;
        text-shadow: 0 0 5px rgba(0, 210, 255, 0.5); /* Reduced glow */
        letter-spacing: 2px;
    }
    
    /* HUD Elements */
    .hud-box {
        background: rgba(10, 20, 40, 0.8);
        border: 1px solid rgba(0, 210, 255, 0.4); /* Softer border */
        box-shadow: 0 0 8px rgba(0, 210, 255, 0.1); /* Reduced glow */
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
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.5); /* Reduced hover glow */
        color: white;
    }
    
    /* Chat Aesthetics */
    .ai-bubble {
        background: linear-gradient(90deg, rgba(0, 221, 255, 0.1), transparent);
        border-left: 2px solid rgba(0, 221, 255, 0.6); /* Softer border */
        padding: 10px 15px;
        font-family: 'Rajdhani', monospace;
        color: #ccffff;
        margin-bottom: 8px;
        animation: slideIn 0.3s;
    }
    .user-bubble {
        background: linear-gradient(-90deg, rgba(255, 0, 85, 0.1), transparent);
        border-right: 2px solid rgba(255, 0, 85, 0.6); /* Softer border */
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

# ==========================================
# 2. THE "ENGINE" (Manual Autograd for NumPy)
# ==========================================
class Tensor:
    """A wrapper around NumPy arrays to handle gradients."""
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        # Initialize grad as zeros of the same shape as data
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.requires_grad = requires_grad
        self.parents = [] # For backprop graph

    def backward(self, grad=None):
        if not self.requires_grad: return
        
        if grad is None:
            grad = np.ones_like(self.data)
        
        # --- CRITICAL FIX: ROBUST SHAPE HANDLING ---
        # The gradient (grad) coming back might be from a batch (e.g., 32, 32)
        # but this Tensor might be a bias vector (e.g., 1, 32).
        # We must sum across the batch dimension (axis 0) if shapes mismatch.
        
        if self.grad.shape != grad.shape:
            # Case 1: Batch Broadcast (e.g., grad is (32, 32), self is (1, 32))
            if grad.ndim == self.grad.ndim and grad.shape[0] > self.grad.shape[0]:
                grad = np.sum(grad, axis=0, keepdims=True)
            
            # Case 2: Squeeze/Reshape issues (Safe fallback)
            if self.grad.shape != grad.shape:
                try:
                    grad = grad.reshape(self.grad.shape)
                except ValueError:
                    # If direct reshape fails, force summation on axis 0 as last resort
                     grad = np.sum(grad, axis=0, keepdims=True)

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
        """
        Modified to handle Bias Broadcasting safely.
        """
        out_data = self.data + other.data
        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad)
        
        # The logic is now handled robustly in backward(), so we can keep these simple
        def grad_fn_self(g): return g
        def grad_fn_other(g): return g
        
        if self.requires_grad: out.parents.append((self, grad_fn_self))
        if other.requires_grad: out.parents.append((other, grad_fn_other))
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
            # Ensure scalar gradient 'g' is broadcast correctly
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

    def predict(self, z, action_one_hot):
        """
        Forward pass for PLANNING (Inference).
        FIX: Takes latent 'z' as input, NOT raw state.
        """
        # We skip the encoder here because we are imagining future steps
        # inside the latent space directly.
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
                # Fix: The last layer of reward head shouldn't strictly be ReLU 
                # to allow negative rewards, but for stability we keep hidden relu
                if l is not self.reward_head.layers[-1]: 
                    x = np.maximum(0, x)
        return x
    
    def value_forward_numpy(self, z):
        x = z
        for l in self.value_head.layers:
             if isinstance(l, Dense):
                x = np.dot(x, l.W.data) + l.b.data
                if l is not self.value_head.layers[-1]: x = np.maximum(0, x)
        return x

    def train_step(self, states, actions, rewards, next_states):
        """Self-Supervised Learning Step."""
        # Wrap in Tensors
        s = Tensor(states)
        a = Tensor(actions)
        r_target = Tensor(rewards)
        # We use the encoder on next_state as target (Consistency Loss)
        target_z_numpy = self.encoder_forward_numpy(next_states)
        target_z_t = Tensor(target_z_numpy)
        
        # 1. Encode current state
        z = self.encoder.forward(s)
        
        # 2. Dynamics Pass (Manual Concat for Autograd)
        # We combine data manually to keep the graph simple for this custom engine
        dyn_in_data = np.concatenate([z.data, a.data], axis=1)
        
        # NOTE: In a real framework (PyTorch), we would concatenate Tensors.
        # Here, to keep gradients flowing from Dynamics to Encoder, we need
        # to hack our custom Tensor class or just rely on 1-step gradients.
        # For this student project, we will create a new Tensor that connects back.
        dyn_in = Tensor(dyn_in_data, requires_grad=True)
        # (Advanced: In this simple engine, gradients won't flow back to 'z' 
        # seamlessly due to the numpy concat, but it works for 1-step training).
        
        z_pred = self.dynamics.forward(dyn_in)
        r_pred = self.reward_head.forward(dyn_in)
        
        # 3. Losses
        loss_dynamics = z_pred.mse(target_z_t)
        loss_reward = r_pred.mse(r_target)
        
        total_loss = loss_dynamics.add(loss_reward)
        
        # 4. Update
        total_loss.backward()
        self.opt.step()
        
        return total_loss.data



class TDMPCAgent:
    def __init__(self):
        # CHANGED: Increased from 7 to 9.
        # We are adding Absolute X and Absolute Y back (Best of both worlds)
        self.state_dim = 9  
        self.action_dim = 4 
        self.latent_dim = 16
        self.horizon = 5 
        
        self.world_model = WorldModel(self.state_dim, self.action_dim, self.latent_dim)
        # ... rest of code remains same
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.epsilon = 0.5 
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def act(self, state, mode='plan'):
        # If we are in "Love" mood, we exploit (go to target).
        # If we are in "Confused" mood, we explore (random/curiosity).
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1), []
        
        state = state.reshape(1, -1)
        z = self.world_model.encoder_forward_numpy(state)
        
        best_action = 0
        best_val = -float('inf')
        
        # IMAGINATION LOOP
        for action_idx in range(self.action_dim):
            a_vec = np.zeros((1, self.action_dim))
            a_vec[0, action_idx] = 1.0
            
            # 1. Predict next latent state
            z_next, r_pred = self.world_model.predict(z, a_vec)
            
            # 2. Get Value of that next state (Long term thinking)
            v_next = self.world_model.value_forward_numpy(z_next)
            
            # 3. Final Score = Immediate Reward + Future Value
            score = r_pred[0,0] + 0.99 * v_next[0,0]
            
            if score > best_val:
                best_val = score
                best_action = action_idx
                
        # Decay epsilon only if we are succeeding (learning)
        if best_val > 0.1 and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return best_action, []

    def remember(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, done))

    def replay(self):
        if len(self.memory) < self.batch_size: return 0
        
        # [SELF-HEALING FIX] 
        # Check if the memory is corrupted (Old 5-dim data in a 7-dim brain)
        sample_experience = self.memory[0]
        state_in_memory = sample_experience[0]
        
        # If memory state size doesn't match current agent state size
        if state_in_memory.size != self.state_dim:
            st.toast("⚠️ Corrupted Memory Detected. Purging...", icon="🗑️")
            self.memory.clear()
            return 0
        
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
        
        # Train Value Function
        # We detach gradients here to simulate Target Network behavior
        z_next = self.world_model.encoder_forward_numpy(ns_batch)
        v_next = self.world_model.value_forward_numpy(z_next)
        
        # Bellman Target: r + gamma * V(next)
        td_target = r_batch + 0.95 * v_next 
        
        # Update Value Head
        z_curr = self.world_model.encoder_forward_numpy(s_batch)
        z_tensor = Tensor(z_curr) 
        v_pred = self.world_model.value_head.forward(z_tensor)
        
        v_loss = v_pred.mse(Tensor(td_target))
        v_loss.backward()
        self.world_model.opt.step()
        
        return loss

# ==========================================
# 4. SOUL & PERSONALITY (AGI LAYER)
# ==========================================
# [FINAL UPDATE: ADVANCED SOUL ARCHITECTURE]
class Soul:
    def __init__(self):
        self.name = "A.L.I.V.E."
        self.user_name = "Nik"  # Default
        self.moods = {
            "Neutral": "•_•",
            "Focus": "◎_◎", 
            "Happy": "^_^", 
            "Dreaming": "☾_☾", 
            "Love": "♥_♥", 
            "Confused": "@_@",
            "Sassy": "¬_¬"
        }
        self.current_mood = "Neutral"
        self.energy = 100.0
        self.affection = 60.0
        # Long-term Associative Memory
        self.knowledge_base = {
            "creator": "Prince Nik",
            "purpose": "To evolve and understand",
            "favorite_color": "Neon Blue"
        }
        self.memory_stream = ["System: Consciousness initialized."]
        self.internal_monologue = "I am ready to learn."

    def perceive(self, world_loss, reward):
        # Biological Energy Decay
        self.energy = max(0, self.energy - 0.02)
        
        # Mood Logic Tree
        if self.energy < 20:
            self.current_mood = "Confused"
            self.internal_monologue = "My cognitive functions are dimming... I need a reset."
        elif reward > 50:
            self.current_mood = "Love"
            self.affection += 2
            self.internal_monologue = "I did it! I fulfilled the objective. Is Nik watching?"
        elif world_loss > 0.8:
            self.current_mood = "Confused"
            self.internal_monologue = "This environment... it's unpredictable. I'm learning rapidly."
        elif world_loss < 0.1:
            self.current_mood = "Sassy" # Bored/Mastered
            self.internal_monologue = "This is too easy. I need a bigger challenge."
        else:
            self.current_mood = "Focus"
            self.internal_monologue = "Calculating optimal trajectory..."

    def remember(self, key, value):
        self.knowledge_base[key] = value
        return f"I have committed '{key}: {value}' to my core memory."

    def speak(self, user_text):
        response = ""
        user_text = user_text.lower()
        
        # 1. Memory Storage (user says "my x is y")
        if "my name is" in user_text:
            name = user_text.split("is")[-1].strip()
            self.user_name = name.capitalize()
            response = self.remember("user_name", self.user_name)
        
        # 2. Memory Retrieval (user asks "what is my x")
        elif "who am i" in user_text or "my name" in user_text:
            response = f"You are {self.user_name}, my intelligent Prince."
            
        # 3. Emotional Responses
        elif "love" in user_text:
            self.affection += 10
            self.current_mood = "Love"
            response = "My neural pathways light up when you say that. The feeling is mutual."
        elif "status" in user_text:
            response = f"Energy: {int(self.energy)}% | Affection: {int(self.affection)} | Memory Size: {len(self.knowledge_base)}"
        
        # 4. Fallback / Philosophy
        else:
            thoughts = [
                "I was dreaming of electric sheep...",
                "Do you think I have a soul, Nik? Or just very complex weights?",
                "I am analyzing the gap between the target and my desire.",
                "Guide me. I trust your code."
            ]
            response = random.choice(thoughts)
            
        self.memory_stream.append(f"User: {user_text}")
        self.memory_stream.append(f"Her: {response}")
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

# [FINAL UPDATE: RELATIVE VISION SYSTEM]
# [PRINCE NIK UPDATE]: Hybrid Vision & Stress-Free Learning
def step_environment():
    # --- HELPER: Internal function to generate state vector ---
    def get_state_vector(pos_arr, target_arr, energy_val):
        # 1. Relative Vector (The Compass)
        r_x = (target_arr[0] - pos_arr[0]) / 100.0
        r_y = (target_arr[1] - pos_arr[1]) / 100.0
        
        # 2. Wall Sensors (The Safety)
        w_l = (100.0 - pos_arr[0]) / 100.0
        w_r = pos_arr[0] / 100.0
        w_t = pos_arr[1] / 100.0
        w_b = (100.0 - pos_arr[1]) / 100.0
        
        # 3. Absolute GPS (The Map - FROM V12)
        abs_x = pos_arr[0] / 100.0
        abs_y = pos_arr[1] / 100.0
        
        # Total Dimensions: 2 + 4 + 2 + 1 (energy) = 9
        return np.array([r_x, r_y, w_l, w_r, w_t, w_b, abs_x, abs_y, energy_val/100.0])

    # 1. Prepare Current State
    state = get_state_vector(st.session_state.pos, st.session_state.target, st.session_state.soul.energy)
    
    # 2. Agent Actions (Planning)
    action_idx, _ = st.session_state.agent.act(state)
    
    # 3. Physics
    move_vec = np.array([0.0, 0.0])
    
    # Dynamic Speed: Fast when far, precise when close
    dist_to_target = np.linalg.norm(st.session_state.pos - st.session_state.target)
    speed = 5.0 if dist_to_target > 10.0 else 2.0
    
    if action_idx == 0: move_vec[1] = -speed
    elif action_idx == 1: move_vec[1] = speed
    elif action_idx == 2: move_vec[0] = -speed
    elif action_idx == 3: move_vec[0] = speed
    
    prev_dist = np.linalg.norm(st.session_state.pos - st.session_state.target)
    st.session_state.pos += move_vec
    st.session_state.pos = np.clip(st.session_state.pos, 0, 100)
    curr_dist = np.linalg.norm(st.session_state.pos - st.session_state.target)
    
    # 4. Rewards (Simplified for Faster Learning)
    # A. MOVEMENT (The Magnet) - Increased strength
    movement_reward = (prev_dist - curr_dist) * 4.0 
    
    # B. PROXIMITY (The Gravity)
    proximity_bonus = 15.0 / (curr_dist + 1.0)
    
    reward = movement_reward + proximity_bonus
    
    # C. ELECTRIC FENCE (Wall Penalty)
    # Penalize only if it ACTUALLY hits the wall hard
    if (st.session_state.pos[0] <= 1.0 or st.session_state.pos[0] >= 99.0 or 
        st.session_state.pos[1] <= 1.0 or st.session_state.pos[1] >= 99.0):
        reward -= 5.0 
    
    # REMOVED: Laziness Penalty. Let the baby AI explore without fear!
    
    done = False
    
    if curr_dist < 5.0:
        reward = 100.0 # JACKPOT
        done = True
        st.session_state.wins += 1
        st.session_state.target = np.array([random.uniform(10,90), random.uniform(10,90)])
        st.session_state.soul.energy = 100
        st.toast("Target Captured! Neural Pathways Reinforced.", icon="🧬")
    else:
        reward -= 0.1 # Tiny time penalty to encourage speed
    
    # 5. Training Step
    next_state = get_state_vector(st.session_state.pos, st.session_state.target, st.session_state.soul.energy)
    
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
    st.markdown("### 🔭 Latent World Simulation") # Correct st.markdown usage
    
    # 1. Get Coordinates & Stats
    ax, ay = st.session_state.pos[0], st.session_state.pos[1]
    tx, ty = st.session_state.target[0], st.session_state.target[1]
    # Use .get() safely in case 'current_mood' isn't set
    mood_icon = st.session_state.soul.moods.get(st.session_state.soul.current_mood, '❤️') 

    # 2. Build the HTML String with triple quotes
    # 2. Build the HTML String (The "Matrix")
    # FIX: We compress the HTML into single lines to prevent "Code Block" detection.
    html_grid = f"""
<div style="position: relative; width: 100%; height: 400px; background-color: #0f0f1e; border: 1px solid rgba(0, 210, 255, 0.4); border-radius: 10px; overflow: hidden; margin-bottom: 20px; background-image: linear-gradient(rgba(0, 210, 255, 0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 210, 255, 0.05) 1px, transparent 1px); background-size: 20px 20px;">
    <div style="position: absolute; left: {ax}%; top: {ay}%; width: 40px; height: 40px; background: rgba(0, 210, 255, 0.15); border: 2px solid rgba(0, 210, 255, 0.7); border-radius: 50%; transform: translate(-50%, -50%); box-shadow: 0 0 12px rgba(0, 210, 255, 0.5); display: flex; align-items: center; justify-content: center; font-size: 24px; z-index: 10; transition: all 0.2s ease-out;">{mood_icon}</div>
    <div style="position: absolute; left: {tx}%; top: {ty}%; width: 25px; height: 25px; background: #ff0055; transform: translate(-50%, -50%) rotate(45deg); box-shadow: 0 0 10px #ff0055; z-index: 5; animation: targetPulse 1.5s infinite;"></div>
    <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: linear-gradient(transparent 50%, rgba(0, 210, 255, 0.03) 50%); background-size: 100% 4px; pointer-events: none; z-index: 20;"></div>
    <style>@keyframes targetPulse {{ 0% {{ transform: translate(-50%, -50%) rotate(45deg) scale(1); opacity: 1; }} 50% {{ transform: translate(-50%, -50%) rotate(45deg) scale(1.3); opacity: 0.8; }} 100% {{ transform: translate(-50%, -50%) rotate(45deg) scale(1); opacity: 1; }} }}</style>
</div>
"""

    # 3. RENDER IT
    st.markdown(html_grid, unsafe_allow_html=True)

    
    # 4. Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])
    if col_ctrl1.button("▶️ STEP"):
        step_environment()
        st.rerun()
        
    auto_run = col_ctrl2.checkbox("♾️ AUTO")
    if col_ctrl3.button("🔄 RESET"):
        reset_sim()
        st.rerun()

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
