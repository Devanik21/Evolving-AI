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
# 2. THE ADVANCED MIND (Titan Architecture)
# ==========================================

class AdamOptimizer:
    """
    Implements the Adam algorithm (Adaptive Moment Estimation).
    This gives the AI 'momentum' in learning, allowing it to navigate 
    complex strategies much faster than standard SGD.
    """
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        for k in params.keys():
            if k in grads:
                # 1. Update biased first moment estimate
                self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
                # 2. Update biased second raw moment estimate
                self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k]**2)
                # 3. Compute bias-corrected first moment estimate
                m_hat = self.m[k] / (1 - self.beta1**self.t)
                # 4. Compute bias-corrected second raw moment estimate
                v_hat = self.v[k] / (1 - self.beta2**self.t)
                # 5. Update parameters
                params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class TitanBrain:
    """
    The 'Princess' Upgrade.
    Architecture: Deep Dueling DQN with 2 Hidden Layers + Adam Optimizer.
    """
    def __init__(self, state_size=5, action_size=4, buffer_size=20000, hidden_size=128):
        self.state_size = state_size
        self.action_size = action_size
        # Increased hidden size for 'Deep' thought
        self.hidden_1 = hidden_size 
        self.hidden_2 = int(hidden_size / 2) 
        
        self.memory = PrioritizedReplayBuffer(buffer_size)
        
        # Hyperparameters
        self.gamma = 0.98        # Higher foresight
        self.epsilon = 1.0   
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995 # Slower decay for more exploration
        self.learning_rate = 0.001 
        self.beta = 0.4 
        self.beta_increment = 0.001
        
        # Initialize Networks with He-Initialization (Better for ReLU)
        self.online_net = self.init_network()
        self.target_net = self.init_network()
        
        # Attach Optimizers (One per network isn't needed, just one for Online)
        self.optimizer = AdamOptimizer(self.online_net, lr=self.learning_rate)
        
        self.update_target_network()

    def init_network(self):
        # He-Initialization: randn * sqrt(2/n)
        return {
            # Layer 1: Input -> Hidden 1
            'W1': np.random.randn(self.state_size, self.hidden_1) * np.sqrt(2/self.state_size),
            'b1': np.zeros((1, self.hidden_1)),
            
            # Layer 2: Hidden 1 -> Hidden 2 (The "Deep" Layer)
            'W2': np.random.randn(self.hidden_1, self.hidden_2) * np.sqrt(2/self.hidden_1),
            'b2': np.zeros((1, self.hidden_2)),
            
            # Dueling Stream: Value (State Value)
            'W_val': np.random.randn(self.hidden_2, 1) * np.sqrt(2/self.hidden_2),
            'b_val': np.zeros((1, 1)),
            
            # Dueling Stream: Advantage (Action Value)
            'W_adv': np.random.randn(self.hidden_2, self.action_size) * np.sqrt(2/self.hidden_2),
            'b_adv': np.zeros((1, self.action_size))
        }

    def update_target_network(self):
        self.target_net = {k: v.copy() for k, v in self.online_net.items()}

    def relu(self, z):
        return np.maximum(0, z)
        
    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def forward(self, state, network):
        if state.ndim == 1: state = state.reshape(1, -1)
        
        # Layer 1
        z1 = np.dot(state, network['W1']) + network['b1']
        a1 = self.relu(z1)
        
        # Layer 2 (Deep abstraction)
        z2 = np.dot(a1, network['W2']) + network['b2']
        a2 = self.relu(z2)
        
        # Dueling Heads
        val = np.dot(a2, network['W_val']) + network['b_val']
        adv = np.dot(a2, network['W_adv']) + network['b_adv']
        
        # Aggregation
        q_values = val + (adv - np.mean(adv, axis=1, keepdims=True))
        return q_values, a1, a2, z1, z2

    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values, _, _, _, _ = self.forward(state, self.online_net)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size: return 0, 0
        
        batch, indices, weights = self.memory.sample(batch_size, self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Accumulators for gradients
        grads = {k: np.zeros_like(v) for k, v in self.online_net.items()}
        total_loss = 0
        new_priorities = []
        
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            state = state.reshape(1, -1)
            next_state = next_state.reshape(1, -1)
            
            # 1. Double DQN Target
            target = reward
            if not done:
                # Select action using Online Net
                next_q_online, _, _, _, _ = self.forward(next_state, self.online_net)
                best_action = np.argmax(next_q_online[0])
                # Evaluate action using Target Net
                next_q_target, _, _, _, _ = self.forward(next_state, self.target_net)
                target = reward + self.gamma * next_q_target[0][best_action]
            
            # 2. Forward Pass
            current_q, a1, a2, z1, z2 = self.forward(state, self.online_net)
            
            # 3. Huber Loss Calculation
            td_error = target - current_q[0][action]
            new_priorities.append(abs(td_error))
            
            # Huber Loss derivative clipping
            error_clipped = np.clip(td_error, -1.0, 1.0)
            weighted_error = error_clipped * weights[i]
            
            total_loss += weighted_error ** 2
            
            # 4. Backpropagation (Deep Network)
            
            # -- Output Layer Gradients --
            grad_val = weighted_error * a2.T # (hidden2, 1)
            grad_adv = np.zeros_like(self.online_net['W_adv'])
            grad_adv[:, action] = weighted_error * a2[0] 
            
            # -- Propagate to Hidden Layer 2 --
            # Error comes from both Val and Adv heads
            error_at_h2 = (np.dot(self.online_net['W_val'], weighted_error) + 
                           np.dot(self.online_net['W_adv'][:, action].reshape(-1, 1), weighted_error)).T
            
            # Apply ReLU derivative for Layer 2
            delta_2 = error_at_h2 * self.relu_derivative(z2)
            
            grad_w2 = np.dot(a1.T, delta_2)
            grad_b2 = delta_2
            
            # -- Propagate to Hidden Layer 1 (The new deep layer) --
            error_at_h1 = np.dot(delta_2, self.online_net['W2'].T)
            delta_1 = error_at_h1 * self.relu_derivative(z1)
            
            grad_w1 = np.dot(state.T, delta_1)
            grad_b1 = delta_1
            
            # Accumulate
            grads['W_val'] += grad_val
            grads['W_adv'] += grad_adv
            grads['b_val'] += weighted_error # Bias gradient is just the error
            grads['b_adv'][:, action] += weighted_error
            grads['W2'] += grad_w2
            grads['b2'] += grad_b2
            grads['W1'] += grad_w1
            grads['b1'] += grad_b1

        # 5. Optimization Step (ADAM)
        # Average gradients across batch
        for k in grads: grads[k] /= batch_size
        self.optimizer.update(self.online_net, grads)

        self.memory.update_priorities(indices, new_priorities)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return total_loss / batch_size, np.mean(new_priorities)

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
if 'config' not in st.session_state:
    st.session_state.config = {
        # MOVEMENT
        "sim_speed": 0.05,       # Faster animation
        "move_speed": 12.0,      # Faster movement (Agent covers ground quickly)
        "energy_decay": 0.05,    # Slower energy loss (Less punishment)
        
        # REWARDS
        "shaping_multiplier": 5.0, # HUGE reward for moving towards target (Guidance)
        "hug_reward": 200.0,       # Big payout for winning
        "hug_distance": 10.0,      # Easier to catch the target
        
        # BRAIN PARAMETERS (The Secret Sauce)
        "learning_rate": 0.01,    # 10x Higher! (Force it to learn FAST)
        "gamma": 0.90,            # Lower Gamma (Care about NOW, not later)
        "epsilon_decay": 0.98,    # Decay exploration fast (Stop acting random quickly)
        "epsilon_min": 0.01,      # Allow it to use 99% of its brain eventually
        
        # ARCHITECTURE
        "batch_size": 64,         # Learn from more memories at once
        "hidden_size": 32,        # SMALLER BRAIN = Faster Learning (32 is enough!)
        "buffer_size": 5000,      # Smaller memory (Keep only recent/relevant stuff)
        
        # PER (Memory Priority)
        "per_alpha": 0.6, "per_beta": 0.4, "per_beta_increment": 0.001,
        
        # VISUALS
        "target_update_freq": 20, # Update the "Goal" network more often
        "grid_h": 15, "grid_w": 40, "graph_points": 500
    }

# --- CRITICAL FIX: DETECT OLD BRAIN AND DELETE IT ---
if 'mind' in st.session_state:
    # If the brain in memory doesn't have an optimizer, it's the old version!
    if not hasattr(st.session_state.mind, 'optimizer'):
        st.toast("⚠️ Old Brain Detected. Performing Transplant...", icon="🏥")
        del st.session_state['mind']
        if 'soul' in st.session_state: del st.session_state['soul']
        st.rerun() # Restart to apply changes

# 2. Initialize Mind (Using TitanBrain)
if 'mind' not in st.session_state:
    # IMPORTANT: Make sure you are using TitanBrain here, not AdvancedMind
    st.session_state.mind = TitanBrain(
        buffer_size=st.session_state.config['buffer_size'], 
        hidden_size=st.session_state.config['hidden_size']
    )
    st.session_state.soul = AGICore() 
    st.session_state.agent_pos = np.array([50.0, 50.0])
    st.session_state.target_pos = np.array([80.0, 20.0])
    st.session_state.step_count = 0
    st.session_state.wins = 0
    st.session_state.auto_mode = False
    st.session_state.chat_history = []
    st.session_state.loss_history = []
    st.session_state.reward_history = []
    st.session_state.is_hugging = False

# 3. Ensure History Lists Exist
if 'loss_history' not in st.session_state:
    st.session_state.loss_history = []
    st.session_state.reward_history = []
    

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
    """Saves state using JSON (metadata) and NPZ (arrays) inside a ZIP."""
    
    # 1. PREPARE METADATA (JSON safe)
    # We convert numpy scalars to native python types (float/int) for JSON compliance
    metadata = {
        'version': 3.0,
        'wins': int(st.session_state.wins),
        'steps': int(st.session_state.step_count),
        'epsilon': float(st.session_state.mind.epsilon),
        'beta': float(st.session_state.mind.beta),
        'soul': {
            'name': st.session_state.soul.user_name,
            'rel': int(st.session_state.soul.relationship_score),
            'mood': st.session_state.soul.current_mood,
            'energy': float(st.session_state.soul.energy)
        },
        'config': st.session_state.config,
        # Save history as simple lists
        'loss_hist': [float(x) for x in st.session_state.loss_history],
        'reward_hist': [float(x) for x in st.session_state.reward_history]
    }

    # 2. PREPARE MEMORY (Convert Deque of Tuples -> Separate Numpy Arrays)
    # This is much faster and safer than pickling objects
    buffer_list = list(st.session_state.mind.memory.buffer)
    if len(buffer_list) > 0:
        mem_states = np.array([x[0] for x in buffer_list])
        mem_actions = np.array([x[1] for x in buffer_list])
        mem_rewards = np.array([x[2] for x in buffer_list])
        mem_next_states = np.array([x[3] for x in buffer_list])
        mem_dones = np.array([x[4] for x in buffer_list])
        mem_priorities = st.session_state.mind.memory.priorities
        mem_pos = np.array([st.session_state.mind.memory.pos])
    else:
        # Handle empty memory case
        mem_states = np.array([])
        mem_actions = np.array([]) 
        mem_rewards = np.array([])
        mem_next_states = np.array([])
        mem_dones = np.array([])
        mem_priorities = np.zeros(st.session_state.config['buffer_size'])
        mem_pos = np.array([0])

    # 3. CREATE ZIP IN MEMORY
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # A. Write Metadata JSON
        zf.writestr("metadata.json", json.dumps(metadata, indent=4))
        
        # B. Write Neural Weights (Online Net)
        weights_buffer = io.BytesIO()
        np.savez(weights_buffer, **st.session_state.mind.online_net)
        zf.writestr("online_net.npz", weights_buffer.getvalue())
        
        # C. Write Target Net
        target_buffer = io.BytesIO()
        np.savez(target_buffer, **st.session_state.mind.target_net)
        zf.writestr("target_net.npz", target_buffer.getvalue())
        
        # D. Write Optimizer State
        opt_buffer = io.BytesIO()
        np.savez(opt_buffer, 
                 t=np.array([st.session_state.mind.optimizer.t]),
                 **{f"m_{k}": v for k, v in st.session_state.mind.optimizer.m.items()},
                 **{f"v_{k}": v for k, v in st.session_state.mind.optimizer.v.items()}) # Merge dicts hack
        # Note: We split m and v keys to save in one file
        # Actually simpler: save m and v in separate files or loops. 
        # Let's do separate files for clarity to avoid key collisions.
        
        # Saving Optimizer M (Momentum)
        m_buffer = io.BytesIO()
        np.savez(m_buffer, **st.session_state.mind.optimizer.m)
        zf.writestr("opt_m.npz", m_buffer.getvalue())

        # Saving Optimizer V (Velocity)
        v_buffer = io.BytesIO()
        np.savez(v_buffer, **st.session_state.mind.optimizer.v)
        zf.writestr("opt_v.npz", v_buffer.getvalue())

        # E. Write Memory
        mem_buffer = io.BytesIO()
        np.savez(mem_buffer, 
                 states=mem_states, actions=mem_actions, 
                 rewards=mem_rewards, next_states=mem_next_states, 
                 dones=mem_dones, priorities=mem_priorities, pos=mem_pos)
        zf.writestr("memory.npz", mem_buffer.getvalue())

    return zip_buffer.getvalue()

def load_brain(uploaded_file):
    """Restores state from the 'Safe' ZIP format."""
    try:
        with zipfile.ZipFile(uploaded_file, "r") as z:
            # 1. LOAD METADATA
            with z.open("metadata.json") as f:
                meta = json.load(f)
            
            # Apply Config First
            st.session_state.config.update(meta['config'])
            
            # Apply Stats
            st.session_state.wins = meta['wins']
            st.session_state.step_count = meta['steps']
            st.session_state.loss_history = meta['loss_hist']
            st.session_state.reward_history = meta['reward_hist']
            
            # Apply Soul
            st.session_state.soul.user_name = meta['soul']['name']
            st.session_state.soul.relationship_score = meta['soul']['rel']
            st.session_state.soul.current_mood = meta['soul']['mood']
            st.session_state.soul.energy = meta['soul']['energy']
            
            # Apply Mind Scalars
            st.session_state.mind.epsilon = meta['epsilon']
            st.session_state.mind.beta = meta['beta']
            
            # 2. LOAD WEIGHTS (Using np.load)
            # Helper to read .npz from zip
            def load_npz(filename):
                with z.open(filename) as f:
                    buf = io.BytesIO(f.read())
                    return np.load(buf)

            # Online Net
            raw_online = load_npz("online_net.npz")
            st.session_state.mind.online_net = {k: raw_online[k] for k in raw_online.files}

            # Target Net
            raw_target = load_npz("target_net.npz")
            st.session_state.mind.target_net = {k: raw_target[k] for k in raw_target.files}
            
            # Optimizer
            raw_m = load_npz("opt_m.npz")
            st.session_state.mind.optimizer.m = {k: raw_m[k] for k in raw_m.files}
            
            raw_v = load_npz("opt_v.npz")
            st.session_state.mind.optimizer.v = {k: raw_v[k] for k in raw_v.files}
            
            # 3. LOAD MEMORY
            mem_data = load_npz("memory.npz")
            
            # Reconstruct Buffer (The tricky part!)
            if 'states' in mem_data and len(mem_data['states']) > 0:
                s = mem_data['states']
                a = mem_data['actions']
                r = mem_data['rewards']
                ns = mem_data['next_states']
                d = mem_data['dones']
                
                # Zip them back into a list of tuples
                # We iterate up to the stored length
                reconstructed_buffer = []
                for i in range(len(s)):
                    reconstructed_buffer.append( (s[i], int(a[i]), r[i], ns[i], bool(d[i])) )
                
                st.session_state.mind.memory.buffer = reconstructed_buffer
                st.session_state.mind.memory.priorities = mem_data['priorities']
                st.session_state.mind.memory.pos = int(mem_data['pos'][0])

        st.toast("✅ Brain Successfully Implanted! No Errors.", icon="🧠")
        time.sleep(1)
        
    except Exception as e:
        st.error(f"Critical System Failure during Load: {e}")





def reset_simulation():
    """Resets the agent, target, and stats."""
    st.session_state.agent_pos = np.array([50.0, 50.0])
    st.session_state.target_pos = np.array([80.0, 20.0])
    st.session_state.soul = AGICore() # Reset the AI Personality
    st.session_state.mind = TitanBrain(
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
    with st.expander("💾 Memory Card (Safe Mode)", expanded=True):
        
        # 1. DOWNLOAD
        if st.button("📦 Create Backup (.zip)"):
            zip_data = save_brain()
            st.download_button(
                label="⬇️ Download Brain",
                data=zip_data,
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
