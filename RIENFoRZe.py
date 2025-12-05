import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random
from collections import deque
import time

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Project Newborn: AI Lifeform",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧬"
)

# Custom CSS for that Sci-Fi Lab feel
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
        color: #e0e0e0;
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #00ddff;
        color: #000;
        border: none;
        border-radius: 4px;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #00ddff !important;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. THE BRAIN (NEURAL NETWORK)
# ==========================================
class DQNAgent:
    def __init__(self, state_size=4, action_size=4):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        
        # Hyperparameters
        self.gamma = 0.95    # Discount rate (Future vs Now)
        self.epsilon = 1.0   # Exploration rate (Curiosity)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        
        # Simple Neural Network Weights (NumPy implementation for speed in Streamlit)
        # Architecture: 4 (Input) -> 64 (Hidden) -> 32 (Hidden) -> 4 (Output)
        np.random.seed(42)
        self.W1 = np.random.randn(state_size, 64) / np.sqrt(state_size)
        self.b1 = np.zeros((1, 64))
        self.W2 = np.random.randn(64, 32) / np.sqrt(64)
        self.b2 = np.zeros((1, 32))
        self.W3 = np.random.randn(32, action_size) / np.sqrt(32)
        self.b3 = np.zeros((1, action_size))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_deriv(self, z):
        return (z > 0).astype(float)

    def forward(self, state):
        # Reshape state if necessary
        if state.ndim == 1:
            state = state.reshape(1, -1)
            
        # Layer 1
        z1 = np.dot(state, self.W1) + self.b1
        a1 = self.relu(z1)
        
        # Layer 2
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.relu(z2)
        
        # Output Layer (Q-Values)
        z3 = np.dot(a2, self.W3) + self.b3
        q_values = z3 # Linear activation for output
        
        # We return EVERYTHING so we can inspect the brain, 
        # but this caused the error before. We handle it in replay() now.
        return q_values, a1, a2, z1, z2

    def act(self, state):
        # Epsilon-greedy strategy (Explore vs Exploit)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values, _, _, _, _ = self.forward(state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return 0
        
        batch = random.sample(self.memory, batch_size)
        loss = 0
        
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                # FIX: Using *rest to ignore extra return values (activations)
                next_q_values, *rest = self.forward(next_state)
                target = reward + self.gamma * np.max(next_q_values)
            
            # Forward pass for current state
            current_q, a1, a2, z1, z2 = self.forward(state)
            
            # The target for the specific action we took
            target_f = current_q.copy()
            target_f[0][action] = target
            
            # Calculate Loss (Mean Squared Error) for visualization
            loss += np.mean((target_f - current_q)**2)
            
            # Backpropagation (simplified gradient descent)
            # Output Error
            error = target_f - current_q
            
            # Backprop through layers (Updating weights)
            dW3 = np.dot(a2.T, error)
            db3 = np.sum(error, axis=0, keepdims=True)
            
            d2 = np.dot(error, self.W3.T) * self.relu_deriv(z2)
            dW2 = np.dot(a1.T, d2)
            db2 = np.sum(d2, axis=0, keepdims=True)
            
            d1 = np.dot(d2, self.W2.T) * self.relu_deriv(z1)
            dW1 = np.dot(state.reshape(1,-1).T, d1)
            db1 = np.sum(d1, axis=0, keepdims=True)
            
            # Apply updates
            self.W3 += self.learning_rate * dW3
            self.b3 += self.learning_rate * db3
            self.W2 += self.learning_rate * dW2
            self.b2 += self.learning_rate * db2
            self.W1 += self.learning_rate * dW1
            self.b1 += self.learning_rate * db1
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss / batch_size

# ==========================================
# 3. APP LOGIC & STATE MANAGEMENT
# ==========================================

# Initialize Session State
if 'agent' not in st.session_state:
    st.session_state.agent = DQNAgent()
    st.session_state.agent_pos = np.array([50, 50]) # Start in middle of 100x100 grid
    st.session_state.target_pos = np.array([80, 20]) # Initial target
    st.session_state.step_count = 0
    st.session_state.rewards_history = []
    st.session_state.loss_history = []
    st.session_state.wins = 0

def reset_game():
    st.session_state.agent_pos = np.array([50, 50])
    st.session_state.target_pos = np.random.randint(0, 100, size=2)
    st.session_state.step_count = 0

# ==========================================
# 4. SIDEBAR DASHBOARD
# ==========================================
with st.sidebar:
    st.title("🔬 Lab Controls")
    
    st.metric("🧬 Generation (Steps)", st.session_state.step_count)
    st.metric("🍪 Targets Eaten", st.session_state.wins)
    st.metric("🧠 Brain Plasticity (Epsilon)", f"{st.session_state.agent.epsilon:.4f}")
    
    if st.button("Reset Simulation"):
        reset_game()
        st.session_state.agent = DQNAgent() # New brain
        
    st.markdown("### Neural Status")
    if len(st.session_state.loss_history) > 0:
        st.line_chart(st.session_state.loss_history[-50:])
        st.caption("Learning Loss (Lower is better)")

# ==========================================
# 5. MAIN SIMULATION AREA
# ==========================================
st.title("Project Newborn: The Learning AI")
st.markdown("This AI is a **Baby**. It starts knowing nothing. Click 'Take Step' to let it move and learn. It gets a 'cookie' (Reward +1) when it gets closer to the target.")

col1, col2 = st.columns([3, 1])

with col1:
    # ------------------------------------
    # A. VISUALIZATION (Plotly Grid World)
    # ------------------------------------
    fig = go.Figure()

    # The Arena
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, 
                  line=dict(color="RoyalBlue", width=2), fillcolor="rgba(0,0,0,0)")

    # The Agent (Baby)
    fig.add_trace(go.Scatter(
        x=[st.session_state.agent_pos[0]], 
        y=[st.session_state.agent_pos[1]],
        mode='markers+text',
        marker=dict(size=20, color='#00ddff', symbol='circle'),
        text=["👶"], textposition="top center",
        name='AI Agent'
    ))

    # The Target (Mouse Cursor / Food)
    fig.add_trace(go.Scatter(
        x=[st.session_state.target_pos[0]], 
        y=[st.session_state.target_pos[1]],
        mode='markers+text',
        marker=dict(size=15, color='#ff0055', symbol='x'),
        text=["🍪"], textposition="top center",
        name='Target'
    ))

    fig.update_layout(
        xaxis=dict(range=[-5, 105], showgrid=True, zeroline=False, visible=False),
        yaxis=dict(range=[-5, 105], showgrid=True, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------
    # B. MANUAL INTERACTION (Click to Move Target)
    # ------------------------------------
    # Note: Streamlit doesn't support direct click-to-coordinate easily without plugins.
    # We simulate it with sliders for robust functionality.
    st.markdown("### 🎯 Move the Target")
    new_tx = st.slider("Target X", 0, 100, int(st.session_state.target_pos[0]), key="sx")
    new_ty = st.slider("Target Y", 0, 100, int(st.session_state.target_pos[1]), key="sy")
    
    # Check if target moved manually
    if new_tx != st.session_state.target_pos[0] or new_ty != st.session_state.target_pos[1]:
        st.session_state.target_pos = np.array([new_tx, new_ty])
        st.rerun()

with col2:
    # ------------------------------------
    # C. GAME LOOP & TRAINING
    # ------------------------------------
    st.markdown("### 🕹️ Controls")
    
    # Calculate State: [AgentX, AgentY, TargetX, TargetY] normalized
    state = np.array([
        st.session_state.agent_pos[0]/100, 
        st.session_state.agent_pos[1]/100, 
        st.session_state.target_pos[0]/100, 
        st.session_state.target_pos[1]/100
    ])
    
    # Distance before moving
    dist_before = np.linalg.norm(st.session_state.agent_pos - st.session_state.target_pos)
    
    if st.button("RUN STEP (Train)", type="primary", use_container_width=True):
        
        # 1. AI Chooses Action
        action = st.session_state.agent.act(state)
        
        # 2. Perform Action (0: Up, 1: Down, 2: Left, 3: Right)
        move_dist = 5
        old_pos = st.session_state.agent_pos.copy()
        
        if action == 0: st.session_state.agent_pos[1] += move_dist # Up
        elif action == 1: st.session_state.agent_pos[1] -= move_dist # Down
        elif action == 2: st.session_state.agent_pos[0] -= move_dist # Left
        elif action == 3: st.session_state.agent_pos[0] += move_dist # Right
        
        # Boundary Check
        st.session_state.agent_pos = np.clip(st.session_state.agent_pos, 0, 100)
        
        # 3. Calculate Reward
        dist_after = np.linalg.norm(st.session_state.agent_pos - st.session_state.target_pos)
        
        reward = 0
        done = False
        
        if dist_after < 5: # Reached target
            reward = 10
            done = True
            st.session_state.wins += 1
            st.toast("🍪 Yummy! Target reached!", icon="🎉")
            st.session_state.target_pos = np.random.randint(10, 90, size=2)
        elif dist_after < dist_before:
            reward = 1 # Good boy, getting closer
        else:
            reward = -1 # Wrong way!
            
        # 4. Remember & Train
        next_state = np.array([
            st.session_state.agent_pos[0]/100, 
            st.session_state.agent_pos[1]/100, 
            st.session_state.target_pos[0]/100, 
            st.session_state.target_pos[1]/100
        ])
        
        st.session_state.agent.remember(state, action, reward, next_state, done)
        loss = st.session_state.agent.replay(batch_size=8) # Small batch for realtime feel
        
        if loss:
            st.session_state.loss_history.append(loss)
            
        st.session_state.step_count += 1
        st.rerun()

    # Debug Info
    st.markdown("---")
    st.write(f"**Distance:** {dist_before:.1f}")
    st.write(f"**Last Action:** {['Up','Down','Left','Right'][st.session_state.agent.act(state)]} (Predicted)")
