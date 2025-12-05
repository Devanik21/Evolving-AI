import streamlit as st
import numpy as np
import pandas as pd
import json
from collections import deque
import time

# Page config
st.set_page_config(
    page_title="AI Autonomous Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    h1, h2, h3, p, label {
        color: #e0e0e0 !important;
    }
    .metric-card {
        background: rgba(26, 26, 46, 0.6);
        border: 1px solid rgba(100, 200, 255, 0.3);
        border-radius: 8px;
        padding: 15px;
        backdrop-filter: blur(10px);
    }
    .status-active {
        color: #00ff88;
        font-weight: bold;
    }
    .status-learning {
        color: #ffaa00;
        font-weight: bold;
    }
    .status-converged {
        color: #00aaff;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Neural Network (Simple DQN)
class DQNAgent:
    def __init__(self, state_size=4, action_size=4, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        
        # Neural network weights (simplified)
        self.weights = {
            'layer1': np.random.randn(state_size, 64) * 0.1,
            'bias1': np.zeros(64),
            'layer2': np.random.randn(64, 32) * 0.1,
            'bias2': np.zeros(32),
            'layer3': np.random.randn(32, action_size) * 0.1,
            'bias3': np.zeros(action_size)
        }
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, state):
        """Forward pass through network"""
        x = self.relu(np.dot(state, self.weights['layer1']) + self.weights['bias1'])
        x = self.relu(np.dot(x, self.weights['layer2']) + self.weights['bias2'])
        return np.dot(x, self.weights['layer3']) + self.weights['bias3']
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.forward(state)
        return np.argmax(q_values)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        """Train on batch of experiences"""
        if len(self.memory) < batch_size:
            return 0.0
        
        batch = [self.memory[i] for i in np.random.choice(len(self.memory), batch_size, replace=False)]
        total_loss = 0.0
        
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.gamma * np.max(self.forward(next_state))
            
            # Forward pass
            q_values = self.forward(state)
            target_f = q_values.copy()
            target_f[action] = target
            
            # Simplified backprop (gradient descent on output layer only)
            loss = np.mean((q_values - target_f) ** 2)
            total_loss += loss
            
            # Update weights (simplified)
            grad = 2 * (q_values - target_f) * self.learning_rate
            self.weights['layer3'] -= 0.001 * np.outer(state, grad)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return total_loss / batch_size

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = DQNAgent()
    st.session_state.agent_x = 400.0
    st.session_state.agent_y = 300.0
    st.session_state.velocity_x = 0.0
    st.session_state.velocity_y = 0.0
    st.session_state.target_x = 600.0
    st.session_state.target_y = 400.0
    st.session_state.episodes = 0
    st.session_state.total_reward = 0.0
    st.session_state.avg_loss = 0.0
    st.session_state.training_mode = True
    st.session_state.steps = 0
    st.session_state.reward_history = []
    st.session_state.loss_history = []

# Constants
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
MAX_SPEED = 5.0
ACTIONS = {
    0: (0, -1),   # UP
    1: (0, 1),    # DOWN
    2: (-1, 0),   # LEFT
    3: (1, 0)     # RIGHT
}

def get_state():
    """Get current state representation"""
    dx = st.session_state.target_x - st.session_state.agent_x
    dy = st.session_state.target_y - st.session_state.agent_y
    distance = np.sqrt(dx**2 + dy**2)
    
    # Normalize
    dx_norm = np.clip(dx / CANVAS_WIDTH, -1, 1)
    dy_norm = np.clip(dy / CANVAS_HEIGHT, -1, 1)
    vx_norm = st.session_state.velocity_x / MAX_SPEED
    vy_norm = st.session_state.velocity_y / MAX_SPEED
    
    return np.array([dx_norm, dy_norm, vx_norm, vy_norm])

def calculate_reward(old_distance, new_distance):
    """Calculate reward based on progress"""
    progress = old_distance - new_distance
    
    if new_distance < 20:
        return 10.0  # Reached target
    elif progress > 0:
        return progress * 0.5  # Moving closer
    else:
        return progress * 1.0  # Moving away (penalize more)

def step_simulation():
    """Execute one simulation step"""
    old_state = get_state()
    old_x, old_y = st.session_state.agent_x, st.session_state.agent_y
    old_distance = np.sqrt((st.session_state.target_x - old_x)**2 + 
                          (st.session_state.target_y - old_y)**2)
    
    # Choose action
    action = st.session_state.agent.act(old_state)
    dx, dy = ACTIONS[action]
    
    # Update velocity with momentum
    st.session_state.velocity_x = st.session_state.velocity_x * 0.8 + dx * 2.0
    st.session_state.velocity_y = st.session_state.velocity_y * 0.8 + dy * 2.0
    
    # Limit speed
    speed = np.sqrt(st.session_state.velocity_x**2 + st.session_state.velocity_y**2)
    if speed > MAX_SPEED:
        st.session_state.velocity_x = (st.session_state.velocity_x / speed) * MAX_SPEED
        st.session_state.velocity_y = (st.session_state.velocity_y / speed) * MAX_SPEED
    
    # Update position
    st.session_state.agent_x += st.session_state.velocity_x
    st.session_state.agent_y += st.session_state.velocity_y
    
    # Boundary checks
    st.session_state.agent_x = np.clip(st.session_state.agent_x, 0, CANVAS_WIDTH)
    st.session_state.agent_y = np.clip(st.session_state.agent_y, 0, CANVAS_HEIGHT)
    
    # Calculate new state and reward
    new_distance = np.sqrt((st.session_state.target_x - st.session_state.agent_x)**2 + 
                          (st.session_state.target_y - st.session_state.agent_y)**2)
    reward = calculate_reward(old_distance, new_distance)
    done = new_distance < 20
    
    new_state = get_state()
    
    # Store experience and train
    if st.session_state.training_mode:
        st.session_state.agent.remember(old_state, action, reward, new_state, done)
        loss = st.session_state.agent.replay(batch_size=32)
        st.session_state.avg_loss = loss
        st.session_state.loss_history.append(loss)
    
    st.session_state.total_reward += reward
    st.session_state.steps += 1
    
    # Episode end
    if done:
        st.session_state.episodes += 1
        st.session_state.reward_history.append(st.session_state.total_reward)
        # Respawn target
        st.session_state.target_x = np.random.uniform(50, CANVAS_WIDTH - 50)
        st.session_state.target_y = np.random.uniform(50, CANVAS_HEIGHT - 50)
        return True
    
    return False

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🔷 Autonomous AI Agent")
    st.caption("Deep Q-Network with Experience Replay | Neural Architecture: 4→64→32→4")
with col2:
    st.metric("Neural Network", "Active", delta="DQN")

# Main layout
main_col, sidebar_col = st.columns([2, 1])

with main_col:
    st.markdown("### Simulation Environment")
    
    # Canvas with HTML/CSS
    canvas_html = f"""
    <div style="position: relative; width: {CANVAS_WIDTH}px; height: {CANVAS_HEIGHT}px; 
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border: 2px solid rgba(100, 200, 255, 0.4);
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 0 30px rgba(0, 150, 255, 0.3);">
        
        <!-- Grid background -->
        <svg style="position: absolute; width: 100%; height: 100%; opacity: 0.1;">
            <defs>
                <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                    <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(100, 200, 255, 0.3)" stroke-width="1"/>
                </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>
        
        <!-- Target -->
        <div style="position: absolute; 
                    left: {st.session_state.target_x - 15}px; 
                    top: {st.session_state.target_y - 15}px;
                    width: 30px; height: 30px;
                    background: radial-gradient(circle, rgba(255, 100, 100, 0.8) 0%, rgba(255, 50, 50, 0.3) 70%);
                    border: 2px solid rgba(255, 100, 100, 0.9);
                    border-radius: 50%;
                    box-shadow: 0 0 20px rgba(255, 100, 100, 0.6);
                    animation: pulse 2s infinite;">
        </div>
        
        <!-- Agent -->
        <div style="position: absolute; 
                    left: {st.session_state.agent_x - 12}px; 
                    top: {st.session_state.agent_y - 12}px;
                    width: 24px; height: 24px;
                    background: radial-gradient(circle, rgba(0, 255, 150, 0.9) 0%, rgba(0, 200, 255, 0.5) 70%);
                    border: 2px solid rgba(0, 255, 200, 1);
                    border-radius: 50%;
                    box-shadow: 0 0 25px rgba(0, 255, 150, 0.8);
                    animation: glow 1.5s infinite;">
            <!-- Direction indicator -->
            <div style="position: absolute; 
                        left: 50%; top: 50%;
                        transform: translate(-50%, -50%) rotate({np.arctan2(st.session_state.velocity_y, st.session_state.velocity_x) * 180 / np.pi}deg);
                        width: 20px; height: 2px;
                        background: linear-gradient(90deg, rgba(0, 255, 150, 1) 0%, transparent 100%);
                        transform-origin: 0% 50%;">
            </div>
        </div>
        
        <!-- Connection line -->
        <svg style="position: absolute; width: 100%; height: 100%; pointer-events: none;">
            <line x1="{st.session_state.agent_x}" y1="{st.session_state.agent_y}" 
                  x2="{st.session_state.target_x}" y2="{st.session_state.target_y}" 
                  stroke="rgba(100, 150, 255, 0.2)" stroke-width="1" stroke-dasharray="5,5" />
        </svg>
        
        <style>
            @keyframes pulse {{
                0%, 100% {{ transform: scale(1); opacity: 0.8; }}
                50% {{ transform: scale(1.1); opacity: 1; }}
            }}
            @keyframes glow {{
                0%, 100% {{ box-shadow: 0 0 25px rgba(0, 255, 150, 0.8); }}
                50% {{ box-shadow: 0 0 35px rgba(0, 255, 150, 1); }}
            }}
        </style>
    </div>
    """
    st.markdown(canvas_html, unsafe_allow_html=True)
    
    # Controls
    st.markdown("---")
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)
    
    with ctrl_col1:
        if st.button("▶ Run Step", use_container_width=True):
            step_simulation()
            st.rerun()
    
    with ctrl_col2:
        if st.button("⏩ Run 10 Steps", use_container_width=True):
            for _ in range(10):
                if step_simulation():
                    break
            st.rerun()
    
    with ctrl_col3:
        if st.button("🎯 New Target", use_container_width=True):
            st.session_state.target_x = np.random.uniform(50, CANVAS_WIDTH - 50)
            st.session_state.target_y = np.random.uniform(50, CANVAS_HEIGHT - 50)
            st.rerun()
    
    with ctrl_col4:
        if st.button("🔄 Reset Agent", use_container_width=True):
            st.session_state.agent = DQNAgent()
            st.session_state.agent_x = CANVAS_WIDTH / 2
            st.session_state.agent_y = CANVAS_HEIGHT / 2
            st.session_state.episodes = 0
            st.session_state.total_reward = 0.0
            st.session_state.steps = 0
            st.session_state.reward_history = []
            st.session_state.loss_history = []
            st.rerun()

with sidebar_col:
    st.markdown("### Agent Statistics")
    
    # Status
    epsilon = st.session_state.agent.epsilon
    if epsilon > 0.5:
        status = "EXPLORING"
        status_class = "status-learning"
    elif epsilon > 0.1:
        status = "LEARNING"
        status_class = "status-learning"
    else:
        status = "CONVERGED"
        status_class = "status-converged"
    
    st.markdown(f'<p class="{status_class}">Status: {status}</p>', unsafe_allow_html=True)
    
    # Metrics
    st.metric("Episodes Completed", st.session_state.episodes)
    st.metric("Total Steps", st.session_state.steps)
    st.metric("Cumulative Reward", f"{st.session_state.total_reward:.2f}")
    st.metric("Exploration Rate (ε)", f"{epsilon:.4f}")
    st.metric("Average Loss", f"{st.session_state.avg_loss:.6f}")
    
    distance = np.sqrt((st.session_state.target_x - st.session_state.agent_x)**2 + 
                      (st.session_state.target_y - st.session_state.agent_y)**2)
    st.metric("Distance to Target", f"{distance:.1f}px")
    
    speed = np.sqrt(st.session_state.velocity_x**2 + st.session_state.velocity_y**2)
    st.metric("Current Speed", f"{speed:.2f} px/step")
    
    st.markdown("---")
    
    # Training toggle
    st.session_state.training_mode = st.checkbox("Training Mode", value=st.session_state.training_mode)
    
    # Network info
    with st.expander("🧠 Neural Network Info"):
        st.markdown(f"""
        **Architecture:**
        - Input Layer: 4 neurons (dx, dy, vx, vy)
        - Hidden Layer 1: 64 neurons (ReLU)
        - Hidden Layer 2: 32 neurons (ReLU)
        - Output Layer: 4 neurons (Q-values)
        
        **Parameters:**
        - Total weights: {sum(w.size for w in st.session_state.agent.weights.values())}
        - Learning rate: {st.session_state.agent.learning_rate}
        - Discount factor (γ): {st.session_state.agent.gamma}
        - Memory size: {len(st.session_state.agent.memory)}/2000
        """)
    
    # Performance charts
    if len(st.session_state.reward_history) > 0:
        with st.expander("📊 Performance Metrics"):
            st.line_chart(st.session_state.reward_history[-50:], use_container_width=True)
            st.caption("Episode Rewards (Last 50)")
            
            if len(st.session_state.loss_history) > 10:
                st.line_chart(st.session_state.loss_history[-100:], use_container_width=True)
                st.caption("Training Loss (Last 100)")

# Footer info
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9em;'>
    <p>Algorithm: Deep Q-Network (DQN) | State Space: Continuous | Action Space: Discrete (4 actions)</p>
    <p>Future Migration: Compatible with ROS2, Raspberry Pi, Arduino robotics platforms</p>
</div>
""", unsafe_allow_html=True)
