import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from collections import deque

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
    h1, h2, h3, p, label, .stMetric {
        color: #e0e0e0 !important;
    }
    .stMetric label {
        color: #888 !important;
    }
    .status-learning {
        color: #ffaa00;
        font-weight: bold;
        font-size: 1.2em;
    }
    .status-converged {
        color: #00ff88;
        font-weight: bold;
        font-size: 1.2em;
    }
    div[data-testid="stMetricValue"] {
        color: #00ddff !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Neural Network (Deep Q-Network)
class DQNAgent:
    def __init__(self, state_size=4, action_size=4, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        
        # Neural network weights
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
        layer1_output = self.relu(np.dot(state, self.weights['layer1']) + self.weights['bias1'])
        layer2_output = self.relu(np.dot(layer1_output, self.weights['layer2']) + self.weights['bias2'])
        q_values = np.dot(layer2_output, self.weights['layer3']) + self.weights['bias3']
        return q_values, layer1_output, layer2_output
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values, _, _ = self.forward(state)
        return np.argmax(q_values)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return 0.0
        
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        total_loss = 0.0
        
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_q_values, _, _ = self.forward(next_state) # Unpack the values
                target += self.gamma * np.max(next_q_values)
            
            q_values, layer1_output, layer2_output = self.forward(state) # Unpack the values
            target_f = q_values.copy()
            target_f[action] = target
            
            loss = np.mean((q_values - target_f) ** 2)
            total_loss += loss
            
            # Backpropagation for the output layer (layer3)
            grad = 2 * (q_values - target_f) * self.learning_rate
            self.weights['layer3'] -= np.outer(layer2_output, grad) # Use layer2_output here
            self.weights['bias3'] -= grad
        
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
    st.session_state.trajectory_x = [400.0]
    st.session_state.trajectory_y = [300.0]

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
    dx = st.session_state.target_x - st.session_state.agent_x
    dy = st.session_state.target_y - st.session_state.agent_y
    
    dx_norm = np.clip(dx / CANVAS_WIDTH, -1, 1)
    dy_norm = np.clip(dy / CANVAS_HEIGHT, -1, 1)
    vx_norm = st.session_state.velocity_x / MAX_SPEED
    vy_norm = st.session_state.velocity_y / MAX_SPEED
    
    return np.array([dx_norm, dy_norm, vx_norm, vy_norm])

def calculate_reward(old_distance, new_distance):
    progress = old_distance - new_distance
    
    if new_distance < 20:
        return 10.0
    elif progress > 0:
        return progress * 0.5
    else:
        return progress * 1.0

def step_simulation():
    old_state = get_state()
    old_x, old_y = st.session_state.agent_x, st.session_state.agent_y
    old_distance = np.sqrt((st.session_state.target_x - old_x)**2 + 
                          (st.session_state.target_y - old_y)**2)
    
    action = st.session_state.agent.act(old_state)
    dx, dy = ACTIONS[action]
    
    st.session_state.velocity_x = st.session_state.velocity_x * 0.8 + dx * 2.0
    st.session_state.velocity_y = st.session_state.velocity_y * 0.8 + dy * 2.0
    
    speed = np.sqrt(st.session_state.velocity_x**2 + st.session_state.velocity_y**2)
    if speed > MAX_SPEED:
        st.session_state.velocity_x = (st.session_state.velocity_x / speed) * MAX_SPEED
        st.session_state.velocity_y = (st.session_state.velocity_y / speed) * MAX_SPEED
    
    st.session_state.agent_x += st.session_state.velocity_x
    st.session_state.agent_y += st.session_state.velocity_y
    
    st.session_state.agent_x = np.clip(st.session_state.agent_x, 0, CANVAS_WIDTH)
    st.session_state.agent_y = np.clip(st.session_state.agent_y, 0, CANVAS_HEIGHT)
    
    # Track trajectory
    st.session_state.trajectory_x.append(st.session_state.agent_x)
    st.session_state.trajectory_y.append(st.session_state.agent_y)
    if len(st.session_state.trajectory_x) > 200:
        st.session_state.trajectory_x.pop(0)
        st.session_state.trajectory_y.pop(0)
    
    new_distance = np.sqrt((st.session_state.target_x - st.session_state.agent_x)**2 + 
                          (st.session_state.target_y - st.session_state.agent_y)**2)
    reward = calculate_reward(old_distance, new_distance)
    done = new_distance < 20
    
    new_state = get_state()
    
    if st.session_state.training_mode:
        st.session_state.agent.remember(old_state, action, reward, new_state, done)
        loss = st.session_state.agent.replay(batch_size=32)
        st.session_state.avg_loss = loss
        st.session_state.loss_history.append(loss)
    
    st.session_state.total_reward += reward
    st.session_state.steps += 1
    
    if done:
        st.session_state.episodes += 1
        st.session_state.reward_history.append(st.session_state.total_reward)
        st.session_state.target_x = np.random.uniform(50, CANVAS_WIDTH - 50)
        st.session_state.target_y = np.random.uniform(50, CANVAS_HEIGHT - 50)
        st.session_state.trajectory_x = [st.session_state.agent_x]
        st.session_state.trajectory_y = [st.session_state.agent_y]
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
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add grid background
    for i in range(0, CANVAS_WIDTH, 40):
        fig.add_shape(type="line", x0=i, y0=0, x1=i, y1=CANVAS_HEIGHT,
                     line=dict(color="rgba(100, 200, 255, 0.1)", width=1))
    for i in range(0, CANVAS_HEIGHT, 40):
        fig.add_shape(type="line", x0=0, y0=i, x1=CANVAS_WIDTH, y1=i,
                     line=dict(color="rgba(100, 200, 255, 0.1)", width=1))
    
    # Add trajectory
    if len(st.session_state.trajectory_x) > 1:
        fig.add_trace(go.Scatter(
            x=st.session_state.trajectory_x,
            y=st.session_state.trajectory_y,
            mode='lines',
            line=dict(color='rgba(0, 255, 150, 0.3)', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add connection line
    fig.add_trace(go.Scatter(
        x=[st.session_state.agent_x, st.session_state.target_x],
        y=[st.session_state.agent_y, st.session_state.target_y],
        mode='lines',
        line=dict(color='rgba(100, 150, 255, 0.3)', width=1, dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add target
    fig.add_trace(go.Scatter(
        x=[st.session_state.target_x],
        y=[st.session_state.target_y],
        mode='markers',
        marker=dict(
            size=30,
            color='rgba(255, 100, 100, 0.8)',
            line=dict(color='rgba(255, 100, 100, 1)', width=2)
        ),
        name='Target',
        hovertemplate='Target<br>X: %{x:.0f}<br>Y: %{y:.0f}<extra></extra>'
    ))
    
    # Add agent
    fig.add_trace(go.Scatter(
        x=[st.session_state.agent_x],
        y=[st.session_state.agent_y],
        mode='markers',
        marker=dict(
            size=25,
            color='rgba(0, 255, 150, 0.9)',
            line=dict(color='rgba(0, 255, 200, 1)', width=2)
        ),
        name='AI Agent',
        hovertemplate='Agent<br>X: %{x:.0f}<br>Y: %{y:.0f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        width=CANVAS_WIDTH,
        height=CANVAS_HEIGHT,
        plot_bgcolor='rgba(26, 26, 46, 0.9)',
        paper_bgcolor='rgba(15, 15, 30, 0.8)',
        xaxis=dict(range=[0, CANVAS_WIDTH], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, CANVAS_HEIGHT], showgrid=False, zeroline=False, visible=False),
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)', font=dict(color='white')),
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=False)
    
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
        if st.button("🔄 Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

with sidebar_col:
    st.markdown("### Agent Statistics")
    
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
    
    st.metric("Episodes", st.session_state.episodes)
    st.metric("Steps", st.session_state.steps)
    st.metric("Reward", f"{st.session_state.total_reward:.2f}")
    st.metric("Exploration (ε)", f"{epsilon:.4f}")
    st.metric("Loss", f"{st.session_state.avg_loss:.6f}")
    
    distance = np.sqrt((st.session_state.target_x - st.session_state.agent_x)**2 + 
                      (st.session_state.target_y - st.session_state.agent_y)**2)
    st.metric("Distance", f"{distance:.1f}px")
    
    speed = np.sqrt(st.session_state.velocity_x**2 + st.session_state.velocity_y**2)
    st.metric("Speed", f"{speed:.2f}")
    
    st.markdown("---")
    st.session_state.training_mode = st.checkbox("Training Mode", value=st.session_state.training_mode)
    
    with st.expander("🧠 Network Details"):
        st.markdown(f"""
        **Architecture:**
        - Input: 4 neurons
        - Hidden 1: 64 neurons (ReLU)
        - Hidden 2: 32 neurons (ReLU)
        - Output: 4 neurons (Q-values)
        
        **Hyperparameters:**
        - Learning rate: {st.session_state.agent.learning_rate}
        - Discount (γ): {st.session_state.agent.gamma}
        - Memory: {len(st.session_state.agent.memory)}/2000
        - Batch size: 32
        """)
    
    if len(st.session_state.reward_history) > 0:
        with st.expander("📊 Performance"):
            st.line_chart(st.session_state.reward_history[-50:])
            st.caption("Episode Rewards")
            
            if len(st.session_state.loss_history) > 10:
                st.line_chart(st.session_state.loss_history[-100:])
                st.caption("Training Loss")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85em;'>
    Deep Q-Network | Experience Replay | Continuous State Space | Compatible with ROS2/Robotics
</div>
""", unsafe_allow_html=True)
