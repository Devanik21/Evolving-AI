import streamlit as st
import numpy as np
import plotly.graph_objects as go
import random
from collections import deque
import time
import math

# ==========================================
# 1. ADVANCED CONFIGURATION & CSS
# ==========================================
st.set_page_config(
    page_title="Project A.L.I.V.E.",
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
class AdvancedMind:
    def __init__(self, state_size=5, action_size=4):
        # State: [AgentX, AgentY, TargetX, TargetY, Energy]
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        
        # Hyperparameters (Adaptive)
        self.gamma = 0.95    
        self.epsilon = 1.0   
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.learning_rate = 0.005
        
        # Dual Networks (Online + Target for stability)
        self.online_net = self.init_network()
        self.target_net = self.init_network()
        self.update_target_network()

    def init_network(self):
        # Architecture: Dueling DQN (Value Stream + Advantage Stream)
        # We simulate this complexity with numpy matrices
        return {
            'W1': np.random.randn(self.state_size, 64) / np.sqrt(self.state_size),
            'b1': np.zeros((1, 64)),
            'W_val': np.random.randn(64, 1) / np.sqrt(64),     # State Value V(s)
            'b_val': np.zeros((1, 1)),
            'W_adv': np.random.randn(64, self.action_size) / np.sqrt(64), # Advantage A(s,a)
            'b_adv': np.zeros((1, self.action_size))
        }

    def update_target_network(self):
        # Soft copy weights from Online to Target
        self.target_net = {k: v.copy() for k, v in self.online_net.items()}

    def relu(self, z):
        return np.maximum(0, z)

    def forward(self, state, network):
        if state.ndim == 1: state = state.reshape(1, -1)
        
        # Shared Layer
        z1 = np.dot(state, network['W1']) + network['b1']
        a1 = self.relu(z1)
        
        # Dueling Streams
        val = np.dot(a1, network['W_val']) + network['b_val'] # Scalar value of state
        adv = np.dot(a1, network['W_adv']) + network['b_adv'] # Advantage of each action
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = val + (adv - np.mean(adv, axis=1, keepdims=True))
        return q_values, a1

    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values, _ = self.forward(state, self.online_net)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        # Intrinsic Curiosity: High error = Surprise = Good to remember
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size: return 0
        
        batch = random.sample(self.memory, batch_size)
        loss_val = 0
        
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                # Double DQN Logic: Select action with Online, Evaluate with Target
                next_q_online, _ = self.forward(next_state, self.online_net)
                best_next_action = np.argmax(next_q_online[0])
                
                next_q_target, _ = self.forward(next_state, self.target_net)
                target = reward + self.gamma * next_q_target[0][best_next_action]
            
            # Forward pass
            current_q, a1 = self.forward(state, self.online_net)
            
            # Error Calculation
            target_f = current_q.copy()
            error = target - target_f[0][action]
            target_f[0][action] = target
            
            # Simple Backprop (Stochastic Gradient Descent)
            # This is a manual approximation of backprop for the Dueling architecture
            loss_val += error ** 2
            
            # Update weights (simplified for demo speed)
            # In a real heavy library like PyTorch, this is autograd
            grad = (target_f - current_q)
            self.online_net['W1'] += self.learning_rate * np.dot(state.reshape(1,-1).T, np.dot(grad, self.online_net['W_adv'].T) * (a1>0)) 

        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss_val / batch_size

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

    def update(self, reward, loss, recent_wins):
        if self.energy < 20:
            self.current_mood = "Sleeping"
        elif reward > 5:
            self.current_mood = "Excited"
        elif reward > 0:
            self.current_mood = "Happy"
        elif loss > 50:
            self.current_mood = "Confused"
        elif reward < 0:
            self.current_mood = "Sad"
        else:
            self.current_mood = "Curious"
            
        # Dynamic Dialogue Generation
        if self.current_mood == "Excited":
            self.last_chat = random.choice(["I learned something new!", "That was tasty!", "My neurons are firing!"])
        elif self.current_mood == "Confused":
            self.last_chat = random.choice(["This data is noisy...", "Adjusting weights...", "I'm trying to understand."])
        elif self.current_mood == "Sad":
            self.last_chat = random.choice(["Ouch.", "Negative reward detected.", "I'll do better next time."])

# ==========================================
# 4. APP STATE INITIALIZATION
# ==========================================
if 'mind' not in st.session_state:
    st.session_state.mind = AdvancedMind()
    st.session_state.soul = PersonalityCore()
    st.session_state.agent_pos = np.array([50.0, 50.0])
    st.session_state.target_pos = np.array([80.0, 20.0])
    st.session_state.step_count = 0
    st.session_state.wins = 0
    st.session_state.auto_mode = False
    st.session_state.chat_history = []

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
    move_speed = 6.0 # Faster for advanced feel
    old_pos = st.session_state.agent_pos.copy()
    
    # Smooth movement (Interpolation)
    if action == 0: st.session_state.agent_pos[1] += move_speed # Up
    elif action == 1: st.session_state.agent_pos[1] -= move_speed # Down
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
    reward = (dist_before - dist_after) * 2.0 
    
    # Event: Reached Target
    if dist_after < 8:
        reward = 50 # Big dopamine hit
        done = True
        st.session_state.wins += 1
        st.session_state.soul.energy = min(100, st.session_state.soul.energy + 20)
        # Move target randomly
        st.session_state.target_pos = np.random.randint(10, 90, size=2)
    else:
        st.session_state.soul.energy -= 0.1 # Metabolism
        
    # 5. Learn (Plasticity)
    next_state = np.array([
        st.session_state.agent_pos[0]/100, 
        st.session_state.agent_pos[1]/100, 
        st.session_state.target_pos[0]/100, 
        st.session_state.target_pos[1]/100,
        st.session_state.soul.energy/100
    ])
    
    st.session_state.mind.remember(state, action, reward, next_state, done)
    loss = st.session_state.mind.replay(batch_size=16)
    
    # 6. Update Soul
    st.session_state.soul.update(reward, loss, st.session_state.wins)
    
    # Update Target Network periodically
    if st.session_state.step_count % 50 == 0:
        st.session_state.mind.update_target_network()

    st.session_state.step_count += 1

# ==========================================
# 5. UI LAYOUT
# ==========================================
st.title("🧬 Project A.L.I.V.E.")
st.caption("Autonomous Learning Intelligent Virtual Entity")

# Top Bar: Stats
m1, m2, m3, m4 = st.columns(4)
m1.metric("Status", st.session_state.soul.current_mood)
m2.metric("Energy", f"{st.session_state.soul.energy:.1f}%")
m3.metric("IQ (Loss)", f"{st.session_state.mind.epsilon:.3f}")
m4.metric("Experience", st.session_state.step_count)

# Main Interaction Area
row1_1, row1_2 = st.columns([2, 1])

with row1_1:
    # -----------------------------------
    # THE "WORLD" (Plotly Visualization)
    # -----------------------------------
    # We use a trick to make the marker change size based on "breathing"
    breath = math.sin(time.time() * 5) * 2
    
    fig = go.Figure()
    
    # The Agent (Complex Symbol)
    fig.add_trace(go.Scatter(
        x=[st.session_state.agent_pos[0]], 
        y=[st.session_state.agent_pos[1]],
        mode='markers+text',
        marker=dict(
            size=25 + breath, 
            color='#00ddff', 
            symbol='circle-dot',
            line=dict(color='white', width=2)
        ),
        text=[st.session_state.soul.moods[st.session_state.soul.current_mood]],
        textposition="top center",
        textfont=dict(size=14, color='white'),
        name='ALIVE'
    ))
    
    # The Target (Lure)
    fig.add_trace(go.Scatter(
        x=[st.session_state.target_pos[0]], 
        y=[st.session_state.target_pos[1]],
        mode='markers',
        marker=dict(
            size=15, 
            color='#ff0055', 
            symbol='diamond',
            line=dict(color='white', width=1)
        ),
        name='Attractor'
    ))

    fig.update_layout(
        xaxis=dict(range=[0, 100], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, 100], showgrid=False, zeroline=False, visible=False, scaleanchor="x"),
        plot_bgcolor='rgba(15, 15, 30, 0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        height=450,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Manual Override (The "Lure")
    st.markdown("### 🧲 Focus Attention (Lure)")
    cx, cy = st.columns(2)
    # Using sliders as "Hover" replacement for Python backend
    tx = cx.slider("Horizontal Focus", 0, 100, int(st.session_state.target_pos[0]), key='tx')
    ty = cy.slider("Vertical Focus", 0, 100, int(st.session_state.target_pos[1]), key='ty')
    
    # Update target from user input
    if tx != int(st.session_state.target_pos[0]) or ty != int(st.session_state.target_pos[1]):
        st.session_state.target_pos = np.array([float(tx), float(ty)])
        st.rerun() # Immediate update

with row1_2:
    # -----------------------------------
    # THE "MIND" (Communication)
    # -----------------------------------
    st.markdown("### 💬 Neural Link")
    
    # AI Voice
    st.markdown(f"""
    <div class="ai-bubble">
        <b>🤖 ALIVE:</b> {st.session_state.soul.last_chat}
    </div>
    """, unsafe_allow_html=True)
    
    # User Voice
    user_input = st.text_input("Speak to AI:", placeholder="Say 'Good job' or 'Come here'...")
    if user_input:
        st.markdown(f"""
        <div class="user-bubble">
            <b>You:</b> {user_input}
        </div>
        """, unsafe_allow_html=True)
        # NLP Placeholder: Simple keyword reaction
        if "good" in user_input.lower():
            st.session_state.soul.current_mood = "Happy"
            st.session_state.soul.energy += 10
            st.toast("AI felt your praise! ❤️")
        elif "come" in user_input.lower():
            # Teleport target closer to agent
            st.session_state.target_pos = st.session_state.agent_pos + np.random.randint(-10, 10, 2)
            st.toast("AI is looking for you! 👀")

    # -----------------------------------
    # AUTOMATION CONTROL
    # -----------------------------------
    st.markdown("---")
    col_a, col_b = st.columns(2)
    
    # Auto-Run Toggle
    auto = col_a.checkbox("Run Autonomously", value=st.session_state.auto_mode)
    if auto:
        st.session_state.auto_mode = True
        time.sleep(0.1) # Game Loop Speed
        process_step()
        st.rerun()
    else:
        st.session_state.auto_mode = False
        if col_b.button("Step Once"):
            process_step()
            st.rerun()

# Debug / Mind Palace
with st.expander("🧠 Open Mind Palace (Neural Weights)"):
    st.write("First Layer Weights (Visual Cortex):")
    st.bar_chart(st.session_state.mind.online_net['W1'][:10])
