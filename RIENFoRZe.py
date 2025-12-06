import streamlit as st
import numpy as np
import random
from collections import deque
import time
import re # For parsing user commands

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

class AdvancedMind:
    def __init__(self, state_size=5, action_size=4, buffer_size=10000):
        # State: [AgentX, AgentY, TargetX, TargetY, Energy]
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(buffer_size) # UPGRADED MEMORY
        
        # Hyperparameters (Adaptive)
        self.gamma = 0.95    
        self.epsilon = 1.0   
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.learning_rate = 0.005 # Adjusted for more stable learning
        self.beta = 0.4 # Importance sampling exponent
        self.beta_increment = 0.001
        
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
        # With PER, we just add the experience. Priority is updated after learning.
        self.memory.add(state, action, reward, next_state, done)

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size: return 0, 0
        
        # Sample from the prioritized buffer
        batch, indices, weights = self.memory.sample(batch_size, self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment) 
        
        loss_val = 0
        new_priorities = []
        
        # Accumulate gradients (Simulating a batch update)
        # We process item by item for clarity in this numpy implementation
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
            
            # 3. Calculate Error (TD Error)
            # We only care about the error for the action we actually took
            td_error = target - current_q[0][action]
            
            # Store priority
            new_priorities.append(abs(td_error))
            
            # 4. BACKPROPAGATION (The Fix)
            # Apply importance sampling weight
            weighted_error = td_error * weights[i]
            loss_val += weighted_error ** 2
            
            # -- Gradient for Output Layers --
            # dQ/dVal = 1, dQ/dAdv = 1 (at action index)
            # We treat weighted_error as the upstream gradient
            
            grad_val = weighted_error * a1.T # (64, 1)
            
            grad_adv = np.zeros_like(self.online_net['W_adv'])
            grad_adv[:, action] = weighted_error * a1[0] # Only update the specific action taken
            
            # -- Gradient for Hidden Layer (W1) --
            # Backprop error through Val stream and Adv stream
            # Error at Hidden Layer = (Error * W_val) + (Error * W_adv)
            error_from_val = np.dot(self.online_net['W_val'], weighted_error) 
            error_from_adv = np.dot(self.online_net['W_adv'][:, action].reshape(-1, 1), weighted_error)
            
            total_error_at_hidden = (error_from_val + error_from_adv).T # (1, 64)
            
            # Apply ReLU derivative (if a1 <= 0, grad is 0)
            total_error_at_hidden[a1 <= 0] = 0
            
            grad_w1 = np.dot(state.T, total_error_at_hidden)
            
            # -- Update Weights --
            # Using a simplified SGD update rule
            self.online_net['W_val'] += self.learning_rate * grad_val
            self.online_net['W_adv'] += self.learning_rate * grad_adv
            self.online_net['W1']    += self.learning_rate * grad_w1

        self.memory.update_priorities(indices, new_priorities)
        
        # Decay exploration
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

def reset_simulation():
    """Clears the session state to reset the simulation."""
    keys_to_clear = [
        'mind', 'soul', 'agent_pos', 'target_pos', 
        'step_count', 'wins', 'auto_mode', 'chat_history'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

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
    move_speed = 8.0 # Boost speed slightly to make training faster
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
    loss, td_error = st.session_state.mind.replay(batch_size=32) # Increased batch size
    
    # 6. Update Soul
    st.session_state.soul.update(reward, td_error, st.session_state.wins)
    
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
m2.metric("Energy", f"{st.session_state.soul.energy:.1f}%", f"{st.session_state.wins} Wins")
m3.metric("IQ (Loss)", f"{st.session_state.mind.epsilon:.3f}")
m4.metric("Experience", st.session_state.step_count)

# Main Interaction Area
row1_1, row1_2 = st.columns([2, 1])

with row1_1:
    # -----------------------------------
    # THE "WORLD" (ASCII Visualization)
    # -----------------------------------
    st.markdown("### 🌍 Containment Field")
    grid_height = 15
    grid_width = 40
    
    # Create an empty grid
    grid = [['.' for _ in range(grid_width)] for _ in range(grid_height)]
    
    # Scale positions to fit the grid
    agent_y = int(st.session_state.agent_pos[1] / 100 * (grid_height - 1))
    agent_x = int(st.session_state.agent_pos[0] / 100 * (grid_width - 1))
    target_y = int(st.session_state.target_pos[1] / 100 * (grid_height - 1))
    target_x = int(st.session_state.target_pos[0] / 100 * (grid_width - 1))

    # Place agent and target
    # Ensure they don't overwrite each other for clarity
    if (agent_y, agent_x) == (target_y, target_x):
        grid[agent_y][agent_x] = '💥'
    else:
        grid[agent_y][agent_x] = st.session_state.soul.moods[st.session_state.soul.current_mood]
        grid[target_y][target_x] = '💎'

    # Convert grid to a single string and display
    grid_str = "\n".join(" ".join(row) for row in grid)
    st.code(grid_str, language=None)
    
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
        
        # NLP: Keyword and pattern matching
        user_input_lower = user_input.lower()
        
        # Pattern to find coordinates like "at 20, 80" or "to 20 80"
        coord_match = re.search(r'(\d+)\s*,\s*(\d+)', user_input_lower)

        if "good" in user_input.lower():
            st.session_state.soul.current_mood = "Happy"
            st.session_state.soul.energy += 10
            st.session_state.soul.last_chat = "Thank you! Your feedback is a positive reward."
            st.toast("AI felt your praise! ❤️")
        elif "how" in user_input_lower and "reach" in user_input_lower and coord_match:
            x, y = map(int, coord_match.groups())
            st.session_state.soul.last_chat = f"Calculating path to ({x}, {y})..."
            path = plan_path_to_target(st.session_state.agent_pos, (x,y))
            if path:
                # Translate path into directions
                directions = []
                for i in range(len(path) - 1):
                    y1, x1 = path[i]
                    y2, x2 = path[i+1]
                    if y2 > y1: directions.append("Down")
                    elif y2 < y1: directions.append("Up")
                    elif x2 > x1: directions.append("Right")
                    elif x2 < x1: directions.append("Left")
                st.session_state.soul.last_chat = f"Path to ({x},{y}) found! Plan: {', '.join(directions[:4])}..."
            else:
                st.session_state.soul.last_chat = f"I cannot find a path to ({x},{y}) from here."
        else:
            st.session_state.soul.last_chat = random.choice([
                "I do not understand that command.", 
                "My language model is still developing.",
                "Could you rephrase that, Prince?"
            ])

    # -----------------------------------
    # AUTOMATION CONTROL
    # -----------------------------------
    st.markdown("---")
    col_a, col_b, col_c = st.columns([2,2,3])
    
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
    
    if col_c.button("🔄 Reset Simulation"):
        reset_simulation()

# Debug / Mind Palace
with st.expander("🧠 Open Mind Palace (Neural Weights)"):
    st.write("First Layer Weights (Visual Cortex):")
    st.bar_chart(st.session_state.mind.online_net['W1'][:10])
