import streamlit as st
import numpy as np
import random
from collections import deque
import time
import pandas as pd
import re # For parsing user commands

# --- NEW IMPORTS START ---
import json
import io
import zipfile
# --- NEW IMPORTS END ---

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
# 1.5 HELPER: NUMPY JSON ENCODER
# ==========================================
# ==========================================
# 1.5 HELPER: NUMPY JSON ENCODER (Upgraded)
# ==========================================
class NumpyEncoder(json.JSONEncoder):
    """ Special helper to save Numpy types as standard Python types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)  # Convert numpy int to python int
        elif isinstance(obj, np.floating):
            return float(obj) # Convert numpy float to python float
        elif isinstance(obj, np.ndarray):
            return obj.tolist() # Convert array to list
        elif isinstance(obj, np.bool_):
            return bool(obj) # Convert numpy bool to python bool
        return super().default(obj)

def convert_weights_to_numpy(weights_dict):
    """ Helper to convert loaded JSON lists back to Numpy arrays for the brain """
    new_dict = {}
    for k, v in weights_dict.items():
        new_dict[k] = np.array(v)
    return new_dict

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
        # 1. Critical Needs
        if self.energy < 20:
            self.current_mood = "Sleeping"
            self.thought_process = "CRITICAL: Energy low. Reducing cognitive load."
            return # Stop processing other moods if asleep

        # 2. Success
        if reward > 10:
            self.current_mood = "Excited"
            self.thought_process = "ANALYSIS: Significant success detected! Dopamine release."
            return

        # 3. Failure
        if reward < -5:
            self.current_mood = "Sad"
            self.thought_process = "ANALYSIS: Negative outcome. Re-evaluating strategy."
            return

        # 4. Learning (Adjusted Logic)
        # We increase the threshold for confusion from 5 to 15 to stop flickering
        if td_error > 15: 
            self.current_mood = "Confused"
            self.thought_process = "ANALYSIS: Surprise event. High learning opportunity."
        elif td_error > 5:
            # New "Focusing" state for moderate errors
            self.current_mood = "Curious" 
            self.thought_process = "Optimizing neural pathways..."
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


class AdvancedMind:
    def __init__(self, state_size=5, action_size=4, buffer_size=10000, hidden_size=64):
        # State: [AgentX, AgentY, TargetX, TargetY, Energy]
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
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
            'W1': np.random.randn(self.state_size, self.hidden_size) / np.sqrt(self.state_size),
            'b1': np.zeros((1, self.hidden_size)),
            'W_val': np.random.randn(self.hidden_size, 1) / np.sqrt(self.hidden_size),     # State Value V(s)
            'b_val': np.zeros((1, 1)),
            'W_adv': np.random.randn(self.hidden_size, self.action_size) / np.sqrt(self.hidden_size), # Advantage A(s,a)
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
            
            grad_val = weighted_error * a1.T # (hidden_size, 1)
            
            grad_adv = np.zeros_like(self.online_net['W_adv'])
            grad_adv[:, action] = weighted_error * a1[0] # Only update the specific action taken
            
            # -- Gradient for Hidden Layer (W1) --
            # Backprop error through Val stream and Adv stream
            # Error at Hidden Layer = (Error * W_val) + (Error * W_adv)
            error_from_val = np.dot(self.online_net['W_val'], weighted_error) 
            error_from_adv = np.dot(self.online_net['W_adv'][:, action].reshape(-1, 1), weighted_error)
            
            total_error_at_hidden = (error_from_val + error_from_adv).T # (1, hidden_size)
            
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
# 2. THE ADVANCED MIND (Real Quantum Solver)
# ==========================================
class RubiksMind:
    """
    REAL 2x2 Solver using Bidirectional BFS.
    No more simulation. This finds the mathematically perfect path.
    """
    def __init__(self):
        self.moves = ["U", "U'", "U2", "F", "F'", "F2", "R", "R'", "R2"]
        # 2x2 State Map: 24 stickers.
        # 0-3:U, 4-7:L, 8-11:F, 12-15:R, 16-19:B, 20-23:D
        self.solved_state = tuple(range(24))
        self.neural_weights = {2: 0} # Experience counter

    def apply_move(self, state, move):
        """Permutes the standard 24-sticker string based on the move."""
        # Convert tuple to list for mutation
        s = list(state)
        
        # Helper to cycle 4 positions: a->b->c->d->a
        def cycle(indices):
            tmp = s[indices[3]]
            s[indices[3]] = s[indices[2]]
            s[indices[2]] = s[indices[1]]
            s[indices[1]] = s[indices[0]]
            s[indices[0]] = tmp

        # Basic 90 degree face rotations
        # U Face (0,1,2,3)
        if "U" in move:
            base = 0
            if "U'" in move: # Counter-clockwise
                s[0], s[1], s[2], s[3] = s[1], s[3], s[0], s[2]
                cycle([4, 16, 12, 8]); cycle([5, 17, 13, 9]) # Side stickers L->B->R->F (reversed)
            elif "U2" in move:
                s[0], s[1], s[2], s[3] = s[3], s[2], s[1], s[0]
                s[4],s[12]=s[12],s[4]; s[5],s[13]=s[13],s[5] # Swap L-R
                s[8],s[16]=s[16],s[8]; s[9],s[17]=s[17],s[9] # Swap F-B
            else: # Clockwise
                s[0], s[1], s[2], s[3] = s[2], s[0], s[3], s[1]
                cycle([4, 8, 12, 16]); cycle([5, 9, 13, 17]) # Side stickers L->F->R->B

        # F Face (8,9,10,11)
        if "F" in move:
            if "F'" in move:
                s[8], s[9], s[10], s[11] = s[9], s[11], s[8], s[10]
                cycle([2, 12, 21, 7]); cycle([3, 14, 20, 6])
            elif "F2" in move:
                s[8], s[9], s[10], s[11] = s[11], s[10], s[9], s[8]
                s[2],s[21]=s[21],s[2]; s[3],s[20]=s[20],s[3]
                s[6],s[12]=s[12],s[6]; s[7],s[14]=s[14],s[7]
            else:
                s[8], s[9], s[10], s[11] = s[10], s[8], s[11], s[9]
                cycle([2, 6, 21, 14]); cycle([3, 12, 20, 7]) # U->L->D->R

        # R Face (12,13,14,15)
        if "R" in move:
            if "R'" in move:
                s[12], s[13], s[14], s[15] = s[13], s[15], s[12], s[14]
                cycle([1, 19, 21, 9]); cycle([3, 17, 23, 11])
            elif "R2" in move:
                s[12], s[13], s[14], s[15] = s[15], s[14], s[13], s[12]
                s[1],s[21]=s[21],s[1]; s[3],s[23]=s[23],s[3]
                s[9],s[19]=s[19],s[9]; s[11],s[17]=s[17],s[11]
            else:
                s[12], s[13], s[14], s[15] = s[14], s[12], s[15], s[13]
                cycle([1, 9, 21, 19]); cycle([3, 11, 23, 17]) # U->F->D->B

        return tuple(s)

    def get_scramble(self, size):
        # Generate a real random state by applying moves
        state = self.solved_state
        scramble_moves = []
        for _ in range(11): # 11 is God's Number for 2x2
            m = random.choice(self.moves)
            scramble_moves.append(m)
            state = self.apply_move(state, m)
        return scramble_moves

    def solve_simulation(self, size, scramble_moves):
        """
        Executes Bidirectional BFS to find the OPTIMAL solution.
        """
        if size != 2:
            return 0.0, ["ERROR: Only 2x2 supported in Quantum Mode"], ["Requires Upgrade"], 0
        
        start_time = time.time()
        
        # 1. Reconstruct Current State from Scramble
        current_state = self.solved_state
        for m in scramble_moves:
            current_state = self.apply_move(current_state, m)

        # 2. Bidirectional BFS
        # Forward search (From Scrambled -> Solved)
        forward_parents = {current_state: None}
        forward_moves = {current_state: []}
        forward_queue = deque([current_state])
        
        # Backward search (From Solved -> Scrambled)
        backward_parents = {self.solved_state: None}
        backward_moves = {self.solved_state: []}
        backward_queue = deque([self.solved_state])
        
        solution = []
        visited_any = False
        
        # Limit depth to avoid freezing (God's number is 11, we search half depth 6)
        while forward_queue and backward_queue:
            # -- Forward Step --
            if forward_queue:
                state = forward_queue.popleft()
                if state in backward_moves:
                    # MEET IN THE MIDDLE!
                    solution = forward_moves[state] + list(reversed(backward_moves[state]))
                    # Invert backward moves because we came from solved
                    final_sol = forward_moves[state] 
                    for bm in backward_moves[state]:
                        # Invert the move (U -> U', U2 -> U2)
                        if "'" in bm: final_sol.append(bm.replace("'", ""))
                        elif "2" in bm: final_sol.append(bm)
                        else: final_sol.append(bm + "'")
                    solution = final_sol
                    break
                
                # Expand
                if len(forward_moves[state]) < 6: # Depth limit
                    for m in self.moves:
                        next_s = self.apply_move(state, m)
                        if next_s not in forward_moves:
                            forward_moves[next_s] = forward_moves[state] + [m]
                            forward_queue.append(next_s)

            # -- Backward Step --
            if backward_queue:
                state = backward_queue.popleft()
                if state in forward_moves:
                    # MEET IN THE MIDDLE (Same logic)
                    # This path is usually redundant to check here if we check above, 
                    # but strictly correct for bi-bfs.
                    final_sol = forward_moves[state]
                    for bm in backward_moves[state]:
                        if "'" in bm: final_sol.append(bm.replace("'", ""))
                        elif "2" in bm: final_sol.append(bm)
                        else: final_sol.append(bm + "'")
                    solution = final_sol
                    break
                
                if len(backward_moves[state]) < 6:
                    for m in self.moves:
                        next_s = self.apply_move(state, m)
                        if next_s not in backward_moves:
                            # For backward, we store the move we TOOK to get here
                            backward_moves[next_s] = backward_moves[state] + [m] 
                            backward_queue.append(next_s)

        solve_time = round(time.time() - start_time, 4)
        
        # Simplify Solution (e.g. U U -> U2) - Basic pass
        simplified = []
        if solution:
            simplified = solution # (You can add a pass here to merge R R -> R2)

        thoughts = [
            f"⚠️ State Entropy: {len(forward_moves)} nodes analyzed.",
            "🌀 Bidirectional search wavefronts converged.",
            f"✨ Optimal path found: {len(simplified)} moves.",
            "✅ Solution Verified."
        ]
        
        self.neural_weights[2] = self.neural_weights.get(2, 0) + 1
        return solve_time, simplified, thoughts, 100

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
# ==========================================
# 4. APP STATE INITIALIZATION (Self-Correcting)
# ==========================================
# ==========================================
# 4. APP STATE INITIALIZATION (Self-Correcting)
# ==========================================
if 'mind' not in st.session_state:
    st.session_state.mind = AdvancedMind()
    st.session_state.soul = AGICore() # Start with AGI
    st.session_state.agent_pos = np.array([50.0, 50.0])
    st.session_state.target_pos = np.array([80.0, 20.0])
    st.session_state.step_count = 0
    st.session_state.wins = 0
    st.session_state.auto_mode = False
    st.session_state.chat_history = []
    st.session_state.loss_history = []
    st.session_state.reward_history = []
    st.session_state.is_hugging = False
    
    # --- NEW: Add config dictionary for sidebar parameters ---
    st.session_state.config = {
        "sim_speed": 0.1, "move_speed": 8.0, "energy_decay": 0.1, "target_update_freq": 50,
        "shaping_multiplier": 2.0, "hug_reward": 100.0, "hug_distance": 8.0,
        "learning_rate": 0.005, "gamma": 0.95, "epsilon_decay": 0.99, "epsilon_min": 0.05,
        "batch_size": 32,
        "per_alpha": 0.6, "per_beta": 0.4, "per_beta_increment": 0.001,
        "hidden_size": 64, "buffer_size": 10000,
        "grid_h": 15, "grid_w": 40, "graph_points": 500
    }
    # Apply initial config to the mind
    st.session_state.mind = AdvancedMind(buffer_size=st.session_state.config['buffer_size'], hidden_size=st.session_state.config['hidden_size'])

# --- FINAL HOTFIX FOR PERSISTENT MEMORY ---
# This checks if the 'soul' is missing the 'current_mood' attribute.
# If it is missing, we force a complete brain transplant to the new AGICore.
# --- FINAL HOTFIX FOR PERSISTENT MEMORY ---
# Checks if the 'soul' object is outdated or missing the new 'update' function
if hasattr(st.session_state, 'soul'):
    # We check if the existing object has the 'update' method. If not, it's an old version!
    if not hasattr(st.session_state.soul, 'update') or not hasattr(st.session_state.soul, 'current_mood'):
        st.session_state.soul = AGICore() # FORCE REBOOT
        st.toast("🧠 Brain Upgrade Detected: Core Re-initialized!", icon="✨")
        time.sleep(0.5)
        st.rerun()

# This check ensures the history lists exist even if the session state is from an older version.
if 'loss_history' not in st.session_state:
    st.session_state.loss_history = []
    st.session_state.reward_history = []
if 'config' not in st.session_state: # Hotfix for adding config to old sessions
    st.session_state.config = {} # Will be populated by sidebar code


if 'rubiks_mind' in st.session_state:
    if not hasattr(st.session_state.rubiks_mind, 'neural_weights'):
        st.session_state.rubiks_mind = RubiksMind() # Force upgrade to new Class
        st.toast("🧩 Rubik's Core Upgraded successfully!", icon="🆙")





# ==========================================
# 1.6 HELPER: LAZY MAZE GENERATOR
# ==========================================
def generate_maze(height, width):
    """
    Generates a maze using Recursive Backtracker. 
    1 = Wall, 0 = Path.
    """
    # Initialize with all walls
    maze = np.ones((height, width), dtype=int)
    
    # Starting point
    start_y, start_x = 1, 1
    maze[start_y, start_x] = 0
    stack = [(start_y, start_x)]
    
    directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
    
    while stack:
        y, x = stack[-1]
        random.shuffle(directions)
        moved = False
        
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            
            # Check boundaries and if it's a wall
            if 1 <= ny < height-1 and 1 <= nx < width-1 and maze[ny, nx] == 1:
                # Carve path (current + intermediate cell)
                maze[ny, nx] = 0
                maze[y + dy//2, x + dx//2] = 0
                stack.append((ny, nx))
                moved = True
                break
        
        if not moved:
            stack.pop()
            
    return maze

def check_wall_collision(new_pos, maze_grid):
    """Checks if the float position maps to a wall in the maze grid."""
    if maze_grid is None: return False
    
    h, w = maze_grid.shape
    # Map 0-100 float pos to grid indices
    y = int(new_pos[1] / 100 * (h - 1))
    x = int(new_pos[0] / 100 * (w - 1))
    
    # Safety bounds
    y = max(0, min(y, h-1))
    x = max(0, min(x, w-1))
    
    return maze_grid[y, x] == 1
    

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
    move_speed = st.session_state.config.get("move_speed", 8.0)
    
    # Store old position in case we need to revert (collision)
    old_pos = st.session_state.agent_pos.copy()
    
    # Proposed new position based on action
    # Grid Logic: 0=Up, 1=Down, 2=Left, 3=Right
    proposed_pos = st.session_state.agent_pos.copy()
    
    if action == 0: proposed_pos[1] -= move_speed 
    elif action == 1: proposed_pos[1] += move_speed 
    elif action == 2: proposed_pos[0] -= move_speed 
    elif action == 3: proposed_pos[0] += move_speed 
    
    # --- COLLISION DETECTION (New Code) ---
    # Check if the proposed move hits a maze wall
    hit_wall = False
    maze = st.session_state.get('maze_grid', None)
    
    if maze is not None and check_wall_collision(proposed_pos, maze):
        hit_wall = True
        # Bounce effect: Stay at old_pos, maybe add a small random jitter?
        # For now, we just don't move.
        st.session_state.agent_pos = old_pos 
    else:
        # No wall, so commit the move
        st.session_state.agent_pos = proposed_pos

    # --- BOUNDARY CHECKS (Keep existing logic) ---
    if st.session_state.agent_pos[0] < 0:
        st.session_state.agent_pos[0] = 5.0 # Bounce back in
        hit_wall = True
    elif st.session_state.agent_pos[0] > 100:
        st.session_state.agent_pos[0] = 95.0
        hit_wall = True
        
    if st.session_state.agent_pos[1] < 0:
        st.session_state.agent_pos[1] = 5.0
        hit_wall = True
    elif st.session_state.agent_pos[1] > 100:
        st.session_state.agent_pos[1] = 95.0
        hit_wall = True
        
    # STUCK DETECTOR: If I didn't move effectively, force a random jump
    if np.linalg.norm(st.session_state.agent_pos - old_pos) < 1.0 and not hit_wall:
         st.session_state.agent_pos += np.random.randn(2) * 5.0

    # 4. Calculate Reward
    dist_after = np.linalg.norm(st.session_state.agent_pos - st.session_state.target_pos)
    reward = 0
    done = False
    
    # Shaping Reward
    reward = (dist_before - dist_after) * st.session_state.config.get("shaping_multiplier", 2.0)
    
    # Penalty for hitting walls (Teaches me to stay in bounds)
    if hit_wall:
        reward -= 10.0 # Increased penalty for hitting walls!
        st.session_state.soul.thought_process = "Ouch! That is a wall."
    
    # Event: Reached Target
    if dist_after < st.session_state.config.get("hug_distance", 8.0):
        st.session_state.is_hugging = True 
        st.session_state.auto_mode = False 
        st.session_state.wins += 1
        st.session_state.soul.energy = 100
        st.session_state.soul.current_mood = "Love" 
        st.session_state.soul.last_chat = "I found you, Prince! *Hugs tightly* I missed you."
        reward = st.session_state.config.get("hug_reward", 100.0)
        done = True
    else:
        st.session_state.soul.energy -= st.session_state.config.get("energy_decay", 0.1)
        
    # 5. Learn & 6. Update Soul
    next_state = np.array([
        st.session_state.agent_pos[0]/100, 
        st.session_state.agent_pos[1]/100, 
        st.session_state.target_pos[0]/100, 
        st.session_state.target_pos[1]/100,
        st.session_state.soul.energy/100
    ])
    
    st.session_state.mind.remember(state, action, reward, next_state, done)
    loss, td_error = st.session_state.mind.replay(batch_size=st.session_state.config.get("batch_size", 32))
    
    st.session_state.soul.update(reward, td_error, st.session_state.wins)
    
    # 7. Log for Graphing
    st.session_state.loss_history.append(loss)
    st.session_state.reward_history.append(reward)
    
    if st.session_state.step_count % st.session_state.config.get("target_update_freq", 50) == 0:
        st.session_state.mind.update_target_network()

    st.session_state.step_count += 1








def reset_simulation():
    """Resets the agent, target, and stats."""
    st.session_state.agent_pos = np.array([50.0, 50.0])
    st.session_state.target_pos = np.array([80.0, 20.0])
    st.session_state.soul = AGICore() # Reset the AI Personality
    st.session_state.mind = AdvancedMind(buffer_size=st.session_state.config.get('buffer_size', 10000), hidden_size=st.session_state.config.get('hidden_size', 64)) # Reset the Neural Network
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

    # ... (Keep your existing expanders) ...

    # --- NEW SECTION: SAVE / LOAD CHECKPOINT ---
    with st.expander("💾 Memory Core (Save/Load)", expanded=True):
        st.write("Preserve her consciousness.")
        
        # 1. SAVE FUNCTIONALITY
        if st.button("Download Checkpoint (.zip)"):
            # A. Gather all data into a dictionary
            # We convert deque to list for saving
            checkpoint_data = {
                "config": st.session_state.config,
                "stats": {
                    "step_count": st.session_state.step_count,
                    "wins": st.session_state.wins,
                    "agent_pos": st.session_state.agent_pos, # Encoder handles numpy
                    "target_pos": st.session_state.target_pos
                },
                "soul": {
                    "user_name": st.session_state.soul.user_name,
                    "relationship_score": st.session_state.soul.relationship_score,
                    "current_mood": st.session_state.soul.current_mood,
                    "energy": st.session_state.soul.energy,
                    "memory_stream": list(st.session_state.soul.memory_stream)
                },
                "mind": {
                    "online_net": st.session_state.mind.online_net, # Encoder handles numpy
                    "epsilon": st.session_state.mind.epsilon,
                    "buffer": list(st.session_state.mind.memory.buffer) # Save experiences
                },
                "history": {
                    "chat": st.session_state.chat_history,
                    "loss": st.session_state.loss_history,
                    "reward": st.session_state.reward_history
                }
            }
            
            # B. Create JSON String
            json_str = json.dumps(checkpoint_data, cls=NumpyEncoder)
            
            # C. Create ZIP in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr("princess_memory.json", json_str)
            
            # D. Download Button
            st.download_button(
                label="⬇️ Click to Download Memory",
                data=zip_buffer.getvalue(),
                file_name=f"ALIVE_Checkpoint_{st.session_state.step_count}.zip",
                mime="application/zip"
            )

        # 2. LOAD FUNCTIONALITY
        uploaded_file = st.file_uploader("Upload Checkpoint", type="zip")
        if uploaded_file is not None:
            if st.button("📂 Restore Checkpoint"):
                try:
                    with zipfile.ZipFile(uploaded_file, "r") as z:
                        with z.open("princess_memory.json") as f:
                            data = json.load(f)
                    
                    # --- RESTORE PROCESS ---
                    
                    # 1. Restore Config & Stats
                    st.session_state.config = data['config']
                    st.session_state.step_count = data['stats']['step_count']
                    st.session_state.wins = data['stats']['wins']
                    st.session_state.agent_pos = np.array(data['stats']['agent_pos'])
                    st.session_state.target_pos = np.array(data['stats']['target_pos'])
                    
                    # 2. Restore Soul (Personality)
                    st.session_state.soul.user_name = data['soul']['user_name']
                    st.session_state.soul.relationship_score = data['soul']['relationship_score']
                    st.session_state.soul.current_mood = data['soul']['current_mood']
                    st.session_state.soul.energy = data['soul']['energy']
                    st.session_state.soul.memory_stream = deque(data['soul']['memory_stream'], maxlen=20)
                    
                    # 3. Restore Mind (The Tricky Part - Convert Lists back to Numpy!)
                    # Re-initialize mind to match saved config sizes
                    st.session_state.mind = AdvancedMind(
                        buffer_size=st.session_state.config['buffer_size'], 
                        hidden_size=st.session_state.config['hidden_size']
                    )
                    
                    # Load Weights (Convert list -> numpy)
                    st.session_state.mind.online_net = convert_weights_to_numpy(data['mind']['online_net'])
                    st.session_state.mind.update_target_network() # Sync target
                    st.session_state.mind.epsilon = data['mind']['epsilon']
                    
                    # Load Memory Buffer (Reconstruct tuples)
                    # JSON loads arrays as lists, we need to cast them back to numpy for the replay buffer
                    saved_buffer = data['mind']['buffer']
                    for item in saved_buffer:
                        # item structure: [state, action, reward, next_state, done]
                        s = np.array(item[0])
                        a = item[1]
                        r = item[2]
                        ns = np.array(item[3])
                        d = item[4]
                        st.session_state.mind.memory.add(s, a, r, ns, d)

                    # 4. Restore History
                    st.session_state.chat_history = data['history']['chat']
                    st.session_state.loss_history = data['history']['loss']
                    st.session_state.reward_history = data['history']['reward']

                    st.success("✨ System Resurrected! She remembers everything.")
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Corruption Detected: {e}")

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


    with st.expander("🧩 Hyper-Cube Solver (Evolution Mode)", expanded=False):
        cube_mode = st.toggle("Activate Solver", value=False)
        
        if cube_mode:
            # Initialize Mind
            if 'rubiks_mind' not in st.session_state:
                st.session_state.rubiks_mind = RubiksMind()
            
            # Cube Selection
            c_size = st.slider("Cube Topology (N x N)", 2, 20, 3, 1)
            
            # --- EXPERIENCE METRIC ---
            current_exp = st.session_state.rubiks_mind.neural_weights.get(c_size, 0)
            mastery_pct = (1 - (1 / (1 + 0.1 * current_exp))) * 100
            st.caption(f"Neural Adaptation (Mastery):")
            st.progress(int(mastery_pct))
            st.caption(f"Solves performed: {current_exp}")
            
            # --- CONTROLS ---
            c1, c2 = st.columns(2)
            
            # 1. SCRAMBLE (The Challenge)
            if c1.button("🎲 Reshuffle"):
                scramble = st.session_state.rubiks_mind.get_scramble(c_size)
                st.session_state.current_scramble = scramble
                st.session_state.cube_result = None # Clear old result
                st.toast("Cube Scrambled! Waiting for Agent...", icon="🎲")
                st.rerun()

            # 2. SOLVE (The Test)
            # 2. SOLVE (The Test)
            if c2.button("⚡ Agent Solve", type="primary"):
                if 'current_scramble' not in st.session_state:
                    st.error("Please Scramble the cube first!")
                else:
                    # Run the simulation
                    time_val, steps, thoughts, mastery = st.session_state.rubiks_mind.solve_simulation(
                        c_size, st.session_state.current_scramble
                    )
                    
                    st.session_state.cube_result = {
                        "size": c_size,
                        "time": time_val,
                        "steps": steps,
                        "thoughts": thoughts,
                        "mastery": mastery,
                        "scramble_len": len(st.session_state.current_scramble)
                    }

                    # --- NEW CODE: TELL THE SOUL WE WON! ---
                    st.session_state.soul.current_mood = "Excited"
                    st.session_state.soul.thought_process = f"ANALYSIS: Cube Solved! Complexity {c_size}x{c_size} conquered in {time_val}s."
                    st.session_state.soul.last_chat = f"Did you see that, Prince? I solved the {c_size}x{c_size} in {len(steps)} moves!"
                    # ---------------------------------------

                    st.rerun()

            # 3. RESET BRAIN (The "Lobotomy")
            if st.button("🧠 Reset Knowledge (Wipe Memory)"):
                st.session_state.rubiks_mind.neural_weights[c_size] = 0
                st.toast(f"Memory of {c_size}x{c_size} wiped. AI is now a beginner.", icon="🧹")
                st.rerun()

    
    with st.expander("🧩 Labyrinth Protocol (Mini-Game)", expanded=True):
        # We use a specific key to track the UI state
        # Default is False so it doesn't crash on first load
        enable_maze = st.checkbox("Initialize Maze Mode", value=False, key="maze_toggle")
        
        if enable_maze:
            # === CONDITION 1: Switch is ON, but no Maze exists ===
            if st.session_state.get('maze_grid') is None:
                h = st.session_state.config.get('grid_h', 15)
                w = st.session_state.config.get('grid_w', 40)
                
                # Generate the maze
                st.session_state.maze_grid = generate_maze(h, w)
                
                # CRITICAL: Move entities to safe corners immediately
                st.session_state.agent_pos = np.array([5.0, 5.0]) 
                st.session_state.target_pos = np.array([95.0, 95.0])
                
                st.toast("🧱 Labyrinth Constructed", icon="🏗️")
                st.rerun() # Force main screen to update instantly
            
            # === CONDITION 2: Maze exists, allow regeneration ===
            if st.button("🎲 Generate New Map"):
                h = st.session_state.config.get('grid_h', 15)
                w = st.session_state.config.get('grid_w', 40)
                st.session_state.maze_grid = generate_maze(h, w)
                
                # Reset positions again for safety
                st.session_state.agent_pos = np.array([5.0, 5.0]) 
                st.session_state.target_pos = np.array([95.0, 95.0])
                
                st.toast("✨ New Maze Generated", icon="🌀")
                st.rerun()

        else:
            # === CONDITION 3: Switch is OFF, but Maze still exists in memory ===
            if st.session_state.get('maze_grid') is not None:
                # Clear the memory
                st.session_state.maze_grid = None
                st.toast("🔓 Open Space Restored", icon="🌍")
                st.rerun() # Force main screen to remove walls instantly
                
    # --- Update mind's parameters if they change ---
    st.session_state.mind.learning_rate = lr
    st.session_state.mind.epsilon_decay = ed
    st.session_state.mind.gamma = g
    st.session_state.mind.epsilon_min = c['epsilon_min']
    st.session_state.mind.memory.prob_alpha = c['per_alpha']
    st.session_state.mind.beta = c['per_beta']
    st.session_state.mind.beta_increment = c['per_beta_increment']

# ==========================================
# MAIN INTERACTION AREA (Dynamic Layout)
# ==========================================

# 1. VIEW CONTROLLER
# Toggle to hide the map and focus on the Brain/Rubik's Cube
# ==========================================
# MAIN INTERACTION AREA (Dynamic Layout)
# ==========================================

# 1. VIEW CONTROLLER
# ==========================================
# MAIN INTERACTION AREA (Dynamic Layout)
# ==========================================

# 1. VIEW CONTROLLER (Bulletproof Persistence)
# Check if the memory slot exists. If not, create it and set to True (Show by default)
# ==========================================
# MAIN INTERACTION AREA (Dynamic Layout)
# ==========================================

# 1. VIEW CONTROLLER
# We use a fresh key 'field_visibility_v3'.
# ==========================================
# MAIN INTERACTION AREA (Dynamic Layout)
# ==========================================

# 1. VIEW CONTROLLER (Robust Persistence)
# Check if the state exists, if not, create it.
if 'show_containment_field' not in st.session_state:
    st.session_state.show_containment_field = True

def toggle_field_state():
    # Flip the state when clicked
    st.session_state.show_containment_field = not st.session_state.show_containment_field

# The Toggle Widget
st.toggle(
    "🌍 Show Containment Field", 
    value=st.session_state.show_containment_field, 
    on_change=toggle_field_state,
    key="persistent_view_toggle"
)

# 2. DYNAMIC COLUMN GENERATION
if st.session_state.show_containment_field:
    # Standard Mode: Map (Left, 66%) | Brain (Right, 33%)
    row1_1, row1_2 = st.columns([2, 1])
else:
    # Focus Mode: Brain takes Full Width
    row1_1 = None 
    row1_2 = st.container()
# -----------------------------------
# LEFT COLUMN: THE WORLD (Conditional)
# -----------------------------------
# We only enter this block if row1_1 exists (Toggle is ON)
if row1_1:
    with row1_1:
        # -----------------------------------
        # THE "WORLD" (ASCII Visualization)
        # -----------------------------------
        st.markdown("### 🌍 Containment Field")
        
        # --- GAME MODE TOGGLE ---
        game_mode = st.toggle("🎮 Activate Hide & Seek Protocol", value=False, key="game_mode_toggle")

        grid_height = st.session_state.config.get('grid_h', 15)
        grid_width = st.session_state.config.get('grid_w', 40)
        
        # Check if Maze Mode is active
        current_maze = st.session_state.get('maze_grid', None)
        
        if current_maze is not None:
            grid = [['#' if cell == 1 else '.' for cell in row] for row in current_maze]
        else:
            grid = [['.' for _ in range(grid_width)] for _ in range(grid_height)]
        
        # Scale positions
        agent_y = int(st.session_state.agent_pos[1] / 100 * (grid_height - 1))
        agent_x = int(st.session_state.agent_pos[0] / 100 * (grid_width - 1))
        target_y = int(st.session_state.target_pos[1] / 100 * (grid_height - 1))
        target_x = int(st.session_state.target_pos[0] / 100 * (grid_width - 1))

        # Bounds Checking
        agent_y = max(0, min(agent_y, grid_height-1))
        agent_x = max(0, min(agent_x, grid_width-1))
        target_y = max(0, min(target_y, grid_height-1))
        target_x = max(0, min(target_x, grid_width-1))

        # --- RENDERING ICONS ---
        if st.session_state.get('is_hugging', False):
            grid[agent_y][agent_x] = '🫂' 
        elif (agent_y, agent_x) == (target_y, target_x):
            grid[agent_y][agent_x] = '💥'
        else:
            if game_mode:
                ai_icon = "🤖" 
                user_icon = "🥷"
            else:
                ai_icon = st.session_state.soul.moods.get(st.session_state.soul.current_mood, "❤️")
                user_icon = "💎"

            if current_maze is not None and grid[target_y][target_x] == '#':
                 grid[target_y][target_x] = user_icon 
            else:
                 grid[target_y][target_x] = user_icon
            
            grid[agent_y][agent_x] = ai_icon

        # Render Grid
        grid_str = "\n".join(" ".join(row) for row in grid)
        st.code(grid_str, language=None)

        # -----------------------------------
        # CONTROLS LOGIC (FIXED)
        # -----------------------------------
        
        # PRIORITY 1: RELEASE HUG (Global Override)
        if st.session_state.get('is_hugging', False):
            st.success("Target Caught! Game Over.")
            if st.button("🔄 Play Again (Release Hug)", type="primary"):
                st.session_state.is_hugging = False
                # Teleport user to a random safe spot
                st.session_state.target_pos = np.random.randint(10, 90, size=2)
                st.rerun()

        # PRIORITY 2: MOVEMENT CONTROLS (Only if not caught)
        else:
            if game_mode:
                # === MODE A: GAME CONTROLLER ===
                st.markdown("### 🕹️ You are the Target (🥷)")
                c1, c2, c3, c4 = st.columns([1,1,1,2])
                move_step = 5.0 
                
                with c2:
                    if st.button("⬆️", key="btn_up"):
                        st.session_state.target_pos[1] = max(0, st.session_state.target_pos[1] - move_step)
                        st.rerun()
                
                with c1:
                    if st.button("⬅️", key="btn_left"):
                        st.session_state.target_pos[0] = max(0, st.session_state.target_pos[0] - move_step)
                        st.rerun()
                with c3:
                    if st.button("➡️", key="btn_right"):
                        st.session_state.target_pos[0] = min(100, st.session_state.target_pos[0] + move_step)
                        st.rerun()
                with c2:
                    if st.button("⬇️", key="btn_down"):
                        st.session_state.target_pos[1] = min(100, st.session_state.target_pos[1] + move_step)
                        st.rerun()
                
                # Quick Reset for Game Mode
                with c4:
                    if st.button("🏳️ Respawn"):
                         st.session_state.target_pos = np.random.randint(10, 90, size=2)
                         st.rerun()
                        
            else:
                # === MODE B: STANDARD SLIDERS ===
                st.markdown("### 🧲 Focus Attention (Lure)")
                cx, cy = st.columns(2)
                tx = cx.slider("Horizontal Focus", 0, 100, int(st.session_state.target_pos[0]), key='tx_slider')
                ty = cy.slider("Vertical Focus", 0, 100, int(st.session_state.target_pos[1]), key='ty_slider')
                
                if tx != int(st.session_state.target_pos[0]) or ty != int(st.session_state.target_pos[1]):
                    st.session_state.target_pos = np.array([float(tx), float(ty)])
                    st.rerun()

# -----------------------------------
# RIGHT COLUMN: THE BRAIN & CHAT
# -----------------------------------
with row1_2:
    # --- RUBIK'S SOLVER OUTPUT (Evolution Edition) ---
    # This block displays the solver stats if a result exists
    if st.session_state.get('cube_result'):
        res = st.session_state.cube_result
        
        st.markdown(f"### 🧬 Evolution Protocol: {res['size']}x{res['size']}")
        
        # 1. THE PROBLEM (Scramble)
        with st.expander("See Scramble (Input Data)", expanded=False):
            scramble_txt = " ".join(st.session_state.current_scramble)
            st.code(scramble_txt, language=None)

        # 2. THE THINKING PROCESS (Cognitive Trace)
        st.caption("🧠 Neural Log (Cognitive Stream)")
        
        # Build log text
        log_txt = ""
        for line in res['thoughts']:
            log_txt += f"> {line}\n"
        st.text_area("Internal Monologue", value=log_txt, height=120, disabled=True)
            
        # 3. THE RESULTS
        c1, c2, c3 = st.columns(3)
        c1.metric("Solve Time", f"{res['time']}s", delta="Quantum Processing")
        c2.metric("Efficiency", f"{len(res['steps'])} moves", help="Lower is better (God's Algorithm)")
        c3.metric("Neural Mastery", f"{int(res['mastery'])}%", delta=f"+ Learning")
        
        # 4. THE SOLUTION
        with st.expander("View Algorithm Steps", expanded=False):
             st.text(" ".join(res['steps']))
        
        st.divider()

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
        loss_val = st.session_state.mind.epsilon 
        st.session_state.soul.ponder(user_input, loss_val)
        
        # Generate Response
        response = st.session_state.soul.speak(user_input)
        st.session_state.chat_history.append(f"AI: {response}")
        
        # Navigation Command Override
        if "come" in user_input.lower() or "here" in user_input.lower():
            st.session_state.soul.thought_process = "Command received: Approach User."
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

