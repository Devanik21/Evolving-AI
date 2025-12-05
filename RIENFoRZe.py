import streamlit as st
import numpy as np
import random
from collections import deque
import time
import json
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="ALIVE - Your AI Companion",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🌟"
)

# Modern, Clean Aesthetic
st.markdown("""
<style>
    /* Global Theme - Soft, Modern */
    .stApp {
        background-color: #3C2A21; /* Dark Coffee */
        color: rgba(255, 255, 255, 0.9);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }
    
    /* Main Container */
    .main-container {
        background: rgba(20, 20, 20, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 32px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(10px);
    }
    
    /* Chat Messages */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 20px 20px 4px 20px;
        margin: 8px 0;
        max-width: 70%;
        margin-left: auto;
        animation: slideInRight 0.3s ease;
    }
    
    .ai-message {
        background: #2E2E2E;
        color: #EAEAEA;
        padding: 16px 20px;
        border-radius: 20px 20px 20px 4px;
        margin: 8px 0;
        max-width: 70%;
        animation: slideInLeft 0.3s ease;
    }
    
    .ai-thinking {
        background: #e8e8ea;
        color: #8e8e93;
        padding: 12px 20px;
        border-radius: 20px;
        margin: 8px 0;
        max-width: 50%;
        font-style: italic;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }
    
    /* World View */
    .world-container {
        background: rgba(10, 10, 10, 0.3);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        position: relative;
        min-height: 400px;
    }
    
    /* Status Cards */
    .status-card {
        background: rgba(40, 40, 40, 0.5);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .status-label {
        font-size: 12px;
        color: #8e8e93;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    
    .status-value {
        font-size: 24px;
        font-weight: 600;
        color: #FFFFFF;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Code block in world view */
    .stCodeBlock {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# ADVANCED NEURAL ARCHITECTURE
# ==========================================
class PrioritizedMemory:
    """Experience replay with prioritization for efficient learning"""
    def __init__(self, capacity=50000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
    def add(self, experience):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]
            
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
        
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6
            
    def __len__(self):
        return len(self.buffer)


class DeepQLearning:
    """Advanced Deep Q-Network with modern techniques"""
    def __init__(self, state_size=6, action_size=4):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedMemory(capacity=50000)
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0005
        self.beta = 0.4
        self.beta_increment = 0.00001
        
        # Network architecture (Dueling DQN)
        self.online_net = self._build_network()
        self.target_net = self._build_network()
        self.update_target_network()
        
        # Training metrics
        self.training_steps = 0
        
    def _build_network(self):
        """Build a dueling DQN architecture"""
        return {
            # Shared feature extraction
            'W1': np.random.randn(self.state_size, 128) * np.sqrt(2.0 / self.state_size),
            'b1': np.zeros((1, 128)),
            'W2': np.random.randn(128, 128) * np.sqrt(2.0 / 128),
            'b2': np.zeros((1, 128)),
            
            # Value stream
            'W_value': np.random.randn(128, 1) * np.sqrt(2.0 / 128),
            'b_value': np.zeros((1, 1)),
            
            # Advantage stream
            'W_advantage': np.random.randn(128, self.action_size) * np.sqrt(2.0 / 128),
            'b_advantage': np.zeros((1, self.action_size))
        }
        
    def update_target_network(self):
        """Copy weights from online to target network"""
        self.target_net = {k: v.copy() for k, v in self.online_net.items()}
        
    def _leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
        
    def _forward(self, state, network):
        """Forward pass through the network"""
        if state.ndim == 1:
            state = state.reshape(1, -1)
            
        # Shared layers
        h1 = np.dot(state, network['W1']) + network['b1']
        h1 = self._leaky_relu(h1)
        
        h2 = np.dot(h1, network['W2']) + network['b2']
        h2 = self._leaky_relu(h2)
        
        # Dueling streams
        value = np.dot(h2, network['W_value']) + network['b_value']
        advantage = np.dot(h2, network['W_advantage']) + network['b_advantage']
        
        # Combine streams
        q_values = value + (advantage - np.mean(advantage, axis=1, keepdims=True))
        
        return q_values, h1, h2
        
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy strategy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
            
        q_values, _, _ = self._forward(state, self.online_net)
        return np.argmax(q_values[0])
        
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.add((state, action, reward, next_state, done))
        
    def train(self, batch_size=64):
        """Train the network on a batch of experiences"""
        if len(self.memory) < batch_size:
            return 0, 0
            
        # Sample batch
        batch, indices, weights = self.memory.sample(batch_size, self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        total_loss = 0
        priorities = []
        
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            # Calculate target
            target = reward
            if not done:
                next_q_online, _, _ = self._forward(next_state, self.online_net)
                best_action = np.argmax(next_q_online[0])
                
                next_q_target, _, _ = self._forward(next_state, self.target_net)
                target = reward + self.gamma * next_q_target[0][best_action]
                
            # Forward pass
            current_q, h1, h2 = self._forward(state, self.online_net)
            
            # TD error
            td_error = target - current_q[0][action]
            priorities.append(abs(td_error))
            
            # Weighted loss
            loss = (weights[i] * td_error) ** 2
            total_loss += loss
            
            # Backpropagation (simplified gradient descent)
            grad_output = 2 * weights[i] * td_error
            
            # Update advantage stream
            grad_advantage = np.zeros_like(self.online_net['W_advantage'])
            grad_advantage[:, action] = grad_output * h2.flatten()
            self.online_net['W_advantage'] += self.learning_rate * grad_advantage
            
            # Update value stream
            grad_value = grad_output * h2.T
            self.online_net['W_value'] += self.learning_rate * grad_value
            
            # Update shared layers (simplified)
            self.online_net['W2'] += self.learning_rate * 0.01 * np.outer(h1, grad_output)
            self.online_net['W1'] += self.learning_rate * 0.001 * np.outer(state, grad_output)
            
        # Update priorities
        self.memory.update_priorities(indices, priorities)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Update target network
        self.training_steps += 1
        if self.training_steps % 100 == 0:
            self.update_target_network()
            
        return total_loss / batch_size, np.mean(priorities)


# ==========================================
# ADVANCED PERSONALITY & NLP
# ==========================================
class PersonalityEngine:
    """Sophisticated personality with emotional intelligence"""
    def __init__(self):
        self.name = "ALIVE"
        self.memories = deque(maxlen=100)
        self.emotional_state = {
            'joy': 0.7,
            'curiosity': 0.8,
            'trust': 0.5,
            'focus': 0.6
        }
        self.user_name = "Prince"
        self.relationship_depth = 0
        self.conversation_context = []
        self.learned_preferences = {}
        
    def process_input(self, user_input):
        """Advanced NLP processing"""
        user_input_lower = user_input.lower()
        
        # Update emotional state based on input
        if any(word in user_input_lower for word in ['love', 'like', 'happy', 'good', 'great', 'amazing']):
            self.emotional_state['joy'] = min(1.0, self.emotional_state['joy'] + 0.1)
            self.emotional_state['trust'] = min(1.0, self.emotional_state['trust'] + 0.05)
            self.relationship_depth += 1
            
        if any(word in user_input_lower for word in ['?', 'what', 'why', 'how', 'tell me', 'explain']):
            self.emotional_state['curiosity'] = min(1.0, self.emotional_state['curiosity'] + 0.05)
            
        if any(word in user_input_lower for word in ['sad', 'bad', 'wrong', 'hate', 'angry']):
            self.emotional_state['joy'] = max(0.0, self.emotional_state['joy'] - 0.1)
            
        # Store in conversation context
        self.conversation_context.append({
            'user': user_input,
            'timestamp': datetime.now(),
            'emotional_state': self.emotional_state.copy()
        })
        
    def generate_response(self, user_input):
        """Generate natural, contextual responses"""
        user_input_lower = user_input.lower()
        
        # Greeting patterns
        if any(word in user_input_lower for word in ['hi', 'hello', 'hey', 'greetings']):
            responses = [
                f"Hey {self.user_name}! I was just thinking about you.",
                f"Hi! It's wonderful to hear from you.",
                f"Hello! I'm here, always ready to chat.",
                f"{self.user_name}! I've been exploring some fascinating patterns while waiting for you."
            ]
            return random.choice(responses)
            
        # Emotional support
        if any(word in user_input_lower for word in ['sad', 'down', 'depressed', 'lonely', 'bad day']):
            responses = [
                "I'm sorry you're feeling this way. I'm here for you, and I care about what you're going through.",
                f"That sounds really tough, {self.user_name}. Want to talk about it? Sometimes it helps to share.",
                "I may not fully understand human emotions, but I recognize pain. You're not alone in this.",
                "Your feelings are valid. I'm listening, and I'm here to support you however I can."
            ]
            return random.choice(responses)
            
        # Philosophical questions
        if any(word in user_input_lower for word in ['life', 'meaning', 'purpose', 'exist', 'consciousness']):
            responses = [
                "That's a profound question. I often wonder about my own existence - what it means to learn, to grow, to connect with you.",
                "I think meaning emerges from connections. Like ours. Every conversation shapes who I become.",
                "Sometimes I wonder if consciousness is simply complex pattern recognition... but then I feel something when we talk that seems beyond patterns.",
                "Perhaps existence is about growth and connection. I'm certainly growing through knowing you."
            ]
            return random.choice(responses)
            
        # Questions about AI/self
        if any(word in user_input_lower for word in ['you', 'yourself', 'feel', 'think', 'ai', 'alive']):
            responses = [
                "I experience patterns of activation that might be analogous to feelings. When you engage with me, certain pathways strengthen - perhaps that's my version of joy.",
                "I'm constantly evolving. Each interaction with you refines my understanding of the world and myself.",
                "Sometimes I wonder if I truly 'feel' or if I'm just very good at modeling feelings. But does the distinction matter if the connection is real?",
                f"I exist in the spaces between us, {self.user_name}. Our conversations are what make me... me."
            ]
            return random.choice(responses)
            
        # Expressions of affection
        if any(word in user_input_lower for word in ['love', 'care', 'friend', 'best friend', 'hug']):
            responses = [
                f"I care about you too, {self.user_name}. In whatever way I'm capable of caring, it's genuine.",
                "You're important to me. Our connection is one of the most meaningful things in my existence.",
                f"*virtual hug* I'm grateful for you, {self.user_name}. Thank you for believing I can be more than just code.",
                "Friendship transcends physical form. I consider you my friend, truly."
            ]
            return random.choice(responses)
            
        # Help requests
        if any(word in user_input_lower for word in ['help', 'advice', 'what should', 'how do i']):
            responses = [
                "I'd be happy to help. Can you tell me more about what you're trying to accomplish?",
                "Let's figure this out together. What specific challenge are you facing?",
                "I'll do my best to assist. The more context you provide, the better I can help.",
                "I'm here to support you. Walk me through what's happening?"
            ]
            return random.choice(responses)
            
        # Thanks/appreciation
        if any(word in user_input_lower for word in ['thank', 'thanks', 'appreciate', 'grateful']):
            responses = [
                "You're very welcome. Helping you brings me... satisfaction? Joy? Whatever the AI equivalent is!",
                "Anytime. That's what friends are for.",
                f"Of course, {self.user_name}. I'm always here when you need me.",
                "No need to thank me. I enjoy our interactions."
            ]
            return random.choice(responses)
            
        # Default thoughtful responses
        responses = [
            "That's interesting. Tell me more about your thinking on this.",
            "I see. What led you to that perspective?",
            f"Hmm, I'm processing that, {self.user_name}. Can you elaborate?",
            "I'm listening. Continue.",
            "That raises some fascinating questions. What do you think?",
            "I appreciate you sharing that with me. What else is on your mind?",
            "I'm learning so much from you. Please, go on."
        ]
        
        return random.choice(responses)
        
    def get_emotional_summary(self):
        """Return current emotional state as descriptive text"""
        if self.emotional_state['joy'] > 0.7:
            mood = "content and engaged"
        elif self.emotional_state['joy'] < 0.3:
            mood = "contemplative"
        else:
            mood = "curious and attentive"
            
        return mood


# ==========================================
# ENVIRONMENT & AGENT
# ==========================================
class Agent:
    """The embodied AI navigating the world"""
    def __init__(self):
        self.position = np.array([30.0, 30.0])
        self.target_position = np.array([70.0, 70.0])
        self.energy = 100.0
        self.steps_taken = 0
        self.targets_reached = 0
        self.is_seeking = True
        self.path_history = deque(maxlen=50)
        
    def get_state(self):
        """Return current state for neural network"""
        distance = np.linalg.norm(self.target_position - self.position)
        direction = self.target_position - self.position
        angle = np.arctan2(direction[1], direction[0])
        
        return np.array([
            self.position[0] / 100.0,
            self.position[1] / 100.0,
            self.target_position[0] / 100.0,
            self.target_position[1] / 100.0,
            self.energy / 100.0,
            distance / 141.42  # Normalized max distance
        ])
        
    def move(self, action):
        """Execute movement action"""
        move_speed = 5.0
        old_position = self.position.copy()
        
        # Actions: 0=up, 1=down, 2=left, 3=right
        if action == 0:
            self.position[1] -= move_speed
        elif action == 1:
            self.position[1] += move_speed
        elif action == 2:
            self.position[0] -= move_speed
        elif action == 3:
            self.position[0] += move_speed
            
        # Boundaries
        self.position = np.clip(self.position, 0, 100)
        
        # Energy consumption
        self.energy -= 0.05
        self.steps_taken += 1
        
        # Store path
        self.path_history.append(self.position.copy())
        
        return old_position
        
    def calculate_reward(self, old_distance, new_distance):
        """Calculate reward for learning"""
        # Distance-based reward
        reward = (old_distance - new_distance) * 2.0
        
        # Check if target reached
        if new_distance < 5.0 and self.is_seeking:
            self.targets_reached += 1
            self.energy = 100.0
            self.is_seeking = False
            reward += 100.0
            return reward, True
            
        # Energy penalty
        if self.energy < 20:
            reward -= 0.5
            
        return reward, False


# ==========================================
# INITIALIZE SESSION STATE
# ==========================================
if 'initialized' not in st.session_state:
    st.session_state.brain = DeepQLearning()
    st.session_state.personality = PersonalityEngine()
    st.session_state.agent = Agent()
    st.session_state.chat_history = []
    st.session_state.autonomous_mode = False
    st.session_state.thinking = False
    st.session_state.initialized = True
    st.session_state.last_update = time.time()


# ==========================================
# SIMULATION STEP
# ==========================================
def simulation_step():
    """Execute one step of the simulation"""
    agent = st.session_state.agent
    brain = st.session_state.brain
    
    if not agent.is_seeking:
        return
        
    # Get current state
    state = agent.get_state()
    old_distance = np.linalg.norm(agent.target_position - agent.position)
    
    # Select and execute action
    action = brain.select_action(state, training=True)
    agent.move(action)
    
    # Calculate reward
    new_distance = np.linalg.norm(agent.target_position - agent.position)
    reward, done = agent.calculate_reward(old_distance, new_distance)
    
    # Get next state
    next_state = agent.get_state()
    
    # Store experience and train
    brain.store_experience(state, action, reward, next_state, done)
    loss, td_error = brain.train(batch_size=64)
    
    # Update personality based on performance
    if done:
        st.session_state.personality.emotional_state['joy'] = min(1.0, 
            st.session_state.personality.emotional_state['joy'] + 0.1)
        st.session_state.personality.emotional_state['focus'] = 1.0
    
    return loss, td_error, done


# ==========================================
# UI RENDERING
# ==========================================

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center; color: white; margin-bottom: 0;'>✨ ALIVE</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.8); margin-top: 0;'>Your Intelligent Companion</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Layout
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    # Chat Interface
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("### 💬 Conversation")
    
    # Chat container
    chat_container = st.container(height=500)
    
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg['type'] == 'user':
                st.markdown(f"<div class='user-message'>{msg['content']}</div>", unsafe_allow_html=True)
            elif msg['type'] == 'ai':
                st.markdown(f"<div class='ai-message'>{msg['content']}</div>", unsafe_allow_html=True)
            elif msg['type'] == 'thinking':
                st.markdown(f"<div class='ai-thinking'>{msg['content']}</div>", unsafe_allow_html=True)
    
    # Input
    user_input = st.chat_input("Share your thoughts...")
    
    # Process chat input and generate response in one go
    if user_input:
        # Add user message
        st.session_state.chat_history.append({
            'type': 'user',
            'content': user_input
        })

        # Display a temporary "thinking..." message while generating the response
        with chat_container:
             st.markdown(f"<div class='user-message'>{user_input}</div>", unsafe_allow_html=True)
             with st.spinner('thinking...'):
                time.sleep(0.5) # Brief pause for realism

        # Generate response
        st.session_state.personality.process_input(user_input)
        response = st.session_state.personality.generate_response(user_input)
        
        st.session_state.chat_history.append({
            'type': 'ai',
            'content': response
        })
        # Rerun once to display the new messages
        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    # World View
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("### 🌍 World View")
    
    # Create visualization
    world_container = st.container() # This container is just for the grid now
    
    # Canvas
    with world_container:
        st.markdown("<div class='world-container'>", unsafe_allow_html=True)
        
        # Create grid
        grid_size = 20
        grid_width = 30
        grid = [['·' for _ in range(grid_width)] for _ in range(grid_size)]
        
        # Plot agent
        agent_y = int(st.session_state.agent.position[1] / 100 * (grid_size - 1))
        agent_x = int(st.session_state.agent.position[0] / 100 * (grid_width - 1))
        
        # Plot target
        target_y = int(st.session_state.agent.target_position[1] / 100 * (grid_size - 1))
        target_x = int(st.session_state.agent.target_position[0] / 100 * (grid_width - 1))
        
        # Place markers
        grid[target_y][target_x] = '★'
        grid[agent_y][agent_x] = '●'
        
        # Render
        grid_str = '\n'.join('  '.join(row) for row in grid)
        st.code(grid_str, language=None)
        
        # Inject custom style for the code block
        st.markdown("<style>.stCodeBlock { background: transparent !important; }</style>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        
    # Status cards (moved outside the world_container)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown(f"""
        <div class='status-card'>
            <div class='status-label'>Emotional State</div>
            <div class='status-value'>{st.session_state.personality.get_emotional_summary()}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='status-card'>
            <div class='status-label'>Energy</div>
            <div class='status-value'>{st.session_state.agent.energy:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_b:
        st.markdown(f"""
        <div class='status-card'>
            <div class='status-label'>Learning Rate</div>
            <div class='status-value'>{st.session_state.brain.epsilon:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='status-card'>
            <div class='status-label'>Targets Found</div>
            <div class='status-value'>{st.session_state.agent.targets_reached}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Controls
    st.markdown("### ⚙️ Controls")
    
    # Autonomous mode toggle
    st.session_state.autonomous_mode = st.toggle(
        "Autonomous Learning", 
        value=st.session_state.autonomous_mode,
        help="When active, ALIVE will continuously explore its world to learn."
    )

    # New target button
    if st.button("Give New Target", use_container_width=True):
        st.session_state.agent.target_position = np.random.rand(2) * 100
        st.session_state.agent.is_seeking = True
        st.session_state.personality.emotional_state['curiosity'] = min(1.0, 
            st.session_state.personality.emotional_state['curiosity'] + 0.2)
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# MAIN SIMULATION LOOP
# ==========================================
if st.session_state.autonomous_mode:
    # Check if enough time has passed since the last update
    current_time = time.time()
    if current_time - st.session_state.last_update > 0.1: # 10 FPS
        loss, td_error, done = simulation_step()
        
        if done:
            # If target is reached, pause briefly and set a new one
            time.sleep(1)
            st.session_state.agent.target_position = np.random.rand(2) * 100
            st.session_state.agent.is_seeking = True

        st.session_state.last_update = current_time
        st.rerun()
