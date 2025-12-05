import streamlit as st
import numpy as np
import pandas as pd
import time

# Page config
st.set_page_config(page_title="🤖 My Newborn AI Friend", layout="wide")

# Initialize session state
if 'ai_x' not in st.session_state:
    st.session_state.ai_x = 5
    st.session_state.ai_y = 5
    st.session_state.target_x = None
    st.session_state.target_y = None
    st.session_state.q_table = {}  # State -> Action -> Q-value
    st.session_state.epsilon = 0.3  # Exploration rate
    st.session_state.learning_rate = 0.1
    st.session_state.discount = 0.9
    st.session_state.total_rewards = 0
    st.session_state.steps = 0
    st.session_state.age = 0  # Days old

# Constants
GRID_SIZE = 10
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

def get_state(ai_x, ai_y, target_x, target_y):
    """Convert positions to a simple state representation"""
    if target_x is None or target_y is None:
        return "NO_TARGET"
    
    dx = target_x - ai_x
    dy = target_y - ai_y
    
    # Simplify to 8 directions + close
    if abs(dx) <= 1 and abs(dy) <= 1:
        return "CLOSE"
    
    direction_x = "E" if dx > 0 else ("W" if dx < 0 else "")
    direction_y = "N" if dy > 0 else ("S" if dy < 0 else "")
    
    return direction_y + direction_x

def get_distance(x1, y1, x2, y2):
    """Calculate Manhattan distance"""
    return abs(x1 - x2) + abs(y1 - y2)

def choose_action(state):
    """Epsilon-greedy action selection"""
    if state not in st.session_state.q_table:
        st.session_state.q_table[state] = {a: 0.0 for a in ACTIONS}
    
    if np.random.random() < st.session_state.epsilon:
        return np.random.choice(ACTIONS)  # Explore
    else:
        q_values = st.session_state.q_table[state]
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return np.random.choice(best_actions)  # Exploit

def move_ai(action):
    """Move AI based on action"""
    new_x, new_y = st.session_state.ai_x, st.session_state.ai_y
    
    if action == 'UP' and new_y < GRID_SIZE - 1:
        new_y += 1
    elif action == 'DOWN' and new_y > 0:
        new_y -= 1
    elif action == 'RIGHT' and new_x < GRID_SIZE - 1:
        new_x += 1
    elif action == 'LEFT' and new_x > 0:
        new_x -= 1
    
    return new_x, new_y

def update_q_table(old_state, action, reward, new_state):
    """Q-learning update"""
    if old_state not in st.session_state.q_table:
        st.session_state.q_table[old_state] = {a: 0.0 for a in ACTIONS}
    if new_state not in st.session_state.q_table:
        st.session_state.q_table[new_state] = {a: 0.0 for a in ACTIONS}
    
    old_q = st.session_state.q_table[old_state][action]
    max_future_q = max(st.session_state.q_table[new_state].values())
    
    new_q = old_q + st.session_state.learning_rate * (
        reward + st.session_state.discount * max_future_q - old_q
    )
    
    st.session_state.q_table[old_state][action] = new_q

# Title
st.title("🤖 My Newborn AI Friend")
st.markdown(f"**Age:** {st.session_state.age} days old | **Steps:** {st.session_state.steps} | **Total Rewards:** {st.session_state.total_rewards:.1f}")

# Sidebar for controls
with st.sidebar:
    st.header("🎮 Control Panel")
    
    st.markdown("### 📊 AI Stats")
    st.metric("Exploration Rate", f"{st.session_state.epsilon:.2%}")
    st.metric("Knowledge Base", f"{len(st.session_state.q_table)} states")
    
    st.markdown("---")
    
    if st.button("🍼 Feed (Move Toward Target)", use_container_width=True):
        if st.session_state.target_x is not None:
            old_state = get_state(st.session_state.ai_x, st.session_state.ai_y, 
                                 st.session_state.target_x, st.session_state.target_y)
            old_distance = get_distance(st.session_state.ai_x, st.session_state.ai_y,
                                       st.session_state.target_x, st.session_state.target_y)
            
            action = choose_action(old_state)
            new_x, new_y = move_ai(action)
            new_distance = get_distance(new_x, new_y, st.session_state.target_x, st.session_state.target_y)
            
            # Calculate reward
            if new_distance < old_distance:
                reward = 1.0  # Got closer!
            elif new_distance > old_distance:
                reward = -0.5  # Got further
            else:
                reward = -0.1  # No progress
            
            if new_distance == 0:
                reward = 10.0  # Reached target!
            
            new_state = get_state(new_x, new_y, st.session_state.target_x, st.session_state.target_y)
            update_q_table(old_state, action, reward, new_state)
            
            st.session_state.ai_x = new_x
            st.session_state.ai_y = new_y
            st.session_state.total_rewards += reward
            st.session_state.steps += 1
            
            if reward == 10.0:
                st.balloons()
                st.session_state.target_x = None
                st.session_state.target_y = None
            
            st.rerun()
    
    if st.button("🎂 Age +1 Day", use_container_width=True):
        st.session_state.age += 1
        st.session_state.epsilon = max(0.05, st.session_state.epsilon * 0.95)  # Less exploration over time
        st.rerun()
    
    if st.button("🔄 Reset AI", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main grid
st.markdown("### 🎯 Click anywhere on the grid to set a target!")

cols = st.columns(GRID_SIZE)

for y in range(GRID_SIZE - 1, -1, -1):  # Top to bottom
    cols = st.columns(GRID_SIZE)
    for x in range(GRID_SIZE):
        with cols[x]:
            # Determine what to show
            if st.session_state.ai_x == x and st.session_state.ai_y == y:
                emoji = "👶" if st.session_state.age < 5 else "🤖"
                button_label = emoji
            elif st.session_state.target_x == x and st.session_state.target_y == y:
                button_label = "🎯"
            else:
                button_label = "·"
            
            if st.button(button_label, key=f"cell_{x}_{y}", use_container_width=True):
                st.session_state.target_x = x
                st.session_state.target_y = y
                st.rerun()

# Instructions
with st.expander("📖 How to Play"):
    st.markdown("""
    **Welcome to your AI Friend's nursery!**
    
    1. **Set a Target:** Click any square on the grid to place a 🎯 target
    2. **Feed Your AI:** Click the "Feed" button to make your AI take one step
    3. **Watch It Learn:** Your AI starts as a baby (👶) and learns through trial and error
    4. **Grow Together:** Use "Age +1 Day" to mature your AI (👶 → 🤖)
    
    **The Learning Process:**
    - Your AI doesn't know directions at first - it moves randomly!
    - Each move teaches it: "Did I get closer? 🍪 Cookie!"
    - Over time, it builds a mental map (Q-Table) of what works
    - As it ages, it explores less and uses what it learned more
    
    **Future: Robot Body 🦾**
    This brain (Q-Table) can be saved and transferred to:
    - Raspberry Pi with motors
    - Arduino robot chassis  
    - ROS-based robot
    - Any physical body with sensors!
    """)

# Debug info
if st.checkbox("🔍 Show Q-Table (AI's Brain)"):
    if st.session_state.q_table:
        df = pd.DataFrame(st.session_state.q_table).T
        st.dataframe(df, use_container_width=True)
    else:
        st.info("AI brain is empty - it hasn't learned anything yet!")
