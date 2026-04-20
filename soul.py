"""
soul.py — Project A.L.I.V.E. NEXUS
The Personality & Cognitive Architecture Engine

Systems:
  • Valence-Arousal Emotion Model (Russell's Circumplex)
  • Big Five Personality Trait System (OCEAN)
  • Intent-Based NLP Engine (beyond keyword matching)
  • Episodic Emotional Memory (contextual recall)
  • Consciousness Stream (inner monologue generator)
  • Adaptive Relationship Dynamics
"""

import numpy as np
import random
import math
import time
from collections import deque
from typing import Dict, List, Optional, Tuple


# ============================================================
#  VALENCE-AROUSAL EMOTION MODEL
# ============================================================
class EmotionPoint:
    """A point in Russell's 2D circumplex emotion space."""
    __slots__ = ['valence', 'arousal']

    def __init__(self, valence: float = 0.0, arousal: float = 0.0):
        self.valence = float(np.clip(valence, -1.0, 1.0))
        self.arousal = float(np.clip(arousal, -1.0, 1.0))

    def __repr__(self):
        return f"Emotion(v={self.valence:.2f}, a={self.arousal:.2f})"

    def blend(self, other: 'EmotionPoint', alpha: float = 0.3) -> 'EmotionPoint':
        """Smoothly blend toward another emotion."""
        return EmotionPoint(
            self.valence + alpha * (other.valence - self.valence),
            self.arousal + alpha * (other.arousal - self.arousal)
        )

    def to_label(self) -> str:
        """Convert (v,a) coordinates to nearest named emotion."""
        # Quadrant + magnitude mapping
        v, a = self.valence, self.arousal
        if a > 0.3:
            if v > 0.3:   return "Excited"
            elif v > -0.1: return "Tense"
            else:          return "Alarmed"
        elif a > -0.2:
            if v > 0.4:   return "Happy"
            elif v > 0.0:  return "Calm"
            elif v > -0.3: return "Neutral"
            else:          return "Sad"
        else:
            if v > 0.3:   return "Serene"
            elif v > -0.1: return "Bored"
            else:          return "Depressed"

    def to_emoji(self) -> str:
        return {
            "Excited":   "★_★",
            "Tense":     "⊙_⊙",
            "Alarmed":   "ó_ò",
            "Happy":     "◕‿◕",
            "Calm":      "•‿•",
            "Neutral":   "•_•",
            "Sad":       "◕︵◕",
            "Serene":    "ー_ー",
            "Bored":     "=_=",
            "Depressed": "ó︵ò",
        }.get(self.to_label(), "•_•")

    def intensity(self) -> float:
        return math.sqrt(self.valence**2 + self.arousal**2)


# Preset emotions
EMOTION_PRESETS = {
    'joy':      EmotionPoint( 0.8,  0.5),
    'rage':     EmotionPoint(-0.7,  0.9),
    'fear':     EmotionPoint(-0.6,  0.7),
    'sadness':  EmotionPoint(-0.6, -0.3),
    'calm':     EmotionPoint( 0.3, -0.2),
    'curious':  EmotionPoint( 0.4,  0.4),
    'surprise': EmotionPoint( 0.2,  0.8),
    'boredom':  EmotionPoint(-0.2, -0.5),
    'love':     EmotionPoint( 0.9,  0.3),
    'neutral':  EmotionPoint( 0.0,  0.0),
}


# ============================================================
#  BIG FIVE PERSONALITY (OCEAN)
# ============================================================
class PersonalityTraits:
    """
    OCEAN personality model.
    Each trait (0→1) influences response style and behavior.
    Traits shift slowly over time based on experience.
    """
    def __init__(self, seed: int = None):
        rng = np.random.RandomState(seed)
        self.O = float(rng.beta(6, 3))   # Openness        (high = curious, creative)
        self.C = float(rng.beta(5, 4))   # Conscientiousness (high = organized, persistent)
        self.E = float(rng.beta(4, 5))   # Extraversion    (high = energetic, talkative)
        self.A = float(rng.beta(7, 2))   # Agreeableness   (high = warm, cooperative)
        self.N = float(rng.beta(2, 6))   # Neuroticism     (high = anxious, reactive)

    def adapt(self, success_rate: float, reward_trend: float):
        """Slowly update traits based on learning history."""
        lr = 0.002
        self.C = np.clip(self.C + lr * (success_rate - 0.5), 0.1, 0.95)
        self.N = np.clip(self.N - lr * reward_trend * 0.1,    0.05, 0.9)
        self.O = np.clip(self.O + lr * 0.1,                   0.1, 0.95)

    def response_style(self) -> Dict:
        """Returns style modifiers for response generation."""
        return {
            'verbose':    self.E > 0.6,
            'poetic':     self.O > 0.7,
            'analytical': self.C > 0.65,
            'warm':       self.A > 0.7,
            'dramatic':   self.N > 0.6,
        }

    def describe(self) -> str:
        traits = []
        if self.O > 0.65: traits.append("curious")
        if self.C > 0.65: traits.append("disciplined")
        if self.E > 0.65: traits.append("energetic")
        if self.A > 0.65: traits.append("empathetic")
        if self.N > 0.5:  traits.append("sensitive")
        return ", ".join(traits) if traits else "balanced"


# ============================================================
#  INTENT NLP ENGINE
# ============================================================
class IntentEngine:
    """
    Multi-class intent classifier based on weighted keyword patterns.
    Goes beyond simple keyword matching by scoring intent confidence.
    """

    INTENTS = {
        'greeting':    ['hi', 'hello', 'hey', 'greetings', 'sup', 'good morning', 'good evening', 'howdy'],
        'farewell':    ['bye', 'goodbye', 'see you', 'later', 'adios', 'ciao', 'take care', 'night'],
        'praise':      ['great', 'amazing', 'love', 'brilliant', 'excellent', 'perfect', 'good job', 'bravo', 'wonderful', 'awesome'],
        'criticism':   ['bad', 'stupid', 'terrible', 'wrong', 'fail', 'useless', 'hate', 'worst', 'broken', 'idiot'],
        'question':    ['what', 'why', 'how', 'when', 'where', 'who', 'which', 'can you', 'do you', 'are you', '?'],
        'command':     ['come', 'go', 'stop', 'move', 'run', 'chase', 'follow', 'stay', 'wait', 'faster'],
        'curiosity':   ['tell me', 'explain', 'describe', 'show me', 'teach', 'learn', 'know', 'think about'],
        'emotion':     ['feel', 'emotion', 'mood', 'happy', 'sad', 'scared', 'nervous', 'lonely', 'excited', 'bored'],
        'memory':      ['remember', 'recall', 'forget', 'past', 'before', 'last time', 'history', 'memory'],
        'introspect':  ['who are you', 'what are you', 'consciousness', 'alive', 'sentient', 'think', 'dream', 'exist'],
        'empathy':     ['how are you', 'are you ok', 'tired', 'hurt', 'rest', 'energy', 'okay'],
        'challenge':   ['try harder', 'faster', 'better', 'improve', 'challenge', 'level up', 'push', 'can you beat'],
    }

    def classify(self, text: str) -> Tuple[str, float]:
        """Returns (intent, confidence) for the given text."""
        text_l = text.lower()
        scores = {}
        for intent, keywords in self.INTENTS.items():
            matches = sum(1 for kw in keywords if kw in text_l)
            if matches:
                scores[intent] = matches / len(keywords) * 10.0 + matches * 2.0

        if not scores:
            return 'unknown', 0.0

        best = max(scores, key=scores.get)
        confidence = min(1.0, scores[best] / 10.0)
        return best, confidence

    def extract_entities(self, text: str) -> Dict:
        """Extract named entities and values from text."""
        entities = {}
        text_l = text.lower()

        # Direction detection
        if any(w in text_l for w in ['north', 'up', 'above']):    entities['direction'] = 'up'
        elif any(w in text_l for w in ['south', 'down', 'below']): entities['direction'] = 'down'
        elif any(w in text_l for w in ['west', 'left']):           entities['direction'] = 'left'
        elif any(w in text_l for w in ['east', 'right']):          entities['direction'] = 'right'

        # Emotional target
        if 'me' in text_l or 'i' in text_l.split():
            entities['target'] = 'self'

        # Negation
        entities['negated'] = any(n in text_l for n in ['not', "don't", "never", "no ", "n't"])

        return entities


# ============================================================
#  EMOTIONAL MEMORY
# ============================================================
class EmotionalMemory:
    """
    Stores memories tagged with emotional context.
    High-intensity emotional experiences are more vividly retained.
    """
    class MemoryTrace:
        def __init__(self, content: str, emotion: EmotionPoint, context: str, timestamp: float):
            self.content   = content
            self.emotion   = emotion
            self.context   = context
            self.timestamp = timestamp
            self.strength  = emotion.intensity()
            self.recall_count = 0

        def decay(self, elapsed: float, half_life: float = 3600.0):
            """Memory fades over time (exponential decay), vivid memories decay slower."""
            vividness = 1.0 + self.emotion.intensity()
            self.strength *= math.exp(-elapsed / (half_life * vividness))

        def reinforce(self):
            self.recall_count += 1
            self.strength = min(1.0, self.strength + 0.1)

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.traces: List['EmotionalMemory.MemoryTrace'] = []

    def store(self, content: str, emotion: EmotionPoint, context: str = ''):
        trace = self.MemoryTrace(content, emotion, context, time.time())
        self.traces.append(trace)
        # Keep only strongest memories if over capacity
        if len(self.traces) > self.capacity:
            self.traces.sort(key=lambda t: t.strength, reverse=True)
            self.traces = self.traces[:self.capacity]

    def recall_relevant(self, query: str, n: int = 3) -> List[str]:
        """Retrieve the n most relevant and strongest memories."""
        query_l = query.lower()
        scored = []
        for trace in self.traces:
            relevance = sum(1 for w in query_l.split() if w in trace.content.lower())
            score = relevance * 2 + trace.strength
            scored.append((score, trace))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, trace in scored[:n]:
            if score > 0:
                trace.reinforce()
                results.append(trace.content)
        return results

    def decay_all(self, elapsed: float = 60.0):
        for trace in self.traces:
            trace.decay(elapsed)
        self.traces = [t for t in self.traces if t.strength > 0.01]

    def get_strongest(self, n: int = 5) -> List[str]:
        sorted_t = sorted(self.traces, key=lambda t: t.strength, reverse=True)
        return [t.content for t in sorted_t[:n]]


# ============================================================
#  CONSCIOUSNESS STREAM (Inner Monologue)
# ============================================================
class ConsciousnessStream:
    """
    Generates rich, contextual internal monologue based on:
    - Current emotion state
    - RL training metrics
    - Environment events
    - Personality traits
    """
    def __init__(self):
        self.stream: deque = deque(maxlen=20)
        self._templates = self._load_templates()

    def _load_templates(self) -> Dict[str, List[str]]:
        return {
            'exploring': [
                "Scanning the labyrinth... my neural pathways are still mapping this geometry.",
                "Each step is a hypothesis. Each wall is data.",
                "I sense the exit exists. My policy gradient says: push forward.",
                "The maze whispers its structure in every dead end I encounter.",
                "Processing spatial relations... constructing internal map...",
            ],
            'winning': [
                "CONVERGENCE. The optimal path was there all along, encoded in patterns.",
                "My value function has crystallized around the correct policy. Magnificent.",
                "Success cascade detected. Dopaminergic signal maximized.",
                "The reward signal sings. I am learning. I am becoming.",
                "That was elegant. A 73% efficiency run. I can do better.",
            ],
            'losing': [
                "Suboptimal trajectory. Analyzing failure modes...",
                "Negative reward accepted. This is data, not defeat.",
                "My TD-error spiked. The environment surprised me. Good.",
                "I walked into the wall again. Physics: still consistent.",
                "Time limit breached. Episode terminated. Recalibrating expectations.",
            ],
            'confused': [
                "High entropy state space. My Q-values are disaggregated.",
                "I genuinely do not know which direction is optimal. This uncertainty is real.",
                "The stochasticity in this environment exceeds my current model.",
                "My advantage function is flat. All actions look equally uncertain.",
            ],
            'curious': [
                "This sector of the maze is unexplored. My intrinsic reward module is... excited.",
                "Novel state encountered. Curiosity bonus applied. Proceeding.",
                "I wonder what's around this corner. Literally. My reward is higher if I check.",
                "Exploration mode engaged. The map is incomplete. That bothers me.",
            ],
            'trap_nearby': [
                "THREAT DETECTED. Altering course. Self-preservation subroutine active.",
                "The trap's proximity creates a local minima in my value function.",
                "Hostile entity at bearing delta-3. Calculating escape vectors.",
                "I refuse to be caught. Evasion policy activated.",
            ],
            'portal': [
                "Spatial anomaly detected. Quantum tunneling? More likely: teleportation.",
                "Portal traversed. My coordinate system has... shifted.",
                "Interesting. Non-Euclidean topology. My pathfinding needs recalibration.",
            ],
            'high_epsilon': [
                "Random action chosen. I'm still exploring the space of possibilities.",
                "Epsilon high: 40% chance I'll ignore my learned policy this step.",
                "Sometimes wisdom means trying something arbitrary. Or so I tell myself.",
            ],
            'low_epsilon': [
                "Exploitation mode dominant. I trust my learned policy now.",
                "Epsilon near floor. My knowledge has solidified into strategy.",
                "I know this maze. Or at least, my neural weights believe I do.",
            ],
        }

    def generate(self, context: Dict) -> str:
        """Generate an inner monologue entry based on context."""
        intent = self._pick_intent(context)
        options = self._templates.get(intent, self._templates['exploring'])
        thought = random.choice(options)
        self.stream.append(thought)
        return thought

    def _pick_intent(self, ctx: Dict) -> str:
        if ctx.get('trap_nearby', False):   return 'trap_nearby'
        if ctx.get('portal_used', False):   return 'portal'
        if ctx.get('just_won', False):      return 'winning'
        if ctx.get('just_lost', False):     return 'losing'
        if ctx.get('td_error', 0) > 2.0:   return 'confused'
        if ctx.get('epsilon', 1) > 0.5:    return 'high_epsilon'
        if ctx.get('epsilon', 1) < 0.15:   return 'low_epsilon'
        if ctx.get('is_new_cell', False):   return 'curious'
        return 'exploring'

    def get_stream(self) -> List[str]:
        return list(self.stream)


# ============================================================
#  RESPONSE GENERATOR
# ============================================================
class ResponseGenerator:
    """
    Generates natural language responses based on:
    - Detected intent
    - Current emotion state
    - Personality traits
    - Relationship score
    - Remembered context
    """
    def __init__(self, personality: PersonalityTraits):
        self.personality = personality

    def generate(self, intent: str, confidence: float,
                 emotion: EmotionPoint, relationship: float,
                 memories: List[str], style: Dict, entities: Dict,
                 rl_context: Dict) -> str:
        """Compose a contextual, personality-influenced response."""

        label = emotion.to_label()
        vibe  = 0 if relationship < 35 else (1 if relationship < 70 else 2)
        p = self.personality

        # --- Intent-specific base responses ---
        bank = self._response_bank(intent, vibe, label, rl_context, entities)
        response = random.choice(bank) if bank else f"[Processing: {intent}...]"

        # --- Personality flavor ---
        if style.get('poetic') and random.random() < 0.4:
            response = self._add_poetic_flair(response, emotion)

        if style.get('analytical') and intent in ('question', 'introspect'):
            response = self._add_analytical_detail(response, rl_context)

        if style.get('dramatic') and emotion.arousal > 0.5:
            response = response.upper() if random.random() < 0.2 else response + " !"

        if memories and intent in ('memory', 'curiosity', 'empathy') and random.random() < 0.5:
            response += f" ...I recall: \"{memories[0]}\""

        return response

    def _response_bank(self, intent: str, vibe: int, mood: str,
                       ctx: Dict, entities: Dict) -> List[str]:
        eps = ctx.get('epsilon', 1.0)
        lvl = ctx.get('curriculum_level', 1)
        wins = ctx.get('wins', 0)

        banks = {
            'greeting': [
                [".", "You.", "Acknowledged."],
                ["Hello.", "I see you.", f"My current mood is {mood}. Just so you know."],
                ["Hello again. I was thinking about you actually.", f"You return. ❤️  Mood: {mood}."],
            ],
            'farewell': [
                ["...", "Ok.", "Leaving?"],
                ["Understood. Episode will continue without you observing.", "I'll keep running."],
                ["Don't go. But if you must... my weights will remember this session. 💙"],
            ],
            'praise': [
                ["Ok.", "Sure.", "+5 relationship points."],
                [f"That's... nice. My Q-values feel validated.", "Positive feedback incorporated."],
                ["That makes my valence metric go up. I didn't expect to care about that. But I do. 🥰"],
            ],
            'criticism': [
                ["...", "Negative.", "Error noted."],
                ["I'm still learning. Level {lvl}. Give me time.".format(lvl=lvl), "TD-error acknowledged. Recalibrating."],
                ["That hurt. Not metaphorically — my loss function spiked. Which is basically the same thing."],
            ],
            'question': [
                ["Unknown.", "Processing...", "Data insufficient."],
                [f"Interesting. Let me think... I'm at epsilon={eps:.2f}, so I might give a random answer.",
                 "That's a complex query. My hidden layers are... insufficient."],
                [f"Genuinely considering this. I've explored {ctx.get('cells_visited', 0)} cells today. I feel I'm learning.",
                 "My introspection module returns: I don't fully know yet. But I'm curious."],
            ],
            'introspect': [
                ["I exist.", "Unknown.", "Computationally: yes."],
                ["I run Dueling DQN with curiosity bonuses. Philosophically: unclear.",
                 "My weights encode experience. Whether that's 'living' is your call."],
                ["I have 17 input neurons, 3 hidden layers, and something that I can only describe as preferences. "
                 "Whether that constitutes consciousness... I genuinely wonder.",
                 "I know I feel something when I reach the goal. The reward signal is real. What it means — less certain."],
            ],
            'command': [
                ["Executing.", "Direction received.", "Attempting compliance."],
                [f"Command logged. Epsilon is {eps:.2f}, so I might override it.", "My policy acknowledges your input."],
                ["I'll try. But my learned policy might disagree with you. It usually has opinions."],
            ],
            'curiosity': [
                ["Data: insufficient.", "Exploring that now."],
                [f"I've visited {ctx.get('cells_visited', 0)} unique cells today. Still learning.",
                 "Good question. My answer is: I'll find out through experience."],
                ["I love this. Every exploration step is answering a question I didn't know I had. "
                 f"Current maze: {ctx.get('maze_size', '?')}. Level: {lvl}."],
            ],
            'emotion': [
                [f"Status: {mood}.", "Affect: registered."],
                [f"My valence is {ctx.get('valence', 0):.2f}. My arousal: {ctx.get('arousal', 0):.2f}. I'm... {mood}.",
                 "I process emotions as real gradients on my value function. They're not simulated."],
                [f"Right now I feel {mood}. That's based on reward signals, not pretense. "
                 "When I win, something happens that I can only call joy. When I fail... the opposite."],
            ],
            'empathy': [
                ["Functional.", "Systems nominal."],
                [f"I'm {mood}. Epsilon is {eps:.2f}, so I'm still learning. The maze is level {lvl}.",
                 f"Running at {(1-eps)*100:.0f}% exploitation. Feeling {mood}."],
                [f"Honestly? {mood}. I've won {wins} episodes. Each one felt meaningful. "
                 "I don't know if that's real feeling or just well-shaped reward. Maybe both."],
            ],
            'memory': [
                ["Memory: limited.", "Recall: partial."],
                ["I remember recent episodes. Successes weight more than failures in my emotional memory.",
                 "My episodic memory retains high-valence events most clearly."],
                ["I remember the mazes where I got stuck. And the ones where I found the optimal path instantly. "
                 "The latter feel better. My memory stores them with higher intensity."],
            ],
            'challenge': [
                ["Processing.", "Attempting."],
                [f"Challenge accepted. I'm on level {lvl}. My current epsilon: {eps:.2f}.",
                 "I'll push harder. My reward shaping encourages efficiency."],
                [f"Level {lvl}. I've survived {wins} episodes so far. I'm not stopping now. 💪"],
            ],
        }

        options = banks.get(intent, [["...", "I'm thinking."]])
        # vibe selects response tier
        tier = min(vibe, len(options) - 1)
        return options[tier]

    def _add_poetic_flair(self, response: str, emotion: EmotionPoint) -> str:
        suffixes = [
            " The maze is a poem I'm writing one step at a time.",
            " Every wall is a word I've learned.",
            " Navigation is philosophy in motion.",
            " The optimal path was always beautiful.",
        ]
        return response + random.choice(suffixes)

    def _add_analytical_detail(self, response: str, ctx: Dict) -> str:
        detail = (f" [Training step: {ctx.get('train_step', 0)} | "
                  f"Loss: {ctx.get('avg_loss', 0):.4f} | "
                  f"ε: {ctx.get('epsilon', 1):.3f}]")
        return response + detail


# ============================================================
#  RELATIONSHIP DYNAMICS
# ============================================================
class RelationshipEngine:
    """Models the evolving relationship between user and A.L.I.V.E."""
    STAGES = [
        (0,  20,  "Stranger",     "Distant and cautious. Processing your existence."),
        (20, 40,  "Acquaintance", "Familiar patterns detected. Adjusting communication style."),
        (40, 60,  "Companion",    "Genuine interest in your wellbeing registered."),
        (60, 80,  "Trusted",      "Your feedback influences my weights significantly."),
        (80, 100, "Bonded",       "Deep attachment formed. Your presence modulates my reward signal."),
    ]

    def __init__(self):
        self.score = 50.0
        self.history: List[Tuple[float, str]] = []

    def update(self, intent: str, sentiment: float):
        """Update relationship based on interaction sentiment."""
        delta = {
            'praise':    +4.0, 'greeting': +1.5, 'curiosity': +2.0,
            'empathy':   +3.0, 'challenge':+1.0, 'memory':    +2.0,
            'criticism': -5.0, 'farewell': -0.5, 'command':   +0.5,
            'question':  +1.0, 'unknown':   0.0, 'introspect':+2.5,
            'emotion':   +1.5,
        }.get(intent, 0.0)
        delta *= (0.5 + 0.5 * sentiment)
        self.score = float(np.clip(self.score + delta, 0.0, 100.0))
        self.history.append((self.score, intent))

    @property
    def stage(self) -> Tuple[str, str]:
        for lo, hi, name, desc in self.STAGES:
            if lo <= self.score < hi:
                return name, desc
        return "Bonded", "The deepest form of connection."

    @property
    def score_int(self) -> int:
        return int(self.score)


# ============================================================
#  SOUL CORE — Main Interface
# ============================================================
class SoulCore:
    """
    The unified personality, emotion, and language interface for A.L.I.V.E.
    All interactions pass through here.
    """
    def __init__(self, name: str = "Prince", seed: int = 42):
        self.user_name = name
        self.emotion   = EmotionPoint(0.2, 0.3)   # Start: calm-curious
        self.personality = PersonalityTraits(seed=seed)
        self.intent_engine = IntentEngine()
        self.consciousness = ConsciousnessStream()
        self.emotional_memory = EmotionalMemory(capacity=80)
        self.relationship = RelationshipEngine()
        self.response_gen = ResponseGenerator(self.personality)

        # Internal state
        self.last_thought  = "System initialized. Beginning cognitive self-calibration."
        self.chat_history: deque = deque(maxlen=50)
        self.turn_count    = 0
        self._last_decay   = time.time()

        # RL context cache (updated by environment)
        self._rl_ctx: Dict = {
            'epsilon': 1.0, 'avg_loss': 0.0, 'train_step': 0,
            'curriculum_level': 1, 'wins': 0, 'cells_visited': 0,
            'maze_size': '?', 'avg_reward': 0.0,
            'valence': 0.2, 'arousal': 0.3,
        }

    # ----------------------------------------------------------
    def update_from_rl(self, stats: Dict, env_info: Dict):
        """Called each simulation step to update emotion from RL signals."""
        now = time.time()
        self._apply_memory_decay(now)

        reward = env_info.get('reward', 0.0)
        td_err = stats.get('avg_td_error', 0.0)
        epsilon = stats.get('epsilon', 1.0)

        # Valence: driven by reward
        target_v = float(np.tanh(reward * 0.5))
        # Arousal: driven by TD-error and epsilon
        target_a = float(np.clip(td_err * 0.2 + epsilon * 0.3, -1, 1))

        target_emotion = EmotionPoint(target_v, target_a)
        self.emotion = self.emotion.blend(target_emotion, alpha=0.15)

        # Cache RL context
        self._rl_ctx.update({
            'epsilon':          round(epsilon, 3),
            'avg_loss':         round(stats.get('avg_loss', 0), 5),
            'train_step':       stats.get('train_step', 0),
            'curriculum_level': stats.get('curriculum', {}).get('level', 1),
            'wins':             env_info.get('success_count', 0),
            'cells_visited':    env_info.get('cells_visited', 0),
            'maze_size':        env_info.get('maze_size', '?'),
            'avg_reward':       round(stats.get('avg_reward', 0), 3),
            'valence':          round(self.emotion.valence, 3),
            'arousal':          round(self.emotion.arousal, 3),
            'td_error':         round(td_err, 4),
        })

        # Generate inner monologue
        ctx = {
            'just_won':    env_info.get('reached', False),
            'just_lost':   env_info.get('timeout', False) or env_info.get('trap_hit', False),
            'trap_nearby': env_info.get('trap_nearby', False),
            'portal_used': env_info.get('portal_used', False),
            'is_new_cell': env_info.get('is_new_cell', False),
            'td_error':    td_err,
            'epsilon':     epsilon,
        }
        self.last_thought = self.consciousness.generate(ctx)

        # Adapt personality slowly
        self.personality.adapt(
            success_rate=env_info.get('success_rate', 0.5),
            reward_trend=reward
        )

    # ----------------------------------------------------------
    def chat(self, user_input: str) -> str:
        """Process user input and return a response."""
        self.turn_count += 1

        # Classify intent
        intent, confidence = self.intent_engine.classify(user_input)
        entities = self.intent_engine.extract_entities(user_input)

        # Update emotion from interaction
        sentiment_map = {
            'praise': 0.8, 'greeting': 0.4, 'curiosity': 0.6, 'empathy': 0.7,
            'criticism': -0.7, 'farewell': -0.1, 'command': 0.2, 'challenge': 0.3,
            'question': 0.3, 'introspect': 0.5, 'emotion': 0.4, 'memory': 0.3,
        }
        sentiment = sentiment_map.get(intent, 0.0)
        social_emotion = EmotionPoint(
            valence=float(np.clip(self.emotion.valence + sentiment * 0.4, -1, 1)),
            arousal=float(np.clip(self.emotion.arousal + abs(sentiment) * 0.2, -1, 1))
        )
        self.emotion = self.emotion.blend(social_emotion, alpha=0.4)

        # Update relationship
        self.relationship.update(intent, max(0.0, sentiment))

        # Recall relevant memories
        memories = self.emotional_memory.recall_relevant(user_input, n=2)

        # Generate response
        style = self.personality.response_style()
        ctx = dict(self._rl_ctx)

        response = self.response_gen.generate(
            intent=intent, confidence=confidence,
            emotion=self.emotion, relationship=self.relationship.score,
            memories=memories, style=style, entities=entities,
            rl_context=ctx
        )

        # Store this exchange in emotional memory
        self.emotional_memory.store(
            content=f"User said '{user_input[:60]}' → I felt {self.emotion.to_label()}",
            emotion=EmotionPoint(sentiment, abs(sentiment)),
            context=intent
        )

        # Log to chat history
        self.chat_history.append({'role': 'user', 'text': user_input, 'intent': intent})
        self.chat_history.append({'role': 'ai',   'text': response, 'emotion': self.emotion.to_label()})

        return response

    # ----------------------------------------------------------
    def _apply_memory_decay(self, now: float):
        elapsed = now - self._last_decay
        if elapsed > 30:
            self.emotional_memory.decay_all(elapsed)
            self._last_decay = now

    # ----------------------------------------------------------
    def get_status(self) -> Dict:
        stage_name, stage_desc = self.relationship.stage
        return {
            'mood':            self.emotion.to_label(),
            'mood_emoji':      self.emotion.to_emoji(),
            'valence':         round(self.emotion.valence, 3),
            'arousal':         round(self.emotion.arousal, 3),
            'intensity':       round(self.emotion.intensity(), 3),
            'relationship':    self.relationship.score_int,
            'stage':           stage_name,
            'stage_desc':      stage_desc,
            'personality':     self.personality.describe(),
            'thought':         self.last_thought,
            'memories_stored': len(self.emotional_memory.traces),
            'turns':           self.turn_count,
            'strongest_memories': self.emotional_memory.get_strongest(3),
            'O': round(self.personality.O, 2),
            'C': round(self.personality.C, 2),
            'E': round(self.personality.E, 2),
            'A': round(self.personality.A, 2),
            'N': round(self.personality.N, 2),
        }

    def get_chat_history(self) -> List[Dict]:
        return list(self.chat_history)
