"""
brain.py — Project A.L.I.V.E. NEXUS
The Reinforcement Learning Engine

Architecture:
  ┌─────────────────────────────────────────┐
  │   Dueling Double DQN + N-Step Returns   │
  │   Prioritized Experience Replay (PER)   │
  │   Adam Optimizer + He Initialization    │
  │   Intrinsic Curiosity Module (ICM)      │
  │   Adaptive Curriculum Learning          │
  │   Soft Target Network Updates           │
  └─────────────────────────────────────────┘

Author: Project A.L.I.V.E. NEXUS
"""

import numpy as np
import random
import math
from collections import deque, namedtuple
from typing import Tuple, List, Optional, Dict, Any

# ============================================================
#  SEGMENT TREE — O(log N) priority sampling for PER
# ============================================================
class SegmentTree:
    """
    Binary Segment Tree supporting efficient range-sum queries.
    Used internally by the Prioritized Replay Buffer.
    """
    def __init__(self, capacity: int, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0, \
            "Capacity must be a power of 2."
        self.capacity = capacity
        self.operation = operation
        self.neutral_element = neutral_element
        self.tree = [neutral_element] * (2 * capacity)

    def _reduce(self, start: int, end: int, node: int, node_start: int, node_end: int):
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce(start, end, 2 * node, node_start, mid)
        elif start > mid:
            return self._reduce(start, end, 2 * node + 1, mid + 1, node_end)
        left = self._reduce(start, mid, 2 * node, node_start, mid)
        right = self._reduce(mid + 1, end, 2 * node + 1, mid + 1, node_end)
        return self.operation(left, right)

    def query(self, start: int, end: int):
        return self._reduce(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val):
        idx += self.capacity
        self.tree[idx] = val
        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int):
        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity: int):
        super().__init__(capacity, operator_add, 0.0)

    def sum(self, start: int = 0, end: int = None) -> float:
        end = self.capacity - 1 if end is None else end
        return self.query(start, end)

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        """Find the highest index with prefix sum <= prefixsum."""
        idx = 1
        while idx < self.capacity:
            if self.tree[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self.tree[2 * idx]
                idx = 2 * idx + 1
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity: int):
        super().__init__(capacity, min, float('inf'))

    def min(self, start: int = 0, end: int = None) -> float:
        end = self.capacity - 1 if end is None else end
        return self.query(start, end)


def operator_add(a, b):
    return a + b


# ============================================================
#  PRIORITIZED EXPERIENCE REPLAY BUFFER
# ============================================================
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (Schaul et al., 2015).
    Uses a Segment Tree for O(log N) priority-weighted sampling.
    Implements importance-sampling weights to correct bias.
    """
    def __init__(self, capacity: int, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_frames: int = 100_000):
        # Capacity must be power of 2
        cap = 1
        while cap < capacity:
            cap <<= 1
        self.capacity = cap
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0

        self.sum_tree = SumSegmentTree(cap)
        self.min_tree = MinSegmentTree(cap)
        self.max_priority = 1.0

        self.buffer: List[Optional[Experience]] = [None] * cap
        self.pos = 0
        self.size = 0

    @property
    def beta(self) -> float:
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def add(self, state, action, reward, next_state, done):
        self.buffer[self.pos] = Experience(state, action, reward, next_state, done)
        p = self.max_priority ** self.alpha
        self.sum_tree[self.pos] = p
        self.min_tree[self.pos] = p
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        self.frame += 1
        indices = []
        total = self.sum_tree.sum(0, self.size - 1)
        seg = total / batch_size

        for i in range(batch_size):
            a, b = seg * i, seg * (i + 1)
            s = random.uniform(a, b)
            idx = self.sum_tree.find_prefixsum_idx(s)
            idx = min(idx, self.size - 1)
            indices.append(idx)

        beta = self.beta
        min_prob = self.min_tree.min(0, self.size - 1) / total
        max_weight = (min_prob * self.size) ** (-beta)

        weights = []
        for idx in indices:
            prob = self.sum_tree[idx] / total
            weight = (prob * self.size) ** (-beta) / max_weight
            weights.append(weight)

        batch = [self.buffer[i] for i in indices]
        states      = np.array([e.state      for e in batch], dtype=np.float32)
        actions     = np.array([e.action     for e in batch], dtype=np.int32)
        rewards     = np.array([e.reward     for e in batch], dtype=np.float32)
        next_states = np.array([e.next_state for e in batch], dtype=np.float32)
        dones       = np.array([e.done       for e in batch], dtype=np.float32)

        return (states, actions, rewards, next_states, dones,
                np.array(indices, dtype=np.int32),
                np.array(weights, dtype=np.float32))

    def update_priorities(self, indices, priorities):
        for idx, p in zip(indices, priorities):
            p = max(p, 1e-6)
            self.max_priority = max(self.max_priority, p)
            self.sum_tree[idx] = p ** self.alpha
            self.min_tree[idx] = p ** self.alpha

    def __len__(self):
        return self.size


# ============================================================
#  NEURAL NETWORK — Dueling Architecture + Adam Optimizer
# ============================================================
class NeuralNet:
    """
    3-layer Dueling DQN network implemented in pure NumPy.
    - He initialization
    - Leaky ReLU activations
    - Dueling streams: V(s) + A(s,a) - mean(A)
    - Adam optimizer with per-parameter moment estimates
    - Gradient clipping for stability
    """
    PARAMS = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3',
              'W_val', 'b_val', 'W_adv', 'b_adv']

    def __init__(self, input_size: int, h1: int, h2: int, h3: int, output_size: int):
        s = input_size
        # He init
        self.W1  = np.random.randn(s,  h1) * np.sqrt(2.0 / s)
        self.b1  = np.zeros(h1)
        self.W2  = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2  = np.zeros(h2)
        self.W3  = np.random.randn(h2, h3) * np.sqrt(2.0 / h2)
        self.b3  = np.zeros(h3)
        # Dueling heads
        self.W_val = np.random.randn(h3, 1)           * np.sqrt(2.0 / h3)
        self.b_val = np.zeros(1)
        self.W_adv = np.random.randn(h3, output_size) * np.sqrt(2.0 / h3)
        self.b_adv = np.zeros(output_size)

        self.output_size = output_size
        self._init_adam()
        # Cache for backprop
        self._cache: Dict[str, Any] = {}

    def _init_adam(self):
        self._m = {p: np.zeros_like(getattr(self, p)) for p in self.PARAMS}
        self._v = {p: np.zeros_like(getattr(self, p)) for p in self.PARAMS}
        self._t = 0

    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def leaky_relu_grad(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, 1.0, alpha)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if x.ndim == 1:
            x = x[np.newaxis, :]

        z1 = x  @ self.W1 + self.b1;  a1 = self.leaky_relu(z1)
        z2 = a1 @ self.W2 + self.b2;  a2 = self.leaky_relu(z2)
        z3 = a2 @ self.W3 + self.b3;  a3 = self.leaky_relu(z3)

        val = a3 @ self.W_val + self.b_val          # (B, 1)
        adv = a3 @ self.W_adv + self.b_adv          # (B, A)
        q   = val + (adv - adv.mean(axis=1, keepdims=True))

        if training:
            self._cache = dict(x=x, z1=z1, a1=a1, z2=z2, a2=a2, z3=z3, a3=a3, val=val, adv=adv)
        return q

    def backward(self, dq: np.ndarray, lr: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, clip: float = 10.0):
        """Full backprop with Adam update and gradient clipping."""
        c = self._cache
        B = c['x'].shape[0]

        # Dueling gradients
        # dq/dval = 1 (broadcast over actions), mean-centered
        d_val = dq.sum(axis=1, keepdims=True)        # (B,1)
        d_adv = dq - dq.mean(axis=1, keepdims=True)  # (B,A)

        dW_val = c['a3'].T @ d_val / B
        db_val = d_val.mean(axis=0)
        dW_adv = c['a3'].T @ d_adv / B
        db_adv = d_adv.mean(axis=0)

        da3 = d_val @ self.W_val.T + d_adv @ self.W_adv.T
        dz3 = da3 * self.leaky_relu_grad(c['z3'])
        dW3 = c['a2'].T @ dz3 / B;  db3 = dz3.mean(axis=0)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * self.leaky_relu_grad(c['z2'])
        dW2 = c['a1'].T @ dz2 / B;  db2 = dz2.mean(axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.leaky_relu_grad(c['z1'])
        dW1 = c['x'].T @ dz1 / B;   db1 = dz1.mean(axis=0)

        grads = dict(W1=dW1, b1=db1, W2=dW2, b2=db2, W3=dW3, b3=db3,
                     W_val=dW_val, b_val=db_val, W_adv=dW_adv, b_adv=db_adv)

        self._t += 1
        for p, g in grads.items():
            g = np.clip(g, -clip, clip)
            self._m[p] = beta1 * self._m[p] + (1 - beta1) * g
            self._v[p] = beta2 * self._v[p] + (1 - beta2) * (g ** 2)
            m_hat = self._m[p] / (1 - beta1 ** self._t)
            v_hat = self._v[p] / (1 - beta2 ** self._t)
            setattr(self, p, getattr(self, p) - lr * m_hat / (np.sqrt(v_hat) + eps))

    def copy_from(self, other: 'NeuralNet'):
        for p in self.PARAMS:
            setattr(self, p, getattr(other, p).copy())

    def soft_update(self, online: 'NeuralNet', tau: float = 0.005):
        """Polyak averaging: θ_target ← τ·θ_online + (1−τ)·θ_target"""
        for p in self.PARAMS:
            setattr(self, p, tau * getattr(online, p) + (1 - tau) * getattr(self, p))

    def get_weights(self) -> Dict:
        return {p: getattr(self, p).tolist() for p in self.PARAMS}

    def set_weights(self, d: Dict):
        for p, v in d.items():
            if p in self.PARAMS:
                setattr(self, p, np.array(v, dtype=np.float64))


# ============================================================
#  N-STEP RETURN BUFFER
# ============================================================
class NStepBuffer:
    """
    Accumulates N-step returns for richer temporal credit assignment.
    G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^{n-1}·r_{t+n-1} + γ^n · Q(s_{t+n})
    """
    def __init__(self, n_steps: int = 3, gamma: float = 0.99):
        self.n = n_steps
        self.gamma = gamma
        self.buf = deque(maxlen=n_steps)

    def add(self, state, action, reward, next_state, done) -> Optional[Tuple]:
        self.buf.append((state, action, reward, next_state, done))
        if len(self.buf) < self.n:
            return None
        return self._compute()

    def _compute(self) -> Tuple:
        s0, a0 = self.buf[0][:2]
        G, last_ns, last_done = 0.0, None, False
        for i, (_, _, r, ns, d) in enumerate(self.buf):
            G += (self.gamma ** i) * r
            last_ns, last_done = ns, d
            if d:
                break
        return s0, a0, G, last_ns, last_done

    def flush(self) -> List[Tuple]:
        results = []
        while self.buf:
            results.append(self._compute())
            self.buf.popleft()
        return [r for r in results if r is not None]

    def clear(self):
        self.buf.clear()


# ============================================================
#  INTRINSIC CURIOSITY MODULE (Count-Based)
# ============================================================
class IntrinsicCuriosity:
    """
    Count-based exploration bonus: r_i(s) = β / √N(s)
    where N(s) is the visit count of state s.
    Encourages the agent to explore novel regions of state space.
    """
    def __init__(self, bins: int = 16, beta: float = 0.05):
        self.bins = bins
        self.beta = beta
        self.counts: Dict[tuple, int] = {}

    def _key(self, state: np.ndarray) -> tuple:
        return tuple((np.clip(state, 0, 1) * (self.bins - 1)).astype(int).tolist())

    def bonus(self, state: np.ndarray) -> float:
        k = self._key(state)
        self.counts[k] = self.counts.get(k, 0) + 1
        return self.beta / math.sqrt(self.counts[k])

    def visit_count(self, state: np.ndarray) -> int:
        return self.counts.get(self._key(state), 0)

    def coverage(self) -> float:
        """What fraction of the (discretized) state space has been visited?"""
        return len(self.counts)

    def heatmap(self, h: int, w: int) -> np.ndarray:
        grid = np.zeros((h, w), dtype=np.float32)
        for key, cnt in self.counts.items():
            if len(key) >= 2:
                r = min(int(key[0] * h / self.bins), h - 1)
                c = min(int(key[1] * w / self.bins), w - 1)
                grid[r, c] += cnt
        if grid.max() > 0:
            grid /= grid.max()
        return grid

    def reset(self):
        self.counts.clear()


# ============================================================
#  CURRICULUM LEARNING MANAGER
# ============================================================
class CurriculumManager:
    """
    Automatic Curriculum Learning (ACL).
    Monitors rolling performance and promotes/demotes difficulty
    levels to keep the agent in its Zone of Proximal Development (ZPD).

    Levels 1-10 scale maze size, algorithm complexity, and obstacle types.
    """
    LEVEL_CONFIGS = {
        1:  {'maze_h': 7,  'maze_w': 9,  'algorithm': 'backtracker', 'fog': False, 'dynamic': False, 'portals': False},
        2:  {'maze_h': 9,  'maze_w': 11, 'algorithm': 'backtracker', 'fog': False, 'dynamic': False, 'portals': False},
        3:  {'maze_h': 11, 'maze_w': 13, 'algorithm': 'prim',        'fog': False, 'dynamic': False, 'portals': False},
        4:  {'maze_h': 13, 'maze_w': 15, 'algorithm': 'prim',        'fog': True,  'dynamic': False, 'portals': False},
        5:  {'maze_h': 15, 'maze_w': 19, 'algorithm': 'wilson',      'fog': True,  'dynamic': False, 'portals': False},
        6:  {'maze_h': 17, 'maze_w': 21, 'algorithm': 'wilson',      'fog': True,  'dynamic': True,  'portals': False},
        7:  {'maze_h': 21, 'maze_w': 25, 'algorithm': 'backtracker', 'fog': True,  'dynamic': True,  'portals': False},
        8:  {'maze_h': 25, 'maze_w': 29, 'algorithm': 'prim',        'fog': True,  'dynamic': True,  'portals': True},
        9:  {'maze_h': 29, 'maze_w': 33, 'algorithm': 'wilson',      'fog': True,  'dynamic': True,  'portals': True},
        10: {'maze_h': 35, 'maze_w': 41, 'algorithm': 'hybrid',      'fog': True,  'dynamic': True,  'portals': True},
    }

    def __init__(self):
        self.level = 1
        self.max_level = 10
        self.window = deque(maxlen=20)
        self.promote_thresh = 0.72
        self.demote_thresh  = 0.25
        self.promotions = 0
        self.demotions  = 0
        self.history: List[Dict] = []

    def record(self, success: bool, steps: int, max_steps: int, reward: float):
        eff = max(0.0, 1.0 - steps / max(max_steps, 1)) if success else 0.0
        score = 0.5 * float(success) + 0.5 * eff
        self.window.append(score)
        self.history.append({'level': self.level, 'success': success,
                              'steps': steps, 'reward': reward, 'score': score})
        self._evaluate()

    def _evaluate(self):
        if len(self.window) < 8:
            return
        avg = float(np.mean(self.window))
        if avg >= self.promote_thresh and self.level < self.max_level:
            self.level += 1
            self.promotions += 1
            self.window.clear()
        elif avg <= self.demote_thresh and self.level > 1:
            self.level -= 1
            self.demotions += 1
            self.window.clear()

    @property
    def config(self) -> Dict:
        return dict(self.LEVEL_CONFIGS[self.level])

    @property
    def avg_score(self) -> float:
        return float(np.mean(self.window)) if self.window else 0.0

    @property
    def zpd_progress(self) -> float:
        """Progress toward promotion (0→1)."""
        if self.level == self.max_level:
            return 1.0
        return min(1.0, self.avg_score / self.promote_thresh)

    def get_stats(self) -> Dict:
        return {
            'level': self.level,
            'max_level': self.max_level,
            'avg_score': round(self.avg_score, 3),
            'promotions': self.promotions,
            'demotions': self.demotions,
            'zpd_progress': round(self.zpd_progress, 3),
            'config': self.config,
        }


# ============================================================
#  LEARNING RATE SCHEDULER (Plateau Detection)
# ============================================================
class LRScheduler:
    """Reduce LR on reward plateau. Adaptive meta-learning."""
    def __init__(self, base_lr: float = 0.001, factor: float = 0.5,
                 patience: int = 100, min_lr: float = 1e-5):
        self.lr = base_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self._best = -np.inf
        self._wait = 0
        self.reductions = 0

    def step(self, metric: float) -> float:
        if metric > self._best + 1e-4:
            self._best = metric
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self.lr = max(self.min_lr, self.lr * self.factor)
                self._wait = 0
                self.reductions += 1
        return self.lr


# ============================================================
#  AGENT BRAIN — Full RL Controller
# ============================================================
class AgentBrain:
    """
    Top-level RL agent.
    Combines: Dueling DDQN + PER + N-Step + ICM + Curriculum + Adam + Soft-Update
    """
    def __init__(self, state_size: int, action_size: int = 4, config: Dict = None):
        cfg = config or {}
        self.state_size  = state_size
        self.action_size = action_size

        h1 = cfg.get('h1', 256)
        h2 = cfg.get('h2', 128)
        h3 = cfg.get('h3', 64)

        self.online_net = NeuralNet(state_size, h1, h2, h3, action_size)
        self.target_net = NeuralNet(state_size, h1, h2, h3, action_size)
        self.target_net.copy_from(self.online_net)

        self.memory    = PrioritizedReplayBuffer(cfg.get('buffer_size', 50_000))
        self.n_step    = NStepBuffer(cfg.get('n_steps', 3), cfg.get('gamma', 0.99))
        self.curiosity = IntrinsicCuriosity(beta=cfg.get('icm_beta', 0.05))
        self.curriculum = CurriculumManager()
        self.lr_sched  = LRScheduler(base_lr=cfg.get('lr', 0.001))

        # Hyper-parameters
        self.gamma           = cfg.get('gamma', 0.99)
        self.epsilon         = 1.0
        self.epsilon_min     = cfg.get('epsilon_min', 0.05)
        self.epsilon_decay   = cfg.get('epsilon_decay', 0.9995)
        self.learning_rate   = cfg.get('lr', 0.001)
        self.batch_size      = cfg.get('batch_size', 64)
        self.tau             = cfg.get('tau', 0.005)
        self.planning_steps  = cfg.get('planning_steps', 5)  # Dyna-Q training multiplier

        # Counters & stats
        self.train_step     = 0
        self.total_eps      = 0
        self.episode_reward = 0.0
        self.recent_losses  = deque(maxlen=200)
        self.recent_rewards = deque(maxlen=200)
        self.recent_td_errors = deque(maxlen=200)

    # ----------------------------------------------------------
    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        q = self.online_net.forward(state, training=False)
        return int(np.argmax(q[0]))

    # ----------------------------------------------------------
    def step(self, state, action, reward, next_state, done):
        """Process one environment transition."""
        # Add intrinsic curiosity bonus
        intrinsic = self.curiosity.bonus(state)
        augmented_reward = reward + intrinsic
        self.episode_reward += reward

        # Push to n-step buffer; may return an n-step experience
        exp = self.n_step.add(state, action, augmented_reward, next_state, done)
        if exp:
            self.memory.add(*exp)

        if done:
            for exp in self.n_step.flush():
                self.memory.add(*exp)
            self.n_step.clear()

        # Train if buffer is ready
        loss, td_err = 0.0, 0.0
        if len(self.memory) >= self.batch_size:
            # Dyna-Q Hallucination / Accelerated Planning
            # Train Multiple batches from memory per real environmental physical step.
            actual_planning_steps = self.planning_steps
            if done and self.episode_reward > 10.0:
                actual_planning_steps *= 4  # Core breakthrough learning (Super Brain Mode)
                
            for _ in range(actual_planning_steps):
                l, t = self._train()
                loss += l
                td_err += t
            
            loss /= actual_planning_steps
            td_err /= actual_planning_steps
            self.train_step += 1

            # Soft-update target network every step
            self.target_net.soft_update(self.online_net, tau=self.tau)

        if done:
            self.total_eps += 1
            self.recent_rewards.append(self.episode_reward)
            lr = self.lr_sched.step(self.avg_reward)
            self.learning_rate = lr
            self.episode_reward = 0.0
            
            # FIXED BUG: Decay epsilon only ONCE per episode (Match MazE.py physics)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.recent_losses.append(loss)
        self.recent_td_errors.append(td_err)
        return loss, td_err

    # ----------------------------------------------------------
    def _train(self) -> Tuple[float, float]:
        states, actions, rewards, next_states, dones, indices, weights = \
            self.memory.sample(self.batch_size)

        # --- Double DQN target computation ---
        q_online_next = self.online_net.forward(next_states, training=False)
        best_actions  = np.argmax(q_online_next, axis=1)                    # Online selects
        q_target_next = self.target_net.forward(next_states, training=False) # Target evaluates

        targets = rewards + self.gamma * (1 - dones) * \
                  q_target_next[np.arange(self.batch_size), best_actions]

        # --- Current Q values ---
        q_pred = self.online_net.forward(states, training=True)

        # --- TD errors (for PER priority update) ---
        td_errors = np.abs(targets - q_pred[np.arange(self.batch_size), actions])

        # --- Loss gradient ---
        dq = np.zeros_like(q_pred)
        delta = (targets - q_pred[np.arange(self.batch_size), actions]) * weights
        dq[np.arange(self.batch_size), actions] = delta

        # --- Backprop ---
        self.online_net.backward(dq, lr=self.learning_rate)

        # --- Update priorities ---
        self.memory.update_priorities(indices, td_errors.tolist())

        loss = float(np.mean((targets - q_pred[np.arange(self.batch_size), actions]) ** 2))
        return loss, float(np.mean(td_errors))

    # ----------------------------------------------------------
    @property
    def avg_reward(self) -> float:
        return float(np.mean(self.recent_rewards)) if self.recent_rewards else 0.0

    @property
    def avg_loss(self) -> float:
        return float(np.mean(self.recent_losses)) if self.recent_losses else 0.0

    @property
    def avg_td_error(self) -> float:
        return float(np.mean(self.recent_td_errors)) if self.recent_td_errors else 0.0

    def get_stats(self) -> Dict:
        return {
            'epsilon':        round(self.epsilon, 4),
            'avg_reward':     round(self.avg_reward, 3),
            'avg_loss':       round(self.avg_loss, 5),
            'avg_td_error':   round(self.avg_td_error, 4),
            'train_step':     self.train_step,
            'total_episodes': self.total_eps,
            'memory_size':    len(self.memory),
            'lr':             round(self.learning_rate, 6),
            'unique_states':  self.curiosity.coverage(),
            'curriculum':     self.curriculum.get_stats(),
        }

    def get_weights(self) -> Dict:
        return {
            'online': self.online_net.get_weights(),
            'target': self.target_net.get_weights(),
        }

    def set_weights(self, d: Dict):
        if 'online' in d:
            self.online_net.set_weights(d['online'])
        if 'target' in d:
            self.target_net.set_weights(d['target'])
