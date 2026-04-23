# RIENFoRZe — IV
### Project A.L.I.V.E. NEXUS · Tabular Dyna-Q Architecture · April 2026

> *A fundamental paradigm shift. The entire neural network stack — weights, gradients, Adam moments, backpropagation — is discarded. In its place: a pure tabular Q-learning agent with a perfect world model. No approximation errors. No vanishing gradients. No mini-batch sampling bias. Direct, exact, algebraic updates. This is not a simplification — it is a hypothesis: that the maze environment's determinism can be exploited more efficiently by exact tabular methods than by gradient approximation. The neural architecture remains in the codebase, commented out, as an architectural fossil.*

---

## Table of Contents

1. [The Paradigm Shift — Why the Neural Network Was Abandoned](#the-paradigm-shift--why-the-neural-network-was-abandoned)
2. [State Representation — 52D Observation Retained](#state-representation--52d-observation-retained)
3. [Tabular Q-Function — Structure and Addressing](#tabular-q-function--structure-and-addressing)
4. [Exact Q-Learning Update Rule](#exact-q-learning-update-rule)
5. [Perfect World Model — Dyna-Q Without Approximation](#perfect-world-model--dyna-q-without-approximation)
6. [Dyna-Q Planning Loop — 20 Steps + Super Brain Mode](#dyna-q-planning-loop--20-steps--super-brain-mode)
7. [State Hashing — Continuous to Hashable](#state-hashing--continuous-to-hashable)
8. [Curiosity Module — Adapted for Mixed State Types](#curiosity-module--adapted-for-mixed-state-types)
9. [Action Selection — Epsilon-Greedy Over Q-Table](#action-selection--epsilon-greedy-over-q-table)
10. [Convergence Guarantees — Tabular vs Approximate RL](#convergence-guarantees--tabular-vs-approximate-q-vs-approximate-rl)
11. [Memory Complexity Analysis](#memory-complexity-analysis)
12. [Why Learning Rate is 0.3 — Not 0.001](#why-learning-rate-is-03--not-0001)
13. [Persistence — JSON Serialization of Q-Table](#persistence--json-serialization-of-q-table)
14. [Advanced Streamlit UI — Largest Instrumentation Suite](#advanced-streamlit-ui--largest-instrumentation-suite)
15. [Comparison with Neural Versions](#comparison-with-neural-versions)
16. [Hyperparameter Reference](#hyperparameter-reference)

---

## The Paradigm Shift — Why the Neural Network Was Abandoned

RIENFoRZe-I through III use a Dueling DQN — a function approximator that maps the observation vector to Q-values via a sequence of matrix multiplications and nonlinearities. This approximation is necessary when the state space is too large for exact tabular representation.

The RIENFoRZe maze environment is **deterministic and bounded**. The maze is regenerated each episode, but within a single episode, the transition function is:

```math
T(s, a) = s' \quad \text{(deterministic)}
```

This determinism means the world model is **perfectly learnable**: every (state, action) pair the agent visits produces an exact, reproducible (next\_state, reward) pair. There is no stochasticity to approximate.

### The Approximation Error Argument

A neural network Q-function approximates:

```math
Q_\theta(s, a) \approx Q^*(s, a)
```

The approximation error introduces three classes of instability:

1. **Function approximation error**: The network may not have sufficient capacity to represent Q* exactly for large state spaces. In a deterministic maze, Q* is a well-defined piecewise function with sharp transitions at wall boundaries — difficult for smooth neural approximators.

2. **Gradient interference**: Updating Q(s, a) with a gradient step may inadvertently change Q(s', a') for nearby states s' due to weight sharing. This is the Deadly Triad (Sutton & Barto, 2018) — off-policy training + bootstrapping + function approximation creates instability.

3. **Bootstrapping bias**: The Bellman target is computed using the current (or lagged) network, introducing moving target problems.

In a deterministic environment with a bounded, enumerable state space, none of these issues need to exist. A tabular Q-table stores Q*(s, a) exactly, updates are exact, and the target is fixed (the true next Q-value, stored in the same table).

### The Fossil

The entire neural network codebase remains in `brain.py`, commented out with explicit markers:

```python
# --- OLD NEURAL ARCHITECTURE (Bypassed) ---
# self.online_net = NeuralNet(state_size, h1, h2, h3, action_size)
# self.target_net = NeuralNet(state_size, h1, h2, h3, action_size)
# ...
# --- NEW PERFECT MEMORY ARCHITECTURE ---
self.q_table = {}
self.model = {}
```

This commentary is intentional — it preserves the ablation history inline and allows direct inspection of exactly what was replaced.

---

## State Representation — 52D Observation Retained

The 52-dimensional observation vector from RIENFoRZe-II is retained without modification. The same `world.py` encodes:

```
s ∈ R^52  =  [vision(25) | pheromones(13) | pos(2) | tpos(2) | dir(2) | dist(1) | trap(1) | fog(1) | time(1) | momentum(4)]
```

The difference is in how this vector is used: rather than being passed as floating-point input to a neural network, it is **discretized and hashed** into a hashable Python tuple serving as the Q-table dictionary key.

---

## Tabular Q-Function — Structure and Addressing

The Q-function is a Python dictionary:

```python
q_table: Dict[Tuple[Tuple, int], float]
q_table[(s_key, action)] = q_value
```

where `s_key` is a hashable tuple derived from the 52D observation vector, and `action` ∈ {0, 1, 2, 3}.

### Default Value

For any (s, a) pair not yet visited, the Q-value defaults to 0.0:

```math
Q(s, a) = 0.0 \quad \text{if } (s, a) \notin \text{dom}(Q_{\text{table}})
```

This is an optimistic initialization in a sparse reward environment — zero is neither optimistic nor pessimistic relative to an average trajectory (which usually accumulates moderate step penalties before reaching the goal).

### Key Density

The Q-table grows monotonically with exploration. After N unique (state, action) observations, the table has exactly N entries. Unlike a neural network whose parameters are fixed at initialization, the tabular Q-function's memory grows with experience.

---

## Exact Q-Learning Update Rule

For each observed transition (s, a, r_aug, s', done):

### Step 1: Lookup

```math
Q_{\text{current}} = Q(s, a) \quad (\text{default } 0.0 \text{ if unseen})
```

```math
Q^*_{\text{next}} = \max_{a' \in \mathcal{A}} Q(s', a')
```

### Step 2: Target

```math
y = r_{\text{aug}} + (1 - \mathbb{1}[\text{done}]) \cdot \gamma \cdot Q^*_{\text{next}}
```

### Step 3: TD Error

```math
\delta = y - Q_{\text{current}}
```

### Step 4: Update

```math
Q(s, a) \leftarrow Q_{\text{current}} + \alpha \cdot \delta
```

This is the standard **Q-learning** (Watkins, 1989) update. Because the Q-table is exact (no approximation error), there is no Deadly Triad. The update to Q(s, a) does not affect any other table entry.

### Convergence Theorem

For a finite Markov Decision Process (MDP), Q-learning converges to Q* given:

1. Every (s, a) pair is visited infinitely often
2. The learning rate satisfies the Robbins-Monro conditions:

```math
\sum_{t=0}^{\infty} \alpha_t = \infty, \quad \sum_{t=0}^{\infty} \alpha_t^2 < \infty
```

With constant alpha = 0.3, condition 2 is violated (sum diverges). This is deliberate — in a deterministic environment, exact targets mean that the optimal update is a simple overwrite: alpha = 1.0 converges in a single visit per (s, a) pair. alpha = 0.3 provides a compromise that handles any residual noise from the curiosity bonus or curriculum transitions.

---

## Perfect World Model — Dyna-Q Without Approximation

Alongside the Q-table, a **world model** dictionary stores observed transitions:

```python
model: Dict[Tuple, Tuple]
model[(s_key, action)] = (ns_key, augmented_reward)
```

Each entry stores the **exact** next state and augmented reward observed for that (state, action) pair. Because the environment is deterministic, the last-observed transition is the correct transition — there is no need to average multiple observations.

### Model Accuracy

For a deterministic MDP:

```math
P(s' \mid s, a) = \mathbb{1}[s' = T(s, a)]
```

The model M(s, a) = (T(s, a), R(s, a)) is correct with probability 1 after a single visit. This is in sharp contrast to learned stochastic models (e.g., Dyna-Q in stochastic environments), which require multiple observations to estimate transition distributions accurately.

### Why This Matters for Planning

When the model is exact, every Dyna-Q planning step uses **ground-truth transitions** rather than model predictions. There is no model bias to propagate. Each planning step is as reliable as a real environment step, differing only in computational cost (dictionary lookup) vs real cost (maze physics simulation).

---

## Dyna-Q Planning Loop — 20 Steps + Super Brain Mode

### Normal Planning Phase

After each real environment step, 20 simulated updates are performed:

```math
\text{for } k = 1, \ldots, 20:
```

```math
(s_{\text{sim}}, a_{\text{sim}}) \sim \text{Uniform}(\text{dom}(M))
```

```math
(s'_{\text{sim}}, r_{\text{sim}}) \leftarrow M(s_{\text{sim}}, a_{\text{sim}})
```

```math
Q_{\text{sim}}^* = \max_{a'} Q(s'_{\text{sim}}, a')
```

```math
Q(s_{\text{sim}}, a_{\text{sim}}) \leftarrow Q(s_{\text{sim}}, a_{\text{sim}}) + 0.3 \cdot (r_{\text{sim}} + \gamma Q_{\text{sim}}^* - Q(s_{\text{sim}}, a_{\text{sim}}))
```

### Super Brain Mode

On successful goal-reaching (reward > 10.0, done=True):

```math
K_{\text{breakthrough}} = 4 \times 20 = 80 \text{ planning steps}
```

### Effective Update Ratios

| Condition | Real steps | Q updates | Ratio |
|---|---|---|---|
| Normal | 1 | 1 + 20 = 21 | 21:1 |
| Breakthrough episode | T + 1 | T·21 + 81 | ~21:1 avg |

### Value Propagation from Model

Because simulated transitions are drawn uniformly from the model, value information can propagate backward through the stored trajectory graph. After seeing the goal state G with high reward, planning steps can update the predecessor states of G, then their predecessors, etc.

The rate of backward propagation through uniform Dyna-Q is stochastic. With model size |M|:

```math
\Pr[\text{goal's predecessor sampled}] = \frac{|\text{predecessors of G}|}{|M|}
```

This is a probabilistic argument — prioritized sweeping (as in MazE.py) would be more efficient. The choice of uniform sampling here is intentional: it tests whether raw planning volume (20 steps) compensates for lack of priority ordering.

---

## State Hashing — Continuous to Hashable

The 52D continuous observation vector must be converted to a hashable dictionary key. The hashing function is type-aware:

### Type Dispatch

```python
def _key(self, state) -> tuple:
    if isinstance(state, tuple):
        return state                                    # Already discrete (maze coordinates)
    s_arr = np.array(state)
    if s_arr.size <= 2:
        return tuple(s_arr.tolist())                   # Small array → direct tuple
    return tuple(
        (np.clip(s_arr, 0, 1) * (bins - 1)).astype(int).tolist()
    )                                                   # 52D → discretized integer tuple
```

### Discretization

For the 52D case with bins = 16:

```math
k_i = \lfloor \text{clip}(s_i, 0, 1) \cdot 15 \rfloor \in \{0, 1, \ldots, 15\}
```

Each dimension is independently quantized to one of 16 integer levels. The resulting key is a 52-tuple of integers, which is hashable and comparable.

### Collision Rate

The maximum number of distinct keys is:

```math
|\mathcal{K}| \leq 16^{52} \approx 10^{62.7}
```

In practice, the agent visits a tiny fraction of this space. For a 35×41 Level-10 maze with 718 path cells and 4 actions, the relevant state space is much smaller. Discretization collisions — two distinct continuous states mapping to the same key — are unavoidable but acceptable: nearby states in continuous space are mapped to the same discrete region, which is appropriate given the smooth structure of the pheromone and position features.

### Sensitivity

The key is most sensitive to the 25 binary vision dimensions (which change sharply at walls) and least sensitive to continuous gradients like fog coverage and time pressure (which change slowly). This matches the desired sensitivity: the agent's key position and local geometry are the most informative aspects of its situation.

---

## Curiosity Module — Adapted for Mixed State Types

The `IntrinsicCuriosity` class is adapted in RIENFoRZe-IV to handle both continuous numpy arrays and discrete tuple states:

### Key Generation (Extended)

```python
def _key(self, state) -> tuple:
    if isinstance(state, tuple):
        return state
    s_arr = np.array(state)
    if s_arr.size <= 2:
        return tuple(s_arr.tolist())
    return tuple(
        (np.clip(s_arr, 0, 1) * (self.bins - 1)).astype(int).tolist()
    )
```

### Bonus Computation

```math
r_i(s) = \frac{\beta}{\sqrt{N(k(s))}}, \quad \beta = 0.05
```

The curiosity bonus is computed on the numpy array (before hashing) and added to the external reward before storing in the world model:

```math
r_{\text{aug}} = r_{\text{ext}} + r_i(s)
```

This means the world model stores augmented rewards — planning steps automatically include the curiosity shaping. This is a design choice: the curiosity bonus is treated as part of the environment's effective reward function, not as a separate channel.

---

## Action Selection — Epsilon-Greedy Over Q-Table

The greedy action is selected by looking up Q-values for all four actions at the current key and returning the argmax:

```math
a^* = \arg\max_{a \in \mathcal{A}} Q(k(s), a)
```

### Tie-Breaking

If multiple actions share the maximum Q-value (common in early training when the Q-table is sparse and defaults to 0.0 everywhere), a random tie-break is applied:

```math
a^* \sim \text{Uniform}\!\left(\left\{ a : Q(k(s), a) = \max_{a'} Q(k(s), a') \right\}\right)
```

This is an important detail: without tie-breaking, the agent would always select action 0 (up) when all Q-values are equal, creating a systematic bias that slows exploration of other directions.

### Epsilon-Greedy Policy

```math
\pi(a \mid s) = \begin{cases}
\text{Uniform}(\mathcal{A}) & \text{with probability } \epsilon \\
a^*(s) & \text{with probability } 1 - \epsilon
\end{cases}
```

Epsilon is decayed once per episode:

```math
\epsilon_{n+1} = \max(\epsilon_{\text{min}}, \epsilon_n \cdot \lambda_\epsilon)
```

with epsilon_min = 0.05 and lambda = 0.995 (same as RIENFoRZe-II/III).

---

## Convergence Guarantees — Tabular vs Approximate RL

### Tabular Q-Learning Convergence

Under the standard assumptions for finite MDPs (Watkins & Dayan, 1992):

```math
\lim_{t \to \infty} Q_t(s, a) = Q^*(s, a) \quad \text{almost surely}
```

provided every (s, a) is visited infinitely often and the learning rate schedule satisfies the Robbins-Monro conditions.

### Why This Guarantee Fails for Neural DQN

The DQN (Mnih et al., 2015) does **not** have a convergence guarantee to Q* because:

1. The function approximator is not guaranteed to represent Q* exactly
2. Off-policy training with bootstrapped targets and function approximation forms the Deadly Triad
3. The target network introduces a moving target that never fully converges

Empirically, DQN works well in practice. But it is an approximation whose convergence is not theoretically guaranteed.

### The Tabular Advantage in Deterministic Environments

In a deterministic MDP, after visiting every (s, a) pair at least once, the world model M is complete and exact. The Dyna-Q planning loop then performs pure **dynamic programming** over an exact model — equivalent to value iteration:

```math
Q_{k+1}(s, a) = R(s, a) + \gamma \max_{a'} Q_k(s', a')
```

This is guaranteed to converge to Q* for finite MDPs. The rate of convergence depends on the number of planning steps and the discount factor.

---

## Memory Complexity Analysis

### Q-Table

The Q-table stores at most one entry per (key, action) pair. With |S| unique discrete states visited and |A| = 4 actions:

```math
\text{mem}(Q) = O(|S| \cdot 4 \cdot \text{float64}) = O(|S| \cdot 32 \text{ bytes})
```

### World Model

The world model stores one entry per (key, action) pair, mapping to (next\_key, reward):

```math
\text{mem}(M) = O(|S| \cdot 4 \cdot (52 \cdot \text{int32} + \text{float64})) = O(|S| \cdot 4 \cdot 216 \text{ bytes})
```

### PER Buffer (Absent)

RIENFoRZe-I through III maintain a PER replay buffer of capacity 50,000 transitions, each storing:

```math
\text{mem}(\text{PER}) = 50000 \cdot (52 + 1 + 1 + 52 + 1) \cdot 4 \text{ bytes} = 50000 \cdot 428 \text{ bytes} \approx 20.9 \text{ MB}
```

plus segment tree overhead (~6.4 MB). Total ≈ 27 MB.

RIENFoRZe-IV eliminates the PER buffer entirely. For small mazes (|S| ~ 10^3 to 10^4), the Q-table and model together use well under 1 MB — an order-of-magnitude memory reduction.

### Neural Network (Absent)

The 58,117 parameters of the RIENFoRZe-III network, stored as float64:

```math
\text{mem}(\theta) = 58117 \cdot 8 \text{ bytes} \approx 465 \text{ KB}
```

Two copies (online + target) plus Adam moments (2 more copies): 4 × 465 KB ≈ 1.86 MB. Also eliminated.

---

## Why Learning Rate is 0.3 — Not 0.001

The neural network Adam optimizer uses lr = 0.001 because:
- Gradient-based updates are noisy (mini-batch sampling variance)
- The loss landscape has saddle points and sharp minima
- Large learning rates cause oscillation around minima

The tabular Q-learning update uses lr = 0.3 because:
- Updates are exact (no sampling variance in the gradient)
- There is no loss landscape — each update is a direct correction
- The optimal lr for a deterministic environment is 1.0 (overwrite target directly)

### Why Not 1.0

Even in a deterministic environment, the augmented reward (r + curiosity bonus) contains stochasticity from the curiosity count history. With lr = 1.0:

```math
Q(s, a) \leftarrow r_{\text{aug}} + \gamma \max_{a'} Q(s', a')
```

Each update completely overwrites the previous estimate. If r_aug varies between visits due to curiosity bonus decay, the Q-value oscillates. With lr = 0.3, the Q-value is a damped average of recent targets:

```math
Q(s, a) \leftarrow (1 - 0.3) Q(s, a) + 0.3 \cdot y
```

The half-life of old information under this update is:

```math
t_{1/2} = \frac{\ln 2}{\ln(1/(1 - \alpha))} = \frac{\ln 2}{\ln(1/0.7)} = \frac{0.693}{0.357} \approx 1.94 \text{ visits}
```

Within 2 visits to the same (s, a) pair, the Q-value is dominated by recent information.

---

## Persistence — JSON Serialization of Q-Table

The Q-table cannot be saved directly as JSON because Python tuple keys are not JSON-serializable. RIENFoRZe-IV implements string-key serialization:

```python
def get_weights(self) -> Dict:
    return {"q_table": {str(k): v for k, v in self.q_table.items()}}
```

On load, string keys are converted back to tuples using Python's `ast.literal_eval`:

```python
def set_weights(self, d: Dict):
    if "q_table" in d:
        self.q_table = {
            ast.literal_eval(k): v
            for k, v in d["q_table"].items()
        }
```

### Archive Format

The saved state is packaged as a ZIP archive containing:
- `weights.json`: Q-table as string-keyed dictionary
- `config.json`: Hyperparameter snapshot
- `stats.json`: Training statistics snapshot

This mirrors the save format of the neural versions for API compatibility.

---

## Advanced Streamlit UI — Largest Instrumentation Suite

RIENFoRZe-IV's `RIENFoRZeADv.py` is the largest UI file in the series (2,102 lines vs 2,058 for v-III), with the most extensive scientific instrumentation panel:

| Panel | Metric | Detail |
|-------|--------|--------|
| Q-Table Size | Live dict size | Total (s, a) pairs learned |
| Model Coverage | World model size | Fraction of state-action space modeled |
| Value Distribution | Q-value histogram | Distribution of stored Q-values |
| Policy Entropy | H(pi) over table | Entropy of greedy policy across visited states |
| TD Error Stream | Live delta | Per-step TD error magnitude |
| Visit Heatmap | 2D exploration | Visit count grid over maze (position dims) |
| Curiosity decay | w_c over episodes | Curiosity weight schedule visualization |
| Planning trace | Simulated steps | Number of Dyna-Q steps per real step |
| Convergence | Q-value delta | Mean absolute change in Q-table per episode |
| Episode compare | best/worst/current | Three-episode overlay panel |
| Research lab | Full derivations | Tabular RL theory and equations tab |
| Architecture | Neural fossil | Commented-out neural code displayed for reference |

The policy entropy over the Q-table is computed as:

```math
H(\pi) = -\sum_{s \in \mathcal{S}_{\text{visited}}} \pi^*(s) \log \pi^*(s)
```

where pi*(s) is 1 at the greedy action and 0 elsewhere. As the Q-table converges, ties are broken and H(pi) decreases toward 0. High H(pi) indicates many states with equal Q-values (sparse/early Q-table). Low H(pi) indicates a decisive, well-learned policy.

---

## Comparison with Neural Versions

| Property | RIENFoRZe-I | RIENFoRZe-II | RIENFoRZe-III | RIENFoRZe-IV |
|---|---|---|---|---|
| State dim | 17 | 52 | 64 | 52 |
| Q-function | Neural (58K params) | Neural (58K params) | Neural (58K params) | Tabular (dict) |
| Update type | Gradient descent | Gradient descent | Gradient descent | Exact update |
| Convergence guarantee | No | No | No | Yes (finite MDP) |
| Memory (Q-function) | ~465 KB | ~465 KB | ~465 KB | O(|S|·32B) |
| PER buffer | 50K transitions | 50K transitions | 50K transitions | **None** |
| Target network | Yes (soft update) | Yes (soft update) | Yes (soft update) | **None needed** |
| Planning steps | None | 5 | 25 | 20 |
| Breakthrough | None | 4× (20) | 5× (125) | 4× (80) |
| Curiosity in state | No | No | Yes (dim 63) | No |
| Wall radar | No | No | Yes (4D) | No |
| Scent gradients | No | No | Yes (4D) | No |
| Loss function | MSE | Huber | Huber | None (exact) |
| Gradient clipping | Yes | Yes | Yes | **Not applicable** |
| Adam optimizer | Yes | Yes | Yes | **Not applicable** |

---

## Hyperparameter Reference

| Parameter | Value | Note |
|-----------|-------|------|
| `state_size` | 52 | Same observation as v-II |
| `action_size` | 4 | Same |
| `q_table` | `{}` | Empty dict at init, grows with experience |
| `model` | `{}` | Empty dict at init, grows with experience |
| `learning_rate` | 0.3 | Tabular alpha — not gradient step size |
| `gamma` | 0.99 | Discount factor (unchanged across all versions) |
| `epsilon_start` | 0.7 | Warm start (same as v-II/III) |
| `epsilon_min` | 0.05 | Same |
| `epsilon_decay` | 0.995 | Per episode (same) |
| `planning_steps` | 20 | Dyna-Q steps per real step |
| `breakthrough_mult` | 4× | Goal-contingent planning (80 total) |
| `breakthrough_threshold` | reward > 10.0 | Same as v-II |
| `icm_beta` | 0.05 | Curiosity bonus scale |
| `icm_bins` | 16 | State discretization bins for curiosity hashing |
| `lr_scheduler` | Disabled (commented) | Tabular learning does not need LR scheduling |
| Neural network | **Absent** | Entire network class bypassed |
| PER buffer | **Absent** | No replay buffer |
| N-step buffer | **Absent** | No N-step accumulation |
| Segment tree | **Absent** | No priority structure |
| Target network | **Absent** | No Polyak averaging needed |
| Adam optimizer | **Absent** | No gradient-based optimization |
| Gradient clipping | **Absent** | No gradients exist |
| Weight clipping | **Absent** | No weights exist |

---

*RIENFoRZe-IV — Tabular Dyna-Q Edition — April 2026*
*A deliberate departure from approximation. Proof of concept that deterministic environments may be better served by exact methods.*
