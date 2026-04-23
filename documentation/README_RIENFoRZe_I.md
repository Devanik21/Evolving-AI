
# RIENFoRZe — I
### Project A.L.I.V.E. NEXUS · Foundational Architecture · April 2026

> *The origin experiment. A self-contained reinforcement learning system built entirely in pure NumPy — no deep learning framework, no autograd engine. Every gradient is derived by hand. Every optimizer step is explicit. This is the baseline from which all subsequent versions diverge.*

---

## Table of Contents

1. [System Overview](#system-overview)
2. [State Representation — 17-Dimensional Observation Space](#state-representation--17-dimensional-observation-space)
3. [Neural Architecture — Dueling DQN](#neural-architecture--dueling-dqn)
4. [Prioritized Experience Replay](#prioritized-experience-replay)
5. [N-Step Return Estimation](#n-step-return-estimation)
6. [Intrinsic Curiosity Module](#intrinsic-curiosity-module)
7. [Double DQN Target Computation](#double-dqn-target-computation)
8. [Adam Optimizer — Manual Derivation](#adam-optimizer--manual-derivation)
9. [Soft Target Network Updates](#soft-target-network-updates)
10. [Adaptive Curriculum Learning](#adaptive-curriculum-learning)
11. [Environment — Maze Generation Algorithms](#environment--maze-generation-algorithms)
12. [Reward Engineering](#reward-engineering)
13. [MazE Module — SARSA Agent](#maze-module--sarsa-agent)
14. [File Architecture](#file-architecture)
15. [Hyperparameter Reference](#hyperparameter-reference)

---

## System Overview

RIENFoRZe-I is the founding experimental configuration of Project A.L.I.V.E. NEXUS. It establishes the complete learning pipeline: a **Dueling Double DQN** agent navigating procedurally generated mazes, trained via **Prioritized Experience Replay** with **N-step returns** and augmented by a **count-based Intrinsic Curiosity Module**. Difficulty is managed by an **Adaptive Curriculum** spanning 10 levels. The entire system operates without any ML framework — every matrix multiplication, every backpropagation pass, every optimizer moment estimate is written explicitly in NumPy.

The companion `MazE.py` module provides a fully independent Streamlit application implementing a **SARSA + Prioritized Sweeping** agent for pedagogical comparison and ablation.

**Core modules:**

| File | Role | Lines |
|------|------|-------|
| `brain.py` | RL engine, neural net, PER, N-step, ICM, curriculum | ~668 |
| `world.py` | Maze generation, environment dynamics, A* solver | ~600 |
| `soul.py` | Emotion model, personality, NLP, episodic memory | ~700 |
| `memory_palace.py` | Long-term knowledge persistence | ~400 |
| `analytics.py` | Performance instrumentation | ~350 |
| `RIENFoRZe.py` | Main entry point | ~800 |
| `RIENFoRZeADv.py` | Advanced Streamlit UI | ~980 |

---

## State Representation — 17-Dimensional Observation Space

The agent observes the world through a **17-dimensional continuous vector** constructed at each time step. This is the most compact observation space across all four RIENFoRZe versions.

### Decomposition

```
s ∈ R^17  =  [vision(9)  |  pos(2)  |  tpos(2)  |  dist(1)  |  trap(1)  |  fog(1)  |  time(1)]
```

**Component breakdown:**

| Index | Dimension | Description | Range |
|-------|-----------|-------------|-------|
| 0–8 | 9 | 3×3 local vision field (wall/path/fog) | [0, 1] |
| 9–10 | 2 | Normalized agent position (r/H, c/W) | [0, 1] |
| 11–12 | 2 | Normalized target position (r/H, c/W) | [0, 1] |
| 13 | 1 | Normalized Manhattan distance to goal | [0, 1] |
| 14 | 1 | Normalized distance to nearest trap | [0, 1] |
| 15 | 1 | Fog-of-war coverage ratio | [0, 1] |
| 16 | 1 | Time pressure (step / max\_steps) | [0, 1] |

### Vision Encoding

For each of the 9 cells in the 3×3 neighborhood centered on agent position (r, c):

```math
v_{dr,dc} = \begin{cases}
0.0 & \text{if cell is a passable path and visible} \\
1.0 & \text{if cell is a wall or out-of-bounds} \\
0.5 & \text{if cell is within fog-of-war (not yet revealed)}
\end{cases}
```

where (dr, dc) ranges over {-1, 0, 1} x {-1, 0, 1}.

### Distance Features

The normalized Manhattan distance serves as a shaped guidance signal:

```math
d_{\text{manhattan}} = \frac{|r_{\text{agent}} - r_{\text{target}}| + |c_{\text{agent}} - c_{\text{target}}|}{H + W}
```

Time pressure encodes urgency as a linear ramp:

```math
\tau = \frac{t_{\text{current}}}{t_{\text{max}}}
```

Both features are clipped to [0, 1]. The fog coverage is computed as:

```math
\phi_{\text{fog}} = \frac{|\{(r,c) : \text{visible}[r,c] = \text{True}\}|}{H \cdot W}
```

---

## Neural Architecture — Dueling DQN

The network implements a **Dueling architecture** (Wang et al., 2016) with three shared hidden layers followed by two separate output streams.

### Layer Structure

```
Input (17) → [Linear + LeakyReLU] → (256) → [Linear + LeakyReLU] → (128)
           → [Linear + LeakyReLU] → (64)
           → Value stream:    (64) → (1)         → V(s)
           → Advantage stream:(64) → (4 actions)  → A(s, a)
           → Q(s, a) = V(s) + A(s, a) - mean_a[A(s, a)]
```

### Dueling Aggregation

The Q-value is computed as:

```math
Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'; \theta, \alpha) \right)
```

The mean-centering of the advantage stream forces V(s) to learn the true state value rather than absorbing an arbitrary constant. This resolves the identifiability problem inherent in naive separation of V and A.

### Activation Function

Leaky ReLU is used throughout to prevent the dying neuron problem:

```math
f_{\text{leaky}}(x) = \begin{cases} x & x > 0 \\ 0.01 \cdot x & x \leq 0 \end{cases}
```

Its gradient:

```math
\frac{df_{\text{leaky}}}{dx} = \begin{cases} 1.0 & x > 0 \\ 0.01 & x \leq 0 \end{cases}
```

### He Initialization

All weight matrices are initialized using He (Kaiming) normal initialization to maintain variance across layers with ReLU-family activations:

```math
W^{(l)} \sim \mathcal{N}\!\left(0,\ \frac{2}{n_{l-1}}\right)
```

where n_{l-1} is the fan-in of layer l. Bias vectors are initialized to zero.

---

## Prioritized Experience Replay

The replay buffer implements **Prioritized Experience Replay** (Schaul et al., 2015) using a binary **Segment Tree** data structure for O(log N) priority-weighted sampling.

### Priority Assignment

Each transition (s, a, r, s', done) is stored with priority p_i. At insertion time, p_i is set to the current maximum observed priority to ensure new experiences are sampled at least once:

```math
p_i = \max_{j \in \mathcal{B}} p_j
```

Sampling probability is proportional to a smoothed priority:

```math
P(i) = \frac{p_i^{\alpha}}{\sum_j p_j^{\alpha}}
```

where alpha controls the degree of prioritization. alpha = 0 recovers uniform sampling; alpha = 1 gives full greedy prioritization. This experiment uses **alpha = 0.6**.

### Importance-Sampling Correction

To correct the bias introduced by non-uniform sampling, each update uses an importance-sampling weight:

```math
w_i = \left( \frac{1}{N \cdot P(i)} \right)^{\beta}
```

Weights are normalized by the maximum weight in the batch:

```math
\hat{w}_i = \frac{w_i}{\max_j w_j}
```

The annealing schedule for beta runs from beta_start = 0.4 to 1.0 over beta_frames = 100,000 steps:

```math
\beta(t) = \min\!\left(1.0,\ \beta_{\text{start}} + t \cdot \frac{1 - \beta_{\text{start}}}{\text{beta\_frames}}\right)
```

As beta → 1, the correction becomes unbiased in the limit.

### Segment Tree Operations

The Segment Tree supports two concurrent structures:

- **SumSegmentTree**: Enables O(log N) stratified prefix-sum sampling
- **MinSegmentTree**: Enables O(log N) minimum priority lookup for weight normalization

Capacity is forced to the next power of 2 to enable efficient binary indexing:

```math
\text{cap} = 2^{\lceil \log_2 N \rceil}
```

Prefix-sum sampling procedure: divide total priority sum S into batch\_size equal segments. Within each segment [S·i/B, S·(i+1)/B], sample a uniform variate and retrieve the corresponding index via the segment tree's find_prefixsum operation in O(log N) time.

---

## N-Step Return Estimation

To provide richer temporal credit assignment, transitions are stored after computing an **N-step discounted return** (N = 3 by default).

### N-Step Target

For a trajectory starting at time t:

```math
G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max_{a'} Q_{\text{target}}(s_{t+n}, a')
```

The accumulated return G replaces the single-step reward r in the Bellman update, providing a lookahead of n = 3 steps.

### Buffer Logic

The N-step buffer holds a deque of length n. When full, it computes the n-step return for the oldest transition and flushes it to the main PER buffer. Absorbing states (done=True) terminate the accumulation early:

```math
G_t^{(n)} = \sum_{k=0}^{k^*} \gamma^k r_{t+k}
```

where k* is the index of the first terminal state in the window.

---

## Intrinsic Curiosity Module

RIENFoRZe-I uses a **count-based exploration bonus** (Bellemare et al., 2016 inspired) rather than a learned forward model. This is intentional — the architecture experiments with pure tabulation before ascending to gradient-based novelty estimation in later versions.

### Exploration Bonus

For each state s, the visit count N(s) is tracked via discretization of the continuous state vector into a grid of `bins = 16` per dimension:

```math
k(s) = \left\lfloor \text{clip}(s, 0, 1) \cdot (B - 1) \right\rfloor \quad \in \mathbb{Z}^{17}
```

The intrinsic reward is then:

```math
r_i(s) = \frac{\beta}{\sqrt{N(k(s))}}
```

with beta = 0.05. This decays as O(1/sqrt(N)), matching the theoretical prediction from pseudocount theory. The augmented reward signal fed to the agent is:

```math
r_{\text{aug}} = r_{\text{ext}} + r_i(s)
```

### Coverage Metric

The number of unique discretized states ever visited is tracked as an exploration coverage metric:

```math
\mathcal{C} = |\{k(s) : s \text{ ever visited}\}|
```

A spatial heatmap over the (row, col) dimensions of the state can be rendered for diagnostic inspection.

---

## Double DQN Target Computation

Standard DQN suffers from maximization bias because the same network both selects and evaluates the greedy action. **Double DQN** (van Hasselt et al., 2015) decouples these two operations across the online and target networks.

### Target Calculation

```math
a^* = \arg\max_{a'} Q_{\text{online}}(s', a'; \theta)
```

```math
y_i = r_i + \gamma (1 - d_i) \cdot Q_{\text{target}}(s'_i, a^*; \theta^-)
```

where theta is the online network parameters, theta^- is the (lagged) target network parameters, and d_i is the terminal indicator.

### TD Error for PER

After computing targets, the TD error for each transition is:

```math
\delta_i = \left| y_i - Q_{\text{online}}(s_i, a_i; \theta) \right|
```

This is used to update the segment tree priorities:

```math
p_i \leftarrow \max(\delta_i,\ \epsilon_{\text{floor}})
```

with epsilon_floor = 1e-6 to prevent zero-probability sampling.

### Loss Gradient

The weighted MSE loss and its gradient with respect to Q predictions:

```math
\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} \hat{w}_i \cdot (y_i - Q(s_i, a_i))^2
```

```math
\frac{\partial \mathcal{L}}{\partial Q(s_i, a_i)} = -2 \hat{w}_i (y_i - Q(s_i, a_i))
```

Only the Q-value at the taken action a_i is differentiated; other action outputs have zero gradient.

---

## Adam Optimizer — Manual Derivation

The optimizer is implemented from first principles. For each parameter tensor p and its gradient g at update step t:

### Moment Estimates

```math
m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t
```

```math
v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
```

### Bias Correction

Because m and v are initialized at zero, early estimates are biased toward zero. The bias-corrected moments are:

```math
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
```

```math
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
```

### Parameter Update

```math
\theta_t \leftarrow \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t
```

Default hyperparameters: eta (learning rate) = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8.

### Gradient Clipping

Before the Adam update, all gradient tensors are clipped element-wise to [-10, 10] to prevent gradient explosion in the early, high-variance training phase:

```math
g_t \leftarrow \text{clip}(g_t, -c, c), \quad c = 10.0
```

---

## Soft Target Network Updates

Rather than periodically copying the online network to the target (hard update), RIENFoRZe-I uses **Polyak averaging** (soft update) at every training step:

```math
\theta^- \leftarrow \tau \cdot \theta + (1 - \tau) \cdot \theta^-
```

with tau = 0.005. This provides a smoothly lagging target, reducing oscillations during training.

### Convergence Intuition

The target lag introduces a time-scale separation: the online network adapts quickly (via gradient descent) while the target network follows slowly (via exponential moving average). For a scalar parameter with constant online value theta*, the target converges as:

```math
\theta^-_t = \theta^* + (\theta^-_0 - \theta^*)(1 - \tau)^t
```

The half-life of the lag is approximately t_{1/2} = ln(2) / tau ≈ 139 steps.

---

## Adaptive Curriculum Learning

The `CurriculumManager` implements **Automatic Curriculum Learning (ACL)** with 10 discrete difficulty levels. The manager monitors a rolling performance score and triggers level transitions to maintain the agent near its Zone of Proximal Development (ZPD).

### Episode Score

Each episode produces a composite score combining success rate and efficiency:

```math
\text{eff} = \max\!\left(0,\ 1 - \frac{t_{\text{steps}}}{t_{\text{max}}}\right) \cdot \mathbb{1}[\text{success}]
```

```math
\text{score} = 0.5 \cdot \mathbb{1}[\text{success}] + 0.5 \cdot \text{eff}
```

### Promotion / Demotion Logic

A rolling window of 20 recent episodes is maintained. Mean score triggers transitions:

```math
\bar{s} = \frac{1}{|\mathcal{W}|} \sum_{i \in \mathcal{W}} \text{score}_i
```

```math
\text{Level} \leftarrow \begin{cases}
\text{Level} + 1 & \bar{s} \geq 0.72 \ \text{and Level} < 10 \\
\text{Level} - 1 & \bar{s} \leq 0.25 \ \text{and Level} > 1 \\
\text{Level}     & \text{otherwise}
\end{cases}
```

After any transition, the rolling window is cleared to re-evaluate at the new difficulty.

### Level Configurations

| Level | Maze H | Maze W | Algorithm | Fog | Dynamic | Portals |
|-------|--------|--------|-----------|-----|---------|---------|
| 1 | 7 | 9 | Backtracker | No | No | No |
| 2 | 9 | 11 | Backtracker | No | No | No |
| 3 | 11 | 13 | Prim | No | No | No |
| 4 | 13 | 15 | Prim | Yes | No | No |
| 5 | 15 | 19 | Wilson | Yes | No | No |
| 6 | 17 | 21 | Wilson | Yes | Yes | No |
| 7 | 21 | 25 | Backtracker | Yes | Yes | No |
| 8 | 25 | 29 | Prim | Yes | Yes | Yes |
| 9 | 29 | 33 | Wilson | Yes | Yes | Yes |
| 10 | 35 | 41 | Hybrid | Yes | Yes | Yes |

### ZPD Progress

```math
\rho_{\text{ZPD}} = \min\!\left(1.0,\ \frac{\bar{s}}{0.72}\right)
```

This quantity tracks progress toward the promotion threshold at the current level.

---

## Environment — Maze Generation Algorithms

The `MazeGenerator` class supports three distinct topological structures, each inducing different exploration challenges.

### Recursive Backtracker (DFS)

Produces long winding corridors with few dead-ends. A **straight-bias** parameter (65%) preferentially continues in the current direction, reducing winding complexity while preserving perfect maze topology.

Algorithm: Starting from cell (0, 0), carve a passage to an unvisited neighbor chosen via biased shuffle. Recurse. Backtrack when all neighbors are visited. System recursion limit is set to 10,000 to support large mazes.

### Prim's Algorithm

Produces highly branching mazes with many dead-ends. Trains the agent to handle junction decisions and backtracking. Maintains a frontier set of candidate walls; grows the maze by randomly selecting and removing a valid frontier wall.

### Wilson's Algorithm

Produces a **Uniform Spanning Tree** — an unbiased random spanning tree of the grid graph. Uses loop-erased random walks to guarantee that every spanning tree is equally likely. This is the theoretically most difficult generator for the agent, producing globally optimal but locally indistinct structure.

### A* Shortest Path

The environment includes a full A* solver using a min-heap priority queue:

```math
f(n) = g(n) + h(n), \quad h(n) = |r_n - r_{\text{goal}}| + |c_n - c_{\text{goal}}|
```

The shortest path length provides the optimal baseline for efficiency comparison. Agent step-efficiency is reported relative to this optimal.

---

## Reward Engineering

The reward function is potential-based, ensuring optimal policy invariance under shaping (Ng et al., 1999).

### Reward Components

```math
r = r_{\text{goal}} + r_{\text{progress}} + r_{\text{step}} + r_{\text{wall}} + r_{\text{trap}} + r_{\text{portal}} + r_{\text{timeout}}
```

| Component | Value | Condition |
|-----------|-------|-----------|
| Goal reward | +25.0 | Reached target |
| Distance progress | +3.0 · delta_d | Moved closer |
| Distance regression | -1.5 · delta_d | Moved farther |
| Step penalty | -0.05 | Each step |
| Wall penalty | -0.3 | Hit wall |
| Trap penalty | -10.0 | Caught by trap |
| Portal bonus | +2.0 | Used portal |
| Timeout penalty | -1.0 | Episode timeout |

Final reward is clipped: r ∈ [-10.0, 15.0].

The potential-based shaping ensures that with shaping function phi(s) = -d(s, goal), the shaped reward:

```math
r'(s, a, s') = r(s, a, s') + \gamma \cdot \phi(s') - \phi(s)
```

does not alter the set of optimal policies.

---

## MazE Module — SARSA Agent

`MazE.py` is a self-contained Streamlit application implementing **SARSA with Prioritized Sweeping** for independent comparison. This is architecturally distinct from the main DQN agent.

### SARSA Update Rule

Unlike Q-learning, SARSA is an on-policy algorithm. The update uses the action actually taken at the next state (a') rather than the greedy action:

```math
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]
```

This makes SARSA generally more conservative near dangerous states (traps), since it accounts for the probability of taking suboptimal exploration actions.

### Curiosity-Weighted Exploration

The agent tracks visit counts for all states and uses a curiosity weight that decays over training:

```math
r_{\text{curiosity}}(s) = \frac{w_{\text{curiosity}}}{\text{visit\_count}(s) + 1}
```

```math
w_{\text{curiosity}} \leftarrow w_{\text{curiosity}} \cdot 0.99 \quad \text{(per episode)}
```

### BFS Distance Map

At initialization, BFS from the goal computes an exact distance map over all reachable cells:

```math
d_{\text{BFS}}(s) = \min_{\pi} \mathbb{E}\left[\sum_{t=0}^{\infty} \mathbb{1}[s_t \neq s_{\text{goal}}]\right]
```

This distance map serves as the heuristic for reward shaping, providing a dense signal in sparse environments.

### Prioritized Sweeping

After each real environment step, the agent performs simulated updates using its stored world model (transition and reward dictionary). Simulated transitions are prioritized by their predicted TD error, focusing computation where the value function is most out-of-date.

---

## File Architecture

```
Evolving-AI-main/
├── brain.py          — RL engine: Dueling DDQN, PER, N-Step, ICM, Curriculum, Adam
├── world.py          — Environment: maze generation, FoW, traps, portals, A*
├── soul.py           — Personality: emotion model (Russell), OCEAN traits, NLP
├── memory_palace.py  — Persistent knowledge: episodic memory, pattern library
├── analytics.py      — Instrumentation: reward curves, loss tracking, heatmaps
├── RIENFoRZe.py      — Primary entry point and orchestration
├── RIENFoRZeADv.py   — Advanced Streamlit research UI
└── requirements.txt  — Dependency manifest
```

---

## Hyperparameter Reference

| Parameter | Value | Description |
|-----------|-------|-------------|
| `state_size` | 17 | Observation vector dimension |
| `action_size` | 4 | Discrete actions: Up, Down, Left, Right |
| `h1, h2, h3` | 256, 128, 64 | Hidden layer widths |
| `buffer_size` | 50,000 | PER replay buffer capacity |
| `alpha` (PER) | 0.6 | Priority exponent |
| `beta_start` | 0.4 | IS weight annealing start |
| `beta_frames` | 100,000 | IS annealing duration |
| `n_steps` | 3 | N-step return horizon |
| `gamma` | 0.99 | Discount factor |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_min` | 0.04 | Minimum exploration rate |
| `epsilon_decay` | 0.997 | Per-step decay multiplier |
| `tau` | 0.005 | Polyak averaging coefficient |
| `lr` | 0.001 | Adam learning rate |
| `batch_size` | 64 | Training batch size |
| `icm_beta` | 0.05 | Curiosity bonus scale |
| `icm_bins` | 16 | State discretization bins |
| `promote_thresh` | 0.72 | Curriculum promotion threshold |
| `demote_thresh` | 0.25 | Curriculum demotion threshold |
| `curriculum_window` | 20 | Rolling evaluation window |
| `lr_patience` | 100 | Plateau patience for LR reduction |
| `lr_factor` | 0.5 | LR reduction factor |
| `lr_min` | 1e-5 | Minimum learning rate |
| `gradient_clip` | 10.0 | Gradient clipping bound |

---

*RIENFoRZe-I — Foundational Edition — April 2026*
*All derivatives, comparisons, and ablations reference this document as the baseline.*
