
# RIENFoRZe — III
### Project A.L.I.V.E. NEXUS · 64-Dimensional Sensory Architecture · April 2026

> *The apex of the gradient-based lineage. Twelve new sensory dimensions are added to the 52D base: cardinal wall raycasts, logarithmic scent gradients, a geometric target beacon, local pheromone flux, and a live curiosity feedback signal. The Dyna-Q engine is scaled to 25 planning steps with 125-cycle Instant Breakthrough on goal. The curiosity signal closes a self-referential loop — the agent's own novelty drive enters its observation vector.*

---

## Table of Contents

1. [Architectural Divergences from RIENFoRZe-II](#architectural-divergences-from-rienforze-ii)
2. [State Representation — 64-Dimensional Observation Space](#state-representation--64-dimensional-observation-space)
3. [Cardinal Wall Radar — Raycast Sensing](#cardinal-wall-radar--raycast-sensing)
4. [Scent Gradient Channels — Logarithmic Visit Differential](#scent-gradient-channels--logarithmic-visit-differential)
5. [Target Beacon — Unit Direction Encoding](#target-beacon--unit-direction-encoding)
6. [Local Flux — Pheromone Variance Signal](#local-flux--pheromone-variance-signal)
7. [Curiosity Self-Referential Loop](#curiosity-self-referential-loop)
8. [Accelerated Dyna-Q — Instant Breakthrough](#accelerated-dyna-q--instant-breakthrough)
9. [Complete State Vector Derivation](#complete-state-vector-derivation)
10. [Information-Theoretic Analysis of 64D vs 52D](#information-theoretic-analysis-of-64d-vs-52d)
11. [Full Neural Architecture with Updated Input Dimension](#full-neural-architecture-with-updated-input-dimension)
12. [Dueling Head Mathematics](#dueling-head-mathematics)
13. [Backpropagation Through the Dueling Architecture](#backpropagation-through-the-dueling-architecture)
14. [MazE Streamlit Instrumentation](#maze-streamlit-instrumentation)
15. [Hyperparameter Reference](#hyperparameter-reference)

---

## Architectural Divergences from RIENFoRZe-II

RIENFoRZe-III preserves the entire RIENFoRZe-II stack and extends it with five new sensory modalities and an intensified planning regime.

| Component | RIENFoRZe-II | RIENFoRZe-III | Delta |
|---|---|---|---|
| State dimension | 52 | 64 | +12 |
| Wall perception | Implicit in vision | Cardinal raycasts (4D) | New |
| Spatial gradient | None | Scent gradients (4D) | New |
| Goal geometry | Manhattan distance (scalar) | Unit beacon vector (2D) | New |
| Pheromone summary | 13-cell values | + Local flux scalar | New |
| Curiosity in state | Not encoded | Direct ICM signal (1D) | New |
| Planning steps | 5 | 25 | 5× |
| Breakthrough mult | 4× (20 cycles) | 5× (125 cycles) | 6.25× |
| ICM feedback loop | Reward only | Reward + state vector | New |

The 12 new dimensions address three distinct failure modes observed in RIENFoRZe-II:

1. **Tunnel blindness**: The 5×5 vision cannot detect a wall 6 cells away — raycasts solve this.
2. **Gradient invisibility**: The agent cannot compute which direction has less pheromone — scent differentials solve this.
3. **Goal ambiguity**: Distance alone loses geometric orientation — the unit beacon vector restores it.

---

## State Representation — 64-Dimensional Observation Space

### Full Decomposition

```
s ∈ R^64  =  [vision(25) | pheromones(13) | telemetry(10) | momentum(4) | radar(4) | scent(4) | beacon(2) | flux(1) | curiosity(1)]
```

Where `telemetry(10)` is:

```
telemetry = [pos(2) | tpos(2) | dir(2) | dist(1) | trap(1) | fog(1) | time(1)]
```

### Dimension Accounting

| Block | Dims | Source |
|-------|------|--------|
| 5×5 local vision | 25 | Inherited from v-II |
| Pheromone cross | 13 | Inherited from v-II |
| Agent position | 2 | Inherited |
| Target position | 2 | Inherited |
| Direction vector | 2 | Inherited |
| Manhattan dist | 1 | Inherited |
| Trap proximity | 1 | Inherited |
| Fog coverage | 1 | Inherited |
| Time pressure | 1 | Inherited |
| Kinesthetic momentum | 4 | Inherited from v-II |
| **Wall radar** | **4** | **New in v-III** |
| **Scent gradients** | **4** | **New in v-III** |
| **Target beacon** | **2** | **New in v-III** |
| **Pheromone flux** | **1** | **New in v-III** |
| **Curiosity signal** | **1** | **New in v-III** |
| **Total** | **64** | |

---

## Cardinal Wall Radar — Raycast Sensing

The wall radar performs four **raycasts** — one in each cardinal direction (North, South, East, West) — and returns the normalized distance to the first wall or boundary encountered.

### Raycast Algorithm

For each direction (dr, dc) in {(-1,0), (1,0), (0,-1), (0,1)}:

```math
d_{\text{radar}}(dr, dc) = \frac{i^*}{10}, \quad i^* = \min\{i \in \{1,\ldots,10\} : \text{maze}[r + i \cdot dr, c + i \cdot dc] = 1\}
```

where the search is terminated at i = 10 if no wall is found within 10 cells, or at the maze boundary.

The result is normalized to [0.1, 1.0] (nearest wall is 0.1, farthest detectable is 1.0).

### Why Raycasts Complement Local Vision

The 5×5 local vision provides dense coverage within a 2-cell radius but has sharp cutoff at distance 3. Corridors in large mazes (Level 8–10, up to 41 columns) frequently have long straight passages where the agent cannot see the next junction. The radar extends effective perception to 10 cells along each axis, allowing the agent to:

- Distinguish a short dead-end (radar = 0.2) from a long corridor (radar = 0.9) without traversing it
- Anticipate open spaces and adjust planning horizon

### Geometric Interpretation

The four radar values define an implicit bounding box around the agent:

```math
\text{corridor\_length}(\text{dir}) = 10 \cdot d_{\text{radar}}(\text{dir})
```

The **aspect ratio** of this box — long in the current direction versus short perpendicular — is implicitly available to the network as a feature.

---

## Scent Gradient Channels — Logarithmic Visit Differential

The scent gradient encodes the **derivative of the visit-count field** in each cardinal direction. This allows the agent to compute a local gradient of its own historical exploration density.

### Computation

Let `visit_grid[r, c]` be the accumulated visit count at cell (r, c). The scent gradient in direction (dr, dc) is:

```math
g_{\text{scent}}(dr, dc) = \text{clip}\!\left(\log(1 + \text{visit}[r + dr, c + dc]) - \log(1 + \text{visit}[r, c]),\ -1,\ 1\right)
```

Equivalently:

```math
g_{\text{scent}}(dr, dc) = \text{clip}\!\left(\log\!\left(\frac{1 + \text{visit}[r + dr, c + dc]}{1 + \text{visit}[r, c]}\right),\ -1,\ 1\right)
```

### Logarithmic Rationale

Raw visit counts grow unboundedly and are heavily right-skewed (frequently visited cells can accumulate thousands of visits while novel cells have 0–5). The log transformation compresses this dynamic range, and the difference of logs produces a **log-ratio** signal:

- Positive scent gradient: the neighbor has been visited more → less novel, potentially avoid
- Negative scent gradient: the neighbor has been visited less → more novel, potentially explore
- Zero: equal exploration history in both directions

The clipping to [-1, 1] prevents occasional large gradients from dominating the network input.

---

## Target Beacon — Unit Direction Encoding

RIENFoRZe-I and II encode the **scalar distance** to the target. RIENFoRZe-III replaces the scalar with a **unit direction vector** pointing precisely toward the target.

### Encoding

```math
\vec{u}_{\text{beacon}} = \frac{(r_{\text{target}} - r_{\text{agent}},\ c_{\text{target}} - c_{\text{agent}})}{\|(r_{\text{target}} - r_{\text{agent}},\ c_{\text{target}} - c_{\text{agent}})\|_2 + \epsilon}
```

where epsilon = 1e-9 prevents division by zero at the goal.

### Information Content

The scalar Manhattan distance encodes magnitude (how far) but loses direction (which way). The unit beacon vector encodes direction (which quadrant) but loses magnitude. Both are retained: distance is preserved as part of the `telemetry` block, beacon is added as a new 2D block.

Together they provide the network with:

```math
\text{goal\_info} = (d_{\text{manhattan}},\ u_r,\ u_c)
```

This 3D representation spans more of the relevant goal-geometry space than either alone.

### Rotation Invariance

The unit vector is not rotation-invariant (the absolute frame matters for navigation in a fixed maze), but it is scale-invariant: a target 5 cells away and a target 50 cells away in the same direction produce identical beacon vectors, which is desirable since the distance is captured separately.

---

## Local Flux — Pheromone Variance Signal

The 13-cell pheromone cross already encodes local visit density. The **flux** scalar summarizes the statistical spread of these 13 values:

```math
\text{flux} = \text{std}\!\left(P_{\text{norm}}[c_0], P_{\text{norm}}[c_1], \ldots, P_{\text{norm}}[c_{12}]\right)
```

### Interpretation

- **High flux**: Pheromone distribution is uneven — some nearby cells are heavily visited, others are not. The agent is near a frontier between explored and unexplored territory.
- **Low flux**: Pheromone distribution is uniform — either the entire local neighborhood is equally explored or equally unexplored.

High-flux states are informationally richer for exploration decisions. The network can learn to treat high-flux states as requiring more deliberate action selection, and low-flux states as routine navigation.

---

## Curiosity Self-Referential Loop

This is the most structurally novel feature of RIENFoRZe-III. In all prior versions, the ICM bonus r_i(s) influences only the **reward signal**. In RIENFoRZe-III, the ICM bonus is also **directly encoded into the state vector** as the final dimension.

### Mechanism

The last computed ICM bonus is cached in the environment:

```python
self.last_icm_bonus = intrinsic_reward  # updated at each step
```

This value is then included in the observation:

```math
s_{64} = \text{clip}(\text{last\_icm\_bonus}, 0, 1)
```

### Implications

This creates a **metacognitive feedback loop**: the agent can observe its own curiosity level as part of its world state. A high curiosity value (novel state) encoded in s_{64} provides a direct signal that the agent can condition its action selection on.

This is structurally related to **meta-reinforcement learning**: the agent learns not just a policy over environment states, but a policy that takes into account its own internal learning signal. If the network learns to use s_{64} effectively, it can produce curiosity-seeking behavior that is directly derived from observing its own novelty drive rather than from the reward shaping alone.

### Mathematical Closed Loop

Define the curiosity signal as a function of the current state's novelty:

```math
\text{icm}(s_t) = \frac{\beta}{\sqrt{N(k(s_t))}}
```

The next state embedding includes this value:

```math
s_{t+1} = [\ldots,\ \text{icm}(s_t)]
```

So the agent's policy at t+1 is:

```math
\pi(a | s_{t+1}) = \pi(a | [\ldots,\ \text{icm}(s_t)])
```

The agent learns to anticipate that high curiosity at t implies it is in a novel region and can exploit this information directly in action selection, rather than only receiving it as an elevated reward.

---

## Accelerated Dyna-Q — Instant Breakthrough

### Planning Intensity

RIENFoRZe-III dramatically increases the Dyna-Q planning budget:

| Condition | Planning Steps |
|---|---|
| Normal step | 25 |
| Episode success (done=True, r > 20.0) | 25 × 5 = **125** |

The success threshold is tightened: reward > 20.0 (vs > 10.0 in v-II), ensuring the breakthrough trigger fires only on clean goal-reaches, not on partial reward accumulation.

### Effective Learning Multiplier

Per environment step:

```math
K_{\text{eff}}^{\text{normal}} = 1 + 25 = 26
```

```math
K_{\text{eff}}^{\text{breakthrough}} = 1 + 125 = 126
```

Per episode of T steps with one successful termination:

```math
K_{\text{eff}}^{\text{episode}} = T \cdot 26 + 125 = 26T + 125
```

For an episode of 200 steps: 26 × 200 + 125 = 5,325 gradient updates from a single 200-step real episode.

### Value Propagation Speed

With standard Q-learning, value information propagates backward at one cell per episode. With 25-step Dyna-Q planning, value information can propagate up to 25 steps backward in a single planning phase. For a 35×41 Level-10 maze (optimal path ≈ 70 steps), full value propagation requires at minimum 3 real episodes with 25-step planning vs approximately 70 without.

---

## Complete State Vector Derivation

The full 64-dimensional state vector is assembled in the following precise order. This order is fixed and must be consistent between training and inference.

```
Index   Block            Formula
─────────────────────────────────────────────────────────────────────────
0–24    5×5 vision       v_{dr,dc} ∈ {0.0, 0.5, 1.0}, dr,dc ∈ {-2..2}
25–37   Pheromone cross  P_norm[center + offset] for 13 cross positions
38–39   Agent pos        (r/H, c/W)
40–41   Target pos       (r_t/H, c_t/W)
42–43   Direction vec    (delta_r, delta_c) of last step
44      Manhattan dist   (|r-r_t| + |c-c_t|) / (H+W)
45      Trap proximity   min_trap_manhattan / (H+W)
46      Fog coverage     |visible| / (H*W)
47      Time pressure    step_count / max_steps
48–51   Momentum         one-hot(last_action) ∈ R^4
52–55   Wall radar       d_radar(N,S,E,W) / 10
56–59   Scent grads      clip(log(1+v_neighbor) - log(1+v_self), -1, 1)
60–61   Beacon           (r_t - r, c_t - c) / (||...|| + 1e-9)
62      Pheromone flux   std(pheromone_cross_13_values)
63      Curiosity signal clip(last_icm_bonus, 0, 1)
─────────────────────────────────────────────────────────────────────────
Total: 64 dimensions
```

---

## Information-Theoretic Analysis of 64D vs 52D

### Mutual Information Argument

Each new dimension contributes information to the extent that it reduces uncertainty about the optimal action. Denote the optimal action as A* and the new feature as X. The marginal value of X is:

```math
I(A^*; X \mid s_{1:52}) = H(A^* \mid s_{1:52}) - H(A^* \mid s_{1:52}, X)
```

Features are useful if this quantity is positive — i.e., if knowing X reduces uncertainty about the best action beyond what the 52D vector already provides.

### Redundancy Analysis

- **Beacon vs Manhattan distance**: Manhattan distance is scalar (magnitude only). Beacon provides the unit direction vector (direction only). These are nearly orthogonal in information content, since they capture complementary aspects of goal geometry.
- **Radar vs vision**: Vision provides dense 2-step coverage. Radar provides sparse 10-step coverage along 4 axes. Radar catches long corridors invisible to the vision window.
- **Scent vs pheromone cross**: Pheromone cross provides raw density values at 13 locations. Scent provides the **gradient** (rate of change). The gradient is not recoverable from the absolute values alone without knowing the agent's current cell's count — which changes each step.
- **Curiosity vs ICM reward**: The reward is a scalar summed into the episode return. The curiosity observation is available to condition the policy at each step, before the reward is accumulated.

None of the 12 new dimensions are linearly predictable from the existing 52, justifying their inclusion as non-redundant.

---

## Full Neural Architecture with Updated Input Dimension

```
Input (64) → W1 (64×256) + b1 → LeakyReLU → h1 (256)
           → W2 (256×128) + b2 → LeakyReLU → h2 (128)
           → W3 (128×64)  + b3 → LeakyReLU → h3 (64)
           → W_val (64×1) + b_val → V(s) ∈ R
           → W_adv (64×4) + b_adv → A(s,a) ∈ R^4
           → Q(s,a) = V(s) + A(s,a) - (1/4) * sum_a A(s,a)
```

**Parameter count:**

```math
\text{Params} = (64 \times 256 + 256) + (256 \times 128 + 128) + (128 \times 64 + 64) + (64 \times 1 + 1) + (64 \times 4 + 4)
```

```math
= 16,640 + 32,896 + 8,256 + 65 + 260 = 58,117 \text{ parameters}
```

(vs 55,049 for the 52D version — a modest 5.6% increase for 23% more observation richness)

---

## Dueling Head Mathematics

### Forward Pass

Given shared representation h3 ∈ R^64:

```math
V(s) = h_3 W_{\text{val}} + b_{\text{val}} \in \mathbb{R}
```

```math
A(s, \cdot) = h_3 W_{\text{adv}} + b_{\text{adv}} \in \mathbb{R}^4
```

```math
Q(s, a) = V(s) + A(s, a) - \frac{1}{4} \sum_{a'=0}^{3} A(s, a')
```

### Backward Pass — Dueling Gradient Routing

Let dQ ∈ R^(B×4) be the loss gradient with respect to the Q-output. The gradients through the dueling heads are:

```math
\frac{\partial \mathcal{L}}{\partial V(s_i)} = \sum_{a} \frac{\partial \mathcal{L}}{\partial Q(s_i, a)} \cdot \frac{\partial Q}{\partial V} = \sum_{a} dQ_{i,a}
```

```math
\frac{\partial \mathcal{L}}{\partial A(s_i, a)} = dQ_{i,a} - \frac{1}{4} \sum_{a'} dQ_{i,a'} = dQ_{i,a} - \bar{dQ}_i
```

These two gradients route through W_val and W_adv respectively and then sum at h3:

```math
\frac{\partial \mathcal{L}}{\partial h_3} = \frac{\partial \mathcal{L}}{\partial V} W_{\text{val}}^T + \frac{\partial \mathcal{L}}{\partial A} W_{\text{adv}}^T
```

The gradient routing enforces that V(s) is updated by the sum signal (total action value) while A(s,a) is updated by the deviation signal (relative action advantage), maintaining the interpretability of the dueling decomposition throughout training.

---

## Backpropagation Through the Dueling Architecture

Full gradient derivation for one training step, batch size B:

### Layer-by-layer

```math
d_{h3} = d_V W_{\text{val}}^T + d_A W_{\text{adv}}^T
```

```math
d_{z3} = d_{h3} \odot f'_{\text{leaky}}(z_3)
```

```math
dW_3 = h_2^T d_{z3} / B, \quad db_3 = \text{mean}(d_{z3}, \text{axis}=0)
```

```math
d_{h2} = d_{z3} W_3^T, \quad d_{z2} = d_{h2} \odot f'_{\text{leaky}}(z_2)
```

```math
dW_2 = h_1^T d_{z2} / B, \quad db_2 = \text{mean}(d_{z2}, \text{axis}=0)
```

```math
d_{h1} = d_{z2} W_2^T, \quad d_{z1} = d_{h1} \odot f'_{\text{leaky}}(z_1)
```

```math
dW_1 = x^T d_{z1} / B, \quad db_1 = \text{mean}(d_{z1}, \text{axis}=0)
```

All gradients are clipped to [-10, 10] element-wise before the Adam update. After the Adam update, all weight matrices are clamped to [-100, 100].

---

## MazE Streamlit Instrumentation

The `MazE.py` companion app provides the following real-time scientific instrumentation panels:

| Panel | Metric | Description |
|-------|--------|-------------|
| Policy Readout | Q(s, a) vector | Live bar chart of action values at current state |
| Exploration | H(pi) entropy | Policy entropy timeline quantifies exploration-exploitation |
| Reward decomp | r_ext vs r_int | Separation of external and ICM-generated reward |
| Dueling heads | V(s), max A(s,a) | Streaming value and advantage over time |
| Stability | grad_norm proxy | Estimated gradient magnitude as training signal |
| Trajectory | ASCII maze | Agent path overlaid on maze with step history |
| Convergence | t-test on reward | Hypothesis test on positive reward gradient |
| LR tracking | LR vs plateau | Adaptive LR scheduler vs reward plateau detector |
| Bellman | Residual stream | Separate MSE loss vs Bellman residual tracking |
| Actions | U/D/L/R histogram | Distribution of action selections across training |
| Comparison | best/worst/current | Episode performance comparison panel |
| Research | Equations, theory | Full derivation reference tab |

---

## Hyperparameter Reference

| Parameter | Value | Note |
|-----------|-------|------|
| `state_size` | 64 | +12 from v-II |
| `action_size` | 4 | Same |
| `h1, h2, h3` | 256, 128, 64 | Same |
| `buffer_size` | 50,000 | Same |
| `alpha` (PER) | 0.6 | Same |
| `gamma` | 0.99 | Same |
| `epsilon_start` | 0.7 | Same as v-II |
| `epsilon_min` | 0.05 | Same |
| `epsilon_decay` | 0.995 | Per episode |
| `tau` | 0.005 | Same |
| `lr` | 0.001 | Same |
| `batch_size` | 64 | Same |
| `planning_steps` | 25 | 5× from v-II |
| `breakthrough_mult` | 5× (125 total) | "Instant Breakthrough" |
| `breakthrough_threshold` | reward > 20.0 | Tighter than v-II (>10.0) |
| `radar_max_range` | 10 cells | Raycast limit |
| `scent_clip` | ±1.0 | Log-ratio clipping |
| `beacon_epsilon` | 1e-9 | Division safety floor |
| `flux_metric` | std of 13-pheromone cross | Local variance signal |
| `curiosity_encoding` | clip(last\_icm\_bonus, 0, 1) | State dimension 63 |
| `loss_function` | Huber (delta=1.0) | Same as v-II |
| `q_clip` | ±1,000,000 | Same |
| `weight_clip` | ±100.0 | Same |
| `total_parameters` | 58,117 | Neural network only |

---

*RIENFoRZe-III — 64-Dimensional Sensory Edition — April 2026*
*Apex of the gradient-based architecture family. Successor to RIENFoRZe-II. Parallel to RIENFoRZe-IV.*
