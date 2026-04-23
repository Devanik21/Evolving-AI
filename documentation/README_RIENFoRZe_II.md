# RIENFoRZe — II
### Project A.L.I.V.E. NEXUS · 52-Dimensional Advanced Architecture · April 2026

> *The first major revision. The observation space triples in richness. The loss function becomes robust. The epsilon schedule is corrected. A Dyna-Q planning loop supplements real experience. Numerical stability becomes a first-class concern. This version is designed to expose the limits of gradient-based tabular augmentation before the full departure in RIENFoRZe-IV.*

---

## Table of Contents

1. [Architectural Divergences from RIENFoRZe-I](#architectural-divergences-from-rienforze-i)
2. [State Representation — 52-Dimensional Observation Space](#state-representation--52-dimensional-observation-space)
3. [Pheromone Channel Engineering](#pheromone-channel-engineering)
4. [Kinesthetic Momentum Encoding](#kinesthetic-momentum-encoding)
5. [Dyna-Q Hallucination Planning](#dyna-q-hallucination-planning)
6. [Huber Loss — Motivation and Derivation](#huber-loss--motivation-and-derivation)
7. [Numerical Stability Shields](#numerical-stability-shields)
8. [Epsilon Schedule Correction](#epsilon-schedule-correction)
9. [Super Brain Mode — Breakthrough Learning](#super-brain-mode--breakthrough-learning)
10. [MazE Module — SARSA with Prioritized Sweeping](#maze-module--sarsa-with-prioritized-sweeping)
11. [Fog-of-War Vision System](#fog-of-war-vision-system)
12. [Neural Architecture Unchanged](#neural-architecture-unchanged)
13. [Reward Engineering](#reward-engineering)
14. [Complete Hyperparameter Reference](#complete-hyperparameter-reference)
15. [Ablation Map](#ablation-map)

---

## Architectural Divergences from RIENFoRZe-I

RIENFoRZe-II introduces five independent modifications to the baseline. Each is documented separately to facilitate ablation analysis.

| Modification | RIENFoRZe-I | RIENFoRZe-II |
|---|---|---|
| State dimensionality | 17D | 52D |
| Loss function | MSE | Huber |
| Dyna-Q planning | None | 5 steps/transition |
| Epsilon initialization | 1.0 | 0.7 |
| Epsilon decay timing | Per environment step | Per episode (corrected) |
| Numerical shield | None | Weight clamp ±100, Q clamp ±1e6 |
| Breakthrough multiplier | None | 4x on episode success |

None of these changes alter the fundamental DQN objective. They address stability, sample efficiency, and observation richness.

---

## State Representation — 52-Dimensional Observation Space

The 17D observation of RIENFoRZe-I is extended to **52 dimensions** by expanding the local vision field, adding a pheromone trail channel, and encoding kinesthetic momentum.

### Full Decomposition

```
s ∈ R^52  =  [vision(25) | pheromones(13) | pos(2) | tpos(2) | dir(2) | dist(1) | trap(1) | fog(1) | time(1) | momentum(4)]
```

| Block | Dims | Content |
|-------|------|---------|
| Local vision | 25 | 5×5 neighborhood, wall/path/fog encoded |
| Pheromone map | 13 | Visit-count heatmap over 13-cell cross pattern |
| Agent position | 2 | Normalized (r/H, c/W) |
| Target position | 2 | Normalized (r/H, c/W) |
| Direction vector | 2 | Unit vector of last step direction |
| Manhattan distance | 1 | Normalized to (H+W) |
| Trap proximity | 1 | Normalized to (H+W) |
| Fog coverage | 1 | Fraction of maze revealed |
| Time pressure | 1 | t/t\_max |
| Momentum | 4 | One-hot encoding of last action |

**Total: 25 + 13 + 2 + 2 + 2 + 1 + 1 + 1 + 1 + 4 = 52**

### Vision Field Expansion: 3×3 → 5×5

RIENFoRZe-I used a 3×3 (9-cell) local view. RIENFoRZe-II expands to a **5×5 (25-cell)** view, extending the perceivable radius by one additional step in each direction. This allows the agent to see two cells ahead in any direction, which is critical for junction anticipation in branching Prim mazes.

```math
v_{dr,dc} = \begin{cases}
0.0 & \text{passable and visible} \\
1.0 & \text{wall or out-of-bounds} \\
0.5 & \text{within fog, unexplored}
\end{cases}
```

where (dr, dc) ranges over {-2, -1, 0, 1, 2} x {-2, -1, 0, 1, 2}.

---

## Pheromone Channel Engineering

The pheromone system encodes the agent's own historical visit density as a spatial signal, enabling the agent to reason about which regions have been explored. This is an instance of **stigmergic communication** — indirect coordination through environmental modification.

### Pheromone Grid

A floating-point grid `pheromone_grid[r, c]` accumulates visit counts normalized into [0, 1]:

```math
P[r, c] \leftarrow P[r, c] + 1
```

```math
P_{\text{norm}} = \frac{P[r,c]}{\max_{r',c'} P[r',c'] + \epsilon}
```

### 13-Cell Cross Pattern

Rather than encoding a full 2D subgrid, the pheromone observation samples 13 specific cells arranged in a cross-plus-corners pattern centered on the agent:

```
    [ ]                         row - 2
    [ ]                         row - 1
[ ] [X] [ ] [ ] [ ]             row (cardinal + arm)
    [ ]                         row + 1
    [ ]                         row + 2
```

Specifically: center, 4 cardinal neighbors, 4 diagonal neighbors, and 4 extended cardinal positions (range 2). This captures local trail density with 13 values rather than 25, preserving sensitivity to exploration gradients while reducing dimensionality.

### Pheromone Gradient Interpretation

The agent can implicitly compute a pheromone gradient from these 13 values. If cardinal pheromones decrease in direction d, that direction is less explored. Over training, the agent learns to use this gradient to bias exploration toward novel regions — emergent behavior from the reward signal.

---

## Kinesthetic Momentum Encoding

The direction vector and one-hot momentum are two distinct encodings of recent action history.

### Direction Vector (2D continuous)

```math
\vec{d} = (\delta r, \delta c) \in \{(-1,0),(1,0),(0,-1),(0,1)\}
```

Normalized to unit vector. Encodes the continuous geometric direction of the last step.

### Momentum One-Hot (4D)

```math
m_i = \begin{cases} 1.0 & \text{if last\_action} = i \\ 0.0 & \text{otherwise} \end{cases}, \quad i \in \{0,1,2,3\}
```

Together these two encodings give the network both a geometric direction signal and a categorical action identity signal. The redundancy is intentional: different network layers may find different representations more useful, and the two encodings interact differently with the dueling head.

---

## Dyna-Q Hallucination Planning

RIENFoRZe-II introduces **Dyna-Q** (Sutton, 1991) as a sample amplification mechanism. After each real environmental step, the agent performs K = 5 additional simulated updates using a stored world model.

### World Model

The world model is a dictionary:

```
M : (s, a) → (s', r)
```

At each real step (s, a, r', s'), the model is updated:

```math
M(s, a) \leftarrow (s', r_{\text{aug}})
```

This constitutes a **perfect model** assumption: the model stores the exact observed transition, not a learned approximation. This is valid because the maze environment is deterministic (given the same s and a, the same s' is always produced).

### Planning Loop

After the real update, K simulated updates are performed by sampling uniformly from the model:

```math
\text{for } k = 1, \ldots, K:
```

```math
(s_{\text{sim}}, a_{\text{sim}}) \sim \text{Uniform}(\text{dom}(M))
```

```math
(s'_{\text{sim}}, r_{\text{sim}}) \leftarrow M(s_{\text{sim}}, a_{\text{sim}})
```

```math
y_{\text{sim}} = r_{\text{sim}} + \gamma (1 - d) \max_{a'} Q(s'_{\text{sim}}, a'; \theta)
```

```math
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_{\text{Huber}}(y_{\text{sim}}, Q(s_{\text{sim}}, a_{\text{sim}}; \theta))
```

This effectively multiplies the number of gradient updates per environment step by K, improving sample efficiency substantially.

### Effective Sample Ratio

Per real environment step, the agent receives:

```math
K_{\text{eff}} = 1 + K_{\text{planning}} = 1 + 5 = 6 \quad \text{(normal)}
```

```math
K_{\text{eff}} = 1 + 4K = 1 + 20 = 21 \quad \text{(Super Brain Mode, on success)}
```

This ratio grows the effective dataset seen per real step from 1× to 6×–21×, significantly accelerating value propagation.

---

## Huber Loss — Motivation and Derivation

RIENFoRZe-I uses MSE loss. RIENFoRZe-II replaces it with **Huber loss** (also called smooth L1 loss), which is more robust to large TD errors in the early training phase.

### Definition

```math
\mathcal{L}_{\delta}(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & |y - \hat{y}| \leq \delta \\
\delta \cdot |y - \hat{y}| - \frac{1}{2}\delta^2 & |y - \hat{y}| > \delta
\end{cases}
```

This experiment uses delta = 1.0.

### Gradient

```math
\frac{\partial \mathcal{L}_\delta}{\partial \hat{y}} = \begin{cases}
\hat{y} - y & |y - \hat{y}| \leq \delta \\
-\delta \cdot \text{sign}(y - \hat{y}) & |y - \hat{y}| > \delta
\end{cases}
```

### Why Huber Over MSE

MSE loss produces gradients proportional to the TD error:

```math
\frac{\partial}{\partial \hat{y}} (y - \hat{y})^2 = -2(y - \hat{y})
```

For large TD errors (common early in training when Q-values are poorly initialized), this produces very large gradient magnitudes, even after clipping. Huber loss caps the gradient magnitude at delta, providing bounded gradients regardless of error size. This is equivalent to a switch between L2 regression (for small errors, where precision matters) and L1 regression (for large errors, where outlier robustness matters).

---

## Numerical Stability Shields

RIENFoRZe-I experienced numerical overflow during extended training runs (gradient explosion through compounding). RIENFoRZe-II introduces two hard clamps to prevent this.

### Q-Value Shield

After forward pass, Q-values are clipped before being used in any computation:

```math
Q(s, a) \leftarrow \text{clip}(Q(s, a), -10^6, +10^6)
```

This prevents the Bellman target computation from producing NaN or Inf values when multiplied by gamma.

### Weight Shield

After each Adam update, all weight tensors are hard-clamped:

```math
\theta_p \leftarrow \text{clip}(\theta_p, -100.0, +100.0)
```

This prevents individual weights from diverging over long training runs. The chosen bound of ±100 is intentionally generous — it intervenes only in genuine overflow scenarios, not in normal training dynamics.

### Shape Validation on Weight Restore

When loading saved weights (from JSON or ZIP), each restored tensor is validated against the current network's expected shape before assignment:

```math
\text{load}(W_p) \iff \text{shape}(W_p^{\text{saved}}) = \text{shape}(W_p^{\text{current}})
```

Shape mismatch is reported to stderr and the parameter is skipped rather than corrupting the network state.

---

## Epsilon Schedule Correction

RIENFoRZe-I contained a **bug** in epsilon decay: the decay was applied once per training step (i.e., every time a batch was sampled from the replay buffer). Because multiple training steps can occur per episode (one per environment step after the buffer fills), epsilon decayed faster than intended.

### RIENFoRZe-II Fix

Epsilon is now decayed **once per episode** — after `done=True`:

```math
\epsilon_{t+1} = \max(\epsilon_{\text{min}},\ \epsilon_t \cdot \lambda_\epsilon)
```

This is applied exactly once at episode termination, guaranteeing the schedule matches the intended per-episode semantics.

### Schedule Parameters

| Parameter | RIENFoRZe-I | RIENFoRZe-II |
|---|---|---|
| Initial epsilon | 1.0 | 0.7 |
| Minimum epsilon | 0.04 | 0.05 |
| Decay rate | 0.997 | 0.995 |
| Decay timing | Per training step | Per episode |

The warm start (0.7 vs 1.0) reduces the initial purely-random exploration phase, which is inefficient once the agent has a small amount of experience. The per-episode semantics make the exploration schedule far more interpretable.

### Theoretical Schedule

With per-episode decay at rate 0.995 from 0.7 to 0.05:

```math
\epsilon_n = \max\!\left(0.05, \ 0.7 \cdot 0.995^n\right)
```

Reaches minimum after:

```math
n^* = \left\lceil \frac{\ln(0.05 / 0.7)}{\ln(0.995)} \right\rceil = \left\lceil \frac{-2.639}{-0.00501} \right\rceil = \lceil 526 \rceil = 527 \text{ episodes}
```

---

## Super Brain Mode — Breakthrough Learning

When the agent successfully reaches the goal (reward > 10.0) at episode termination, the Dyna-Q planning multiplier is activated:

```math
K_{\text{actual}} = \begin{cases}
K & \text{normal episode} \\
4K & \text{if done and episode\_reward} > 10.0
\end{cases}
```

For RIENFoRZe-II with K = 5:

```math
K_{\text{breakthrough}} = 4 \times 5 = 20 \text{ planning cycles}
```

### Motivation

A goal-reaching episode contains highly valuable information: a complete success trajectory. Replaying from this trajectory's memorized model updates intensifies the value propagation from the terminal reward backward through the state-action chain. This is inspired by the phenomenon of **replay consolidation** in biological memory systems, where high-salience experiences receive elevated replay during quiescent periods.

---

## MazE Module — SARSA with Prioritized Sweeping

The standalone `MazE.py` application uses a fundamentally different algorithm: **SARSA (State-Action-Reward-State-Action)** with an integrated prioritized sweeping world model. This is independent from the main DQN agent.

### SARSA Update

SARSA is an on-policy TD(0) control algorithm:

```math
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]
```

where a_{t+1} is the **actually selected** next action (not the greedy maximum). This makes SARSA's policy estimate more conservative than Q-learning's near high-cost states.

### BFS Distance Map

At initialization, BFS from the goal constructs an exact shortest-path distance map D over all passable cells. This serves as a dense reward shaping potential:

```math
r_{\text{shaped}}(s, s') = r(s, s') + \gamma \cdot (-D(s')) - (-D(s)) = r(s, s') + \gamma D(s) - D(s')
```

The shaped reward is dense everywhere, eliminating the sparse reward problem in large mazes.

### Curiosity Integration

Visit counts are maintained per cell. Curiosity weight decays exponentially over training:

```math
r_{\text{curiosity}}(s) = \frac{w_c}{\text{visit}(s) + 1}, \quad w_c \leftarrow 0.99 \cdot w_c
```

Initial curiosity weight w_c = 1.0. After approximately 459 episodes, w_c < 0.01, effectively disabling intrinsic exploration pressure.

### Prioritized Sweeping

The world model M(s, a) = (s', r) stores observed transitions. After each real step, simulated updates are sampled in order of predicted TD error magnitude, focusing compute on the highest-leverage states.

---

## Fog-of-War Vision System

The `FogOfWar` class maintains a binary visibility grid. A cell (r, c) becomes visible when the agent is within a Chebyshev-ball of radius R centered on the agent's current position:

```math
\text{visible}[r', c'] = \text{True} \iff \max(|r' - r|, |c' - c|) \leq R
```

where R = max(3, min(6, H//4)) adapts to maze height. Visibility is monotonically cumulative — once revealed, a cell remains visible. The fog coverage signal:

```math
\phi_{\text{fog}} = \frac{|\{(r,c) : \text{visible}[r,c]\}|}{H \cdot W} \in [0, 1]
```

encodes the fraction of the maze explored. This value increases monotonically within an episode.

---

## Neural Architecture Unchanged

The Dueling DQN network architecture is identical to RIENFoRZe-I, with the single modification that the input layer now accepts 52-dimensional vectors:

```
Input (52) → Linear → (256) → LeakyReLU
           → Linear → (128) → LeakyReLU
           → Linear → (64)  → LeakyReLU
           → Value stream:     (64) → (1)
           → Advantage stream: (64) → (4)
           → Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]
```

He initialization, Adam optimizer, soft target update, PER, and N-step returns all remain identical to RIENFoRZe-I.

---

## Reward Engineering

Reward structure is identical to RIENFoRZe-I with identical coefficients. The augmented reward including the ICM bonus is:

```math
r_{\text{total}} = \text{clip}(r_{\text{ext}}, -10, 15) + \frac{0.05}{\sqrt{N(s)}}
```

The curiosity bonus is always added before storing in the N-step buffer and world model.

---

## Complete Hyperparameter Reference

| Parameter | Value | Change from I |
|-----------|-------|---------------|
| `state_size` | 52 | +35 dimensions |
| `action_size` | 4 | Same |
| `h1, h2, h3` | 256, 128, 64 | Same |
| `buffer_size` | 50,000 | Same |
| `alpha` (PER) | 0.6 | Same |
| `beta_start` | 0.4 | Same |
| `n_steps` | 3 | Same |
| `gamma` | 0.99 | Same |
| `epsilon_start` | 0.7 | Changed (was 1.0) |
| `epsilon_min` | 0.05 | Changed (was 0.04) |
| `epsilon_decay` | 0.995 | Changed (was 0.997) |
| `epsilon_timing` | Per episode | Fixed (was per step) |
| `tau` | 0.005 | Same |
| `lr` | 0.001 | Same |
| `batch_size` | 64 | Same |
| `planning_steps` | 5 | New |
| `breakthrough_mult` | 4x | New |
| `loss_function` | Huber (delta=1.0) | Changed (was MSE) |
| `q_clip` | ±1,000,000 | New |
| `weight_clip` | ±100.0 | New |
| `gradient_clip` | 10.0 | Same |
| `icm_beta` | 0.05 | Same |
| `vision_radius` | 2 (5×5) | Changed (was 1, 3×3) |
| `pheromone_bins` | 13 cells | New |

---

## Ablation Map

The following table maps each RIENFoRZe-II component to the question it addresses:

| Component | Research Question |
|-----------|------------------|
| 52D state (vs 17D) | Does richer observation improve policy quality or slow convergence? |
| Pheromone channel | Does stigmergic self-history reduce revisiting and improve coverage? |
| Dyna-Q (K=5) | Does model-based planning improve sample efficiency at 5:1 ratio? |
| Huber loss | Does robust loss improve stability in the high-TD-error early phase? |
| Epsilon timing fix | Does correct per-episode decay change the effective exploration curve? |
| Warm start (0.7) | Does reducing random burn-in accelerate useful early learning? |
| Numerical shields | Are weight/Q clamps necessary for stability in long runs? |
| Super Brain Mode | Does goal-contingent planning acceleration improve success rates? |

---

## Learning Rate Scheduler — Plateau Detection

The `LRScheduler` class monitors the rolling average reward and reduces the learning rate when a plateau is detected. This is an adaptive meta-learning mechanism operating on top of the Adam optimizer.

### Plateau Criterion

The best-ever average reward is tracked. If the current average reward does not improve by more than a tolerance threshold for `patience` consecutive episodes, the learning rate is halved:

```math
\eta_{t+1} = \begin{cases}
\eta_t & \text{if } \bar{r}_t > \bar{r}^* + 10^{-4} \\
\max(\eta_{\min}, \eta_t \cdot 0.5) & \text{if wait} \geq \text{patience}
\end{cases}
```

where wait counts consecutive non-improving episodes. Parameters: patience = 100, factor = 0.5, min\_lr = 1e-5.

### Interaction with Dyna-Q

Because Dyna-Q planning steps run at a fixed learning rate (same as the real-step rate), the LR reduction affects both real and simulated updates simultaneously. This can be beneficial: when the agent is stuck on a plateau, reducing the LR for both real and simulated steps may allow it to converge toward a local optimum without oscillating around it.

---

## N-Step Returns — Retained from RIENFoRZe-I

The N-step buffer is retained in RIENFoRZe-II. The N-step return augments the single-step Bellman target with a 3-step lookahead:

```math
G_t^{(3)} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 \max_{a'} Q_{\text{target}}(s_{t+3}, a')
```

N-step returns reduce the variance of the TD estimate relative to TD(0), at the cost of slightly increased bias when the bootstrapped Q-value is inaccurate. With Dyna-Q planning improving Q-value accuracy rapidly, the bootstrapping bias of 3-step returns decreases faster than in RIENFoRZe-I.

The N-step buffer and Dyna-Q planning operate on complementary timescales: N-step returns provide richer credit assignment within a single trajectory, while Dyna-Q propagates value information backward through the stored model.

---

## Segment Tree PER — Unchanged from RIENFoRZe-I

The Prioritized Experience Replay segment tree implementation is identical to RIENFoRZe-I. The key interaction in RIENFoRZe-II: Dyna-Q planning steps do **not** add transitions to the PER buffer. Only real environment steps (via the N-step buffer) contribute to PER. Simulated transitions are sampled directly from the world model and update the neural network, bypassing PER entirely.

This separation is intentional: PER is designed to prioritize real observations by temporal-difference error. Simulated observations (from a perfect deterministic model) would all have low TD error after the first visit, and would crowd out high-priority real transitions.

---

*RIENFoRZe-II — 52-Dimensional Advanced Edition — April 2026*
*Successor to RIENFoRZe-I. Predecessor to RIENFoRZe-III.*


---

## Attribution

**Author:** Devanik (GitHub: [Devanik21](https://github.com/Devanik21))
**Repository:** [Evolving-AI](https://github.com/Devanik21/Evolving-AI) · Project A.L.I.V.E. NEXUS · April 2026
**Affiliation:** Electronics & Communication Engineering, NIT Agartala (2026) · Samsung ISWDP Fellow, IISc (98.58th percentile)

This document and all associated source code are the original work of Devanik. The RIENFoRZe series (versions I–IV), the A.L.I.V.E. NEXUS cognitive architecture, and all mathematical formulations, experimental designs, and implementation decisions documented here were conceived and developed independently.

If you build on, reference, or adapt any part of this work, please cite the original repository:

```
Devanik. Project A.L.I.V.E. NEXUS — RIENFoRZe Series. GitHub, April 2026.
https://github.com/Devanik21/Evolving-AI
```

Licensed under the **Apache License 2.0** — free to use, modify, and distribute with attribution. See `LICENSE` for full terms.
