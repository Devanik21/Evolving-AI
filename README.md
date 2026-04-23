# Project A.L.I.V.E. NEXUS — Master Reference

### Autonomous Learning Intelligent Virtual Entity · RIENFoRZe Series · April 2026

> *A research platform studying reinforcement learning agents that learn to navigate procedurally generated environments while maintaining an evolving cognitive, emotional, and memory architecture — all implemented in pure NumPy, without any ML framework.*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![RL](https://img.shields.io/badge/Algorithm-Dueling%20DDQN%20%2B%20PER%20%2B%20Dyna--Q-green.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Versions](https://img.shields.io/badge/Versions-RIENFoRZe%20I–IV-orange.svg)]()

---

## About This Repository

**Author:** Devanik (GitHub: [Devanik21](https://github.com/Devanik21))  
**Repository:** [Evolving-AI](https://github.com/Devanik21/Evolving-AI) · 237 commits · April 2026  
**Affiliation:** Electronics & Communication Engineering, NIT Agartala · Samsung ISWDP Fellow (IISc, 98.58th percentile)

This repository documents four successive experimental versions of a reinforcement learning agent — collectively the **RIENFoRZe series** — each building on the previous through principled architectural additions. Every mathematical operation, gradient computation, and optimizer update is implemented explicitly in NumPy. No autograd engine. No deep learning framework. The purpose is to make every algorithmic choice fully transparent and independently verifiable.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Version Progression Summary](#2-version-progression-summary)
3. [Module: brain.py — Reinforcement Learning Engine](#3-module-brainpy--reinforcement-learning-engine)
   - [3.1 Neural Architecture — Dueling DQN](#31-neural-architecture--dueling-dqn)
   - [3.2 Weight Initialization — He (Kaiming) Normal](#32-weight-initialization--he-kaiming-normal)
   - [3.3 Activation Function — Leaky ReLU](#33-activation-function--leaky-relu)
   - [3.4 Prioritized Experience Replay and Segment Tree](#34-prioritized-experience-replay-and-segment-tree)
   - [3.5 N-Step Return Estimation](#35-n-step-return-estimation)
   - [3.6 Intrinsic Curiosity Module — Count-Based Exploration](#36-intrinsic-curiosity-module--count-based-exploration)
   - [3.7 Double DQN Target Computation](#37-double-dqn-target-computation)
   - [3.8 Adam Optimizer — Manual Derivation](#38-adam-optimizer--manual-derivation)
   - [3.9 Soft Target Network Updates — Polyak Averaging](#39-soft-target-network-updates--polyak-averaging)
   - [3.10 Adaptive Curriculum Learning](#310-adaptive-curriculum-learning)
   - [3.11 Learning Rate Plateau Scheduler](#311-learning-rate-plateau-scheduler)
4. [Module: world.py — Environment Engine](#4-module-worldpy--environment-engine)
   - [4.1 State Representation Across Versions](#41-state-representation-across-versions)
   - [4.2 Fog-of-War Vision System](#42-fog-of-war-vision-system)
   - [4.3 Maze Generation Algorithms](#43-maze-generation-algorithms)
   - [4.4 Reward Engineering — Potential-Based Shaping](#44-reward-engineering--potential-based-shaping)
   - [4.5 A* Shortest Path Solver](#45-a-shortest-path-solver)
5. [Module: soul.py — Cognitive Architecture](#5-module-soulpy--cognitive-architecture)
   - [5.1 Valence-Arousal Emotion Model — Russell's Circumplex](#51-valence-arousal-emotion-model--russells-circumplex)
   - [5.2 Big Five Personality Traits — OCEAN](#52-big-five-personality-traits--ocean)
   - [5.3 Intent Classification Engine](#53-intent-classification-engine)
   - [5.4 Episodic Emotional Memory — Exponential Decay](#54-episodic-emotional-memory--exponential-decay)
   - [5.5 Relationship Dynamics](#55-relationship-dynamics)
   - [5.6 RL-to-Emotion Coupling](#56-rl-to-emotion-coupling)
6. [Module: memory_palace.py — Memory Architecture](#6-module-memory_palacepy--memory-architecture)
   - [6.1 Complementary Learning Systems Theory](#61-complementary-learning-systems-theory)
   - [6.2 Working Memory — Episodic Trace Buffer](#62-working-memory--episodic-trace-buffer)
   - [6.3 Episodic Memory — Indexed Event Store](#63-episodic-memory--indexed-event-store)
   - [6.4 Semantic Memory — Confidence-Weighted Fact Graph](#64-semantic-memory--confidence-weighted-fact-graph)
   - [6.5 Persistence — JSON Serialization Architecture](#65-persistence--json-serialization-architecture)
7. [Module: analytics.py — Telemetry Engine](#7-module-analyticspy--telemetry-engine)
   - [7.1 Rolling Statistics and Exponential Smoothing](#71-rolling-statistics-and-exponential-smoothing)
   - [7.2 Convergence and Plateau Detection](#72-convergence-and-plateau-detection)
   - [7.3 Capability Score — Composite Metric](#73-capability-score--composite-metric)
8. [RIENFoRZe-I — Foundational Architecture (17D)](#8-rienforze-i--foundational-architecture-17d)
9. [RIENFoRZe-II — Extended Sensory Architecture (52D)](#9-rienforze-ii--extended-sensory-architecture-52d)
   - [9.1 Pheromone Channel Engineering](#91-pheromone-channel-engineering)
   - [9.2 Dyna-Q Hallucination Planning](#92-dyna-q-hallucination-planning)
   - [9.3 Huber Loss — Motivation and Derivation](#93-huber-loss--motivation-and-derivation)
   - [9.4 Epsilon Schedule Correction](#94-epsilon-schedule-correction)
   - [9.5 Super Brain Mode](#95-super-brain-mode)
   - [9.6 Numerical Stability Shields](#96-numerical-stability-shields)
10. [RIENFoRZe-III — Full Sensory Architecture (64D)](#10-rienforze-iii--full-sensory-architecture-64d)
    - [10.1 Cardinal Wall Radar — Raycast Sensing](#101-cardinal-wall-radar--raycast-sensing)
    - [10.2 Scent Gradient Channels — Logarithmic Visit Differential](#102-scent-gradient-channels--logarithmic-visit-differential)
    - [10.3 Target Beacon — Unit Direction Encoding](#103-target-beacon--unit-direction-encoding)
    - [10.4 Local Flux — Pheromone Variance Signal](#104-local-flux--pheromone-variance-signal)
    - [10.5 Curiosity Self-Referential Loop](#105-curiosity-self-referential-loop)
    - [10.6 Accelerated Dyna-Q — Instant Breakthrough](#106-accelerated-dyna-q--instant-breakthrough)
    - [10.7 Backpropagation Through the Dueling Architecture](#107-backpropagation-through-the-dueling-architecture)
    - [10.8 Information-Theoretic Analysis of 64D vs 52D](#108-information-theoretic-analysis-of-64d-vs-52d)
11. [RIENFoRZe-IV — Tabular Dyna-Q Architecture (52D, Exact)](#11-rienforze-iv--tabular-dyna-q-architecture-52d-exact)
    - [11.1 The Approximation Error Argument](#111-the-approximation-error-argument)
    - [11.2 Tabular Q-Function and State Hashing](#112-tabular-q-function-and-state-hashing)
    - [11.3 Exact Q-Learning Update Rule](#113-exact-q-learning-update-rule)
    - [11.4 Convergence Guarantees — Tabular vs Approximate RL](#114-convergence-guarantees--tabular-vs-approximate-rl)
    - [11.5 Perfect World Model — Dyna-Q Without Approximation](#115-perfect-world-model--dyna-q-without-approximation)
    - [11.6 Memory Complexity Analysis](#116-memory-complexity-analysis)
    - [11.7 Why Learning Rate is 0.3, Not 0.001](#117-why-learning-rate-is-03-not-0001)
    - [11.8 JSON Serialization of the Q-Table](#118-json-serialization-of-the-q-table)
12. [MazE Companion Module — SARSA with Prioritized Sweeping](#12-maze-companion-module--sarsa-with-prioritized-sweeping)
13. [Cross-Version Architectural Comparison](#13-cross-version-architectural-comparison)
14. [Consolidated Hyperparameter Reference](#14-consolidated-hyperparameter-reference)
15. [File Architecture](#15-file-architecture)
16. [Installation and Usage](#16-installation-and-usage)

---

## 1. System Architecture Overview

Project A.L.I.V.E. NEXUS is organized as six cooperating modules. The RL engine (`brain.py`) and environment engine (`world.py`) form the core learning loop. The personality engine (`soul.py`) and memory architecture (`memory_palace.py`) form an orthogonal cognitive layer that receives learning signals from the RL engine and produces behavioral outputs observable through the Streamlit interface. The analytics engine (`analytics.py`) instruments both layers. The two entry points (`RIENFoRZe.py`, `RIENFoRZeADv.py`) orchestrate the full system.

```
┌──────────────────────────────────────────────────────────────────────┐
│                     RIENFoRZe.py / RIENFoRZeADv.py                  │
│                     (Orchestration + Streamlit UI)                   │
└────────┬────────────────────────┬───────────────────────────────────┘
         │                        │
┌────────▼────────┐      ┌────────▼────────────────────────────────┐
│   brain.py      │      │          soul.py + memory_palace.py     │
│  RL Engine      │◄────►│  Cognitive / Affective Architecture     │
│  Dueling DDQN   │ RL   │  Emotion · Personality · Memory         │
│  PER / N-Step   │stats │  Relationship · Consciousness Stream    │
│  ICM / Dyna-Q   │      └──────────────────────────────────────────┘
│  Curriculum     │
└────────┬────────┘
         │  action/state/reward
┌────────▼────────┐      ┌──────────────────────────────────────┐
│   world.py      │      │          analytics.py                │
│  Environment    │─────►│  Telemetry · Rolling Stats           │
│  Maze / FoW     │      │  Convergence · Capability Score      │
│  Traps / Portal │      └──────────────────────────────────────┘
└─────────────────┘
```

The RL loop operates at episode granularity: each episode generates a procedurally different maze, the agent navigates until success or timeout, and all modules update their internal state from the episode result. The cognitive layer receives a statistics dictionary `{td_error, reward, epsilon, level, ...}` and updates mood, memory, and relationship state accordingly.

---

## 2. Version Progression Summary

The RIENFoRZe series is a controlled experimental sequence. Each version either adds an independent architectural component or changes a single design decision. This structure enables clean ablation analysis across versions.

| Property | RIENFoRZe-I | RIENFoRZe-II | RIENFoRZe-III | RIENFoRZe-IV |
|---|---|---|---|---|
| State dimension | 17 | 52 | 64 | 52 |
| Q-function type | Neural (Dueling DQN) | Neural | Neural | Tabular (dict) |
| Loss function | MSE | Huber (δ=1.0) | Huber (δ=1.0) | None (exact) |
| Planning (Dyna-Q) | None | 5 steps | 25 steps | 20 steps |
| Breakthrough planning | None | 4× (20) | 5× (125) | 4× (80) |
| Convergence guarantee | No | No | No | Yes (finite MDP) |
| Vision field | 3×3 (9 cells) | 5×5 (25 cells) | 5×5 (25 cells) | 5×5 (25 cells) |
| Pheromone channel | No | 13-cell cross | 13-cell cross | 13-cell cross |
| Wall radar | No | No | 4D raycast | No |
| Scent gradients | No | No | 4D log-ratio | No |
| Target beacon | Distance only | Distance only | Unit vector (2D) | Distance only |
| Pheromone flux | No | No | Yes (std) | No |
| Curiosity in state | No | No | Yes (dim 63) | No |
| PER buffer | Yes (50K) | Yes (50K) | Yes (50K) | None |
| Target network | Yes (Polyak) | Yes (Polyak) | Yes (Polyak) | Not needed |
| Adam optimizer | Yes | Yes | Yes | Not applicable |
| Gradient clipping | ±10 | ±10 | ±10 | Not applicable |
| Weight clamping | No | ±100 | ±100 | Not applicable |
| Epsilon timing | Per train step | Per episode | Per episode | Per episode |
| Epsilon start | 1.0 | 0.7 | 0.7 | 0.7 |

---

## 3. Module: brain.py — Reinforcement Learning Engine

`brain.py` contains the complete RL pipeline: the neural network, the replay buffer, the curiosity module, the optimizer, and the curriculum manager. Approximately 746 lines. All matrix operations are explicit NumPy — there is no framework abstraction.

---

### 3.1 Neural Architecture — Dueling DQN

The network implements the **Dueling architecture** (Wang et al., 2016). Three shared hidden layers feed into two independent output streams: a **value stream** estimating V(s) and an **advantage stream** estimating A(s, a) for each of the four actions.

```
Input (D) → [Linear + LeakyReLU] → (256) → [Linear + LeakyReLU] → (128)
          → [Linear + LeakyReLU] → (64)
          → Value stream:    (64) → Linear → V(s) ∈ R
          → Advantage stream:(64) → Linear → A(s,·) ∈ R^4
          → Q(s, a) = V(s) + A(s, a) − mean_{a'} A(s, a')
```

where D is 17 (v-I), 52 (v-II/IV), or 64 (v-III).

**Dueling Aggregation.** The Q-value is assembled as:

```math
Q(s, a;\, \theta, \alpha, \beta) = V(s;\, \theta, \beta) + \left( A(s, a;\, \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a';\, \theta, \alpha) \right)
```

The mean subtraction over the advantage stream is critical. Without it, V(s) and A(s, a) are not uniquely identified: any constant shift between them leaves Q(s, a) unchanged. By forcing the mean advantage to zero, V(s) is uniquely constrained to represent the true state value, and A(s, a) represents deviation from that baseline. This resolves the **identifiability problem** and has been shown empirically to improve policy stability.

**Parameter count for RIENFoRZe-III (D = 64):**

```math
\text{Params} = (64 \times 256 + 256) + (256 \times 128 + 128) + (128 \times 64 + 64) + (64 \times 1 + 1) + (64 \times 4 + 4)
```

```math
= 16{,}640 + 32{,}896 + 8{,}256 + 65 + 260 = 58{,}117 \text{ parameters}
```

For RIENFoRZe-I/II (D = 17 / 52), the first layer is (17×256+256 = 4,608) or (52×256+256 = 13,568), giving 49,124 and 55,049 total parameters respectively.

---

### 3.2 Weight Initialization — He (Kaiming) Normal

All weight matrices are initialized using **He initialization** (He et al., 2015) to maintain variance of activations across layers when using ReLU-family nonlinearities:

```math
W^{(l)} \sim \mathcal{N}\!\left(0,\; \frac{2}{n_{l-1}}\right)
```

where `n_{l-1}` is the fan-in (input dimension) of layer `l`. Bias vectors are initialized to zero. The scaling factor of 2 (rather than 1 as in Xavier/Glorot initialization) compensates for the fact that ReLU activations zero out approximately half of their inputs, effectively halving the variance; doubling the initialization variance restores the intended signal propagation magnitude.

---

### 3.3 Activation Function — Leaky ReLU

Leaky ReLU is used throughout to prevent dead neurons — a pathology of standard ReLU where neurons receiving consistently negative pre-activations produce zero output and receive zero gradient indefinitely:

```math
f_{\text{leaky}}(x) = \begin{cases} x & x > 0 \\ 0.01 \cdot x & x \leq 0 \end{cases}
```

Its element-wise gradient, required for the backward pass:

```math
\frac{d f_{\text{leaky}}}{dx} = \begin{cases} 1.0 & x > 0 \\ 0.01 & x \leq 0 \end{cases}
```

The slope of 0.01 on the negative half ensures a non-zero gradient flow, allowing neurons to recover from negative pre-activation phases. This is applied element-wise during both the forward and backward passes at each hidden layer.

---

### 3.4 Prioritized Experience Replay and Segment Tree

The replay buffer implements **Prioritized Experience Replay** (PER; Schaul et al., 2015) with a **binary segment tree** data structure enabling O(log N) priority-weighted sampling. Capacity is fixed to the next power of 2:

```math
\text{cap} = 2^{\lceil \log_2 N \rceil}
```

**Priority Assignment.** At insertion, each transition receives a priority equal to the current maximum observed priority, guaranteeing at least one sampling:

```math
p_i = \max_{j \in \mathcal{B}} p_j
```

**Sampling Probability.** Transitions are sampled proportionally to a smoothed priority:

```math
P(i) = \frac{p_i^{\alpha}}{\sum_j p_j^{\alpha}}
```

The exponent `α` controls the degree of prioritization: `α = 0` recovers uniform sampling; `α = 1` gives full greedy prioritization. This system uses `α = 0.6`, providing a balance between exploiting high-TD-error transitions and maintaining diversity in the training distribution.

**Importance-Sampling Correction.** Non-uniform sampling introduces bias into the gradient estimate. Each transition's update is reweighted by an importance-sampling (IS) correction:

```math
w_i = \left( \frac{1}{N \cdot P(i)} \right)^{\beta}
```

These weights are normalized by the maximum in the batch to ensure stability:

```math
\hat{w}_i = \frac{w_i}{\max_j w_j}
```

The exponent `β` is annealed from `β_start = 0.4` toward 1.0 over `beta_frames = 100,000` environment steps:

```math
\beta(t) = \min\!\left(1.0,\; \beta_{\text{start}} + t \cdot \frac{1 - \beta_{\text{start}}}{\text{beta\_frames}}\right)
```

As `β → 1`, the IS correction becomes fully unbiased. The annealing schedule reflects a principled trade-off: in early training, high-variance IS weights can destabilize learning, so bias correction is applied gradually.

**Segment Tree Operations.** The implementation maintains two concurrent trees:

- `SumSegmentTree`: supports O(log N) prefix-sum queries for stratified sampling
- `MinSegmentTree`: supports O(log N) minimum priority lookup for IS weight normalization

The `find_prefixsum_idx` operation partitions the total priority sum `S` into `B` equal segments and retrieves the transition index corresponding to a given prefix-sum value, enabling stratified sampling that reduces sample correlation within a batch.

**Priority Update After Batch.** After computing Bellman targets and obtaining TD errors, each sampled transition's priority is updated:

```math
p_i \leftarrow \max\!\left(\delta_i,\; \epsilon_{\text{floor}}\right), \quad \epsilon_{\text{floor}} = 10^{-6}
```

The floor prevents zero-probability sampling for any transition.

---

### 3.5 N-Step Return Estimation

Rather than the single-step Bellman target, RIENFoRZe uses an **N-step return** (n = 3) to provide richer temporal credit assignment:

```math
G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max_{a'} Q_{\text{target}}(s_{t+n}, a')
```

The N-step buffer maintains a deque of length n. When full, the accumulated return for the oldest transition is computed and flushed to the PER buffer. If a terminal state appears within the window at index k*, the accumulation terminates early:

```math
G_t^{(n)} = \sum_{k=0}^{k^{*}} \gamma^k r_{t+k}
```

**Bias-Variance Trade-off.** N-step returns reduce the variance of the TD estimate relative to single-step TD(0), at the cost of slightly increased bias when the bootstrapped Q-value at `s_{t+n}` is inaccurate. With Dyna-Q planning (in v-II/III/IV) accelerating the accuracy of Q-values, this bootstrapping bias decreases faster than it would for a pure online agent.

**Three-Step Expansion (n = 3):**

```math
G_t^{(3)} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 \max_{a'} Q_{\text{target}}(s_{t+3}, a')
```

With `γ = 0.99`, the discount factors are 1.0, 0.99, and 0.9801, giving substantial weight to all three reward components before the bootstrapped value.

---

### 3.6 Intrinsic Curiosity Module — Count-Based Exploration

RIENFoRZe uses a **count-based exploration bonus** (Bellemare et al., 2016 inspired) rather than a learned forward model. The continuous state vector is discretized into a grid of `bins = 16` per dimension:

```math
k(s) = \left\lfloor \text{clip}(s, 0, 1) \cdot (B - 1) \right\rfloor \quad \in \mathbb{Z}^{D}
```

where `B = 16` is the number of bins and `D` is the state dimension. Visit counts `N(k(s))` are maintained per discretized key. The intrinsic reward is:

```math
r_i(s) = \frac{\beta}{\sqrt{N(k(s))}}, \quad \beta = 0.05
```

This decay rate of O(1/√N) is derived from pseudocount theory (Bellemare et al., 2016): in a model where the agent maintains a density model over states, the intrinsic bonus is proportional to the prediction gain — how much the model is updated by observing s. For a Laplace estimator over discretized states, this reduces to the 1/√N form.

**Augmented Reward.** The total reward signal fed to the agent combines extrinsic and intrinsic components:

```math
r_{\text{aug}} = r_{\text{ext}} + r_i(s)
```

**Coverage Metric.** The exploration coverage is tracked as the number of unique discretized states visited:

```math
\mathcal{C} = \bigl|\{k(s) : s \text{ ever visited}\}\bigr|
```

A spatial heatmap projecting coverage over the (row, col) dimensions of the state vector provides a visual diagnostic of exploration quality.

---

### 3.7 Double DQN Target Computation

Standard DQN suffers from **maximization bias**: the same network both selects the greedy action and evaluates it, causing systematic overestimation of Q-values (van Hasselt et al., 2010). **Double DQN** (van Hasselt et al., 2015) decouples action selection from action evaluation across the online and target networks.

**Action Selection (online network):**

```math
a^{*} = \arg\max_{a'} Q_{\text{online}}(s', a';\, \theta)
```

**Action Evaluation (target network):**

```math
y_i = r_i + \gamma (1 - d_i) \cdot Q_{\text{target}}(s'_i, a^{*};\, \theta^{-})
```

where `θ` are the online network parameters, `θ⁻` are the (lagged) target network parameters, and `d_i ∈ {0, 1}` is the terminal indicator.

**TD Error for PER Update:**

```math
\delta_i = \left| y_i - Q_{\text{online}}(s_i, a_i;\, \theta) \right|
```

**Weighted Loss Function:**

```math
\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} \hat{w}_i \cdot \ell\!\left(y_i,\; Q_{\text{online}}(s_i, a_i;\, \theta)\right)
```

where `ℓ` is either MSE (v-I) or Huber (v-II/III). Only the Q-value at the taken action `a_i` contributes to the loss; gradients with respect to all other action outputs are zero.

**MSE Gradient (RIENFoRZe-I):**

```math
\frac{\partial \mathcal{L}}{\partial Q(s_i, a_i)} = -2\hat{w}_i (y_i - Q(s_i, a_i))
```

---

### 3.8 Adam Optimizer — Manual Derivation

The Adam optimizer (Kingma and Ba, 2014) is implemented from first principles. For each parameter tensor `p` and its gradient `g` at update step `t`:

**First Moment (Exponential Moving Average of Gradients):**

```math
m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t
```

**Second Moment (Exponential Moving Average of Squared Gradients):**

```math
v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
```

**Bias Correction.** Because `m` and `v` are initialized to zero, early estimates are biased toward zero (particularly when `β₁` and `β₂` are close to 1). The bias-corrected moments are:

```math
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
```

```math
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
```

**Parameter Update:**

```math
\theta_t \leftarrow \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t
```

Default hyperparameters: `η = 0.001`, `β₁ = 0.9`, `β₂ = 0.999`, `ε = 10⁻⁸`.

**Gradient Clipping.** Before the Adam update, all gradient tensors are element-wise clipped to prevent gradient explosion in early, high-variance training:

```math
g_t \leftarrow \text{clip}(g_t,\; -c,\; c), \quad c = 10.0
```

This is distinct from norm-based clipping: element-wise clipping applies independently to each scalar, whereas norm-based clipping scales the entire gradient tensor by a scalar when its norm exceeds a threshold.

---

### 3.9 Soft Target Network Updates — Polyak Averaging

Rather than periodic hard copies from online to target network, RIENFoRZe uses **Polyak averaging** (soft update) at every training step:

```math
\theta^{-} \leftarrow \tau \cdot \theta + (1 - \tau) \cdot \theta^{-}, \quad \tau = 0.005
```

This introduces a time-scale separation: the online network adapts quickly via gradient descent while the target network tracks it with exponential lag. For a scalar parameter with constant online value `θ*`, the target converges as:

```math
\theta^{-}_t = \theta^{*} + (\theta^{-}_0 - \theta^{*})(1 - \tau)^t
```

**Half-life of target lag:**

```math
t_{1/2} = \frac{\ln 2}{\tau} = \frac{\ln 2}{0.005} \approx 139 \text{ steps}
```

The target network therefore lags approximately 139 gradient steps behind the online network. This is the intended behavior: a slowly-moving target reduces oscillations in the training signal that arise when both the Q-network and its own target move simultaneously.

---

### 3.10 Adaptive Curriculum Learning

The `CurriculumManager` implements **Automatic Curriculum Learning (ACL)** across 10 discrete difficulty levels. The manager tracks a rolling window of 20 recent episode scores and adjusts the difficulty level to maintain the agent near its Zone of Proximal Development (ZPD).

**Episode Score.** Each episode produces a composite performance score:

```math
\text{eff} = \max\!\left(0,\; 1 - \frac{t_{\text{steps}}}{t_{\text{max}}}\right) \cdot \mathbb{1}[\text{success}]
```

```math
\text{score} = 0.5 \cdot \mathbb{1}[\text{success}] + 0.5 \cdot \text{eff}
```

The score is 0 for failed episodes (regardless of efficiency), 0.5 for inefficient successes, and up to 1.0 for maximally efficient successes.

**Rolling Window Average:**

```math
\bar{s} = \frac{1}{|\mathcal{W}|} \sum_{i \in \mathcal{W}} \text{score}_i
```

**Level Transition Logic:**

```math
\text{Level} \leftarrow \begin{cases} \text{Level} + 1 & \bar{s} \geq 0.72 \text{ and Level} < 10 \\ \text{Level} - 1 & \bar{s} \leq 0.25 \text{ and Level} > 1 \\ \text{Level} & \text{otherwise} \end{cases}
```

After any level transition, the rolling window is cleared so performance is re-evaluated at the new difficulty from a clean slate.

**ZPD Progress Metric:**

```math
\rho_{\text{ZPD}} = \min\!\left(1.0,\; \frac{\bar{s}}{0.72}\right)
```

This quantity tracks progress toward the promotion threshold at the current level and is streamed as a real-time indicator in the analytics panel.

**Level Configuration Table.**

| Level | Maze H | Maze W | Algorithm | Fog | Dynamic Traps | Portals |
|-------|--------|--------|-----------|-----|----------------|---------|
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

---

### 3.11 Learning Rate Plateau Scheduler

A `LRScheduler` monitors rolling average reward and reduces the learning rate when no improvement is detected for `patience = 100` consecutive episodes:

```math
\eta_{t+1} = \begin{cases} \eta_t & \text{if } \bar{r}_t > \bar{r}^{*} + 10^{-4} \\ \max(\eta_{\min},\; \eta_t \cdot 0.5) & \text{if wait} \geq \text{patience} \end{cases}
```

where `wait` counts consecutive non-improving episodes. The minimum learning rate is `η_min = 10⁻⁵`. When a plateau is detected, the learning rate is halved and the wait counter resets.

This operates on a timescale much slower than individual gradient steps, providing a coarse-grained adaptation that complements the per-step Adam moment estimates.

---

## 4. Module: world.py — Environment Engine

`world.py` (~744 lines) implements the maze environment: procedural generation via three distinct algorithms, fog-of-war, dynamic traps, teleport portals, potential-based reward shaping, and an A* optimal path solver.

---

### 4.1 State Representation Across Versions

**RIENFoRZe-I (17D):**

```
s ∈ R^17 = [vision(9) | pos(2) | tpos(2) | dist(1) | trap(1) | fog(1) | time(1)]
```

**RIENFoRZe-II (52D):**

```
s ∈ R^52 = [vision(25) | pheromones(13) | pos(2) | tpos(2) | dir(2) | dist(1) | trap(1) | fog(1) | time(1) | momentum(4)]
```

**RIENFoRZe-III (64D):**

```
s ∈ R^64 = [vision(25) | pheromones(13) | telemetry(10) | momentum(4) | radar(4) | scent(4) | beacon(2) | flux(1) | curiosity(1)]
```

**RIENFoRZe-IV (52D, tabular):** same layout as v-II; different usage (discretized hash key rather than neural network input).

**Vision Encoding (all versions):**

```math
v_{dr,dc} = \begin{cases} 0.0 & \text{cell is a passable path and visible} \\ 1.0 & \text{cell is a wall or out-of-bounds} \\ 0.5 & \text{cell is within fog-of-war (unexplored)} \end{cases}
```

where (dr, dc) ranges over the local neighborhood: {-1,0,1}² for the 3×3 view (v-I) or {-2,-1,0,1,2}² for the 5×5 view (v-II/III/IV).

**Normalized Manhattan Distance:**

```math
d_{\text{manhattan}} = \frac{|r_{\text{agent}} - r_{\text{target}}| + |c_{\text{agent}} - c_{\text{target}}|}{H + W}
```

**Time Pressure:**

```math
\tau = \frac{t_{\text{current}}}{t_{\text{max}}}
```

Both features are clipped to [0, 1] before inclusion in the state vector.

---

### 4.2 Fog-of-War Vision System

The `FogOfWar` class maintains a binary visibility grid. Cell (r', c') becomes visible when it falls within a Chebyshev ball of radius `R` centered on the agent's current position (r, c):

```math
\text{visible}[r', c'] = \text{True} \iff \max(|r' - r|,\; |c' - c|) \leq R
```

The vision radius adapts to maze height:

```math
R = \max(3,\; \min(6,\; H \div 4))
```

Visibility is monotonically cumulative within an episode: once revealed, a cell remains visible. The fog coverage scalar in the state vector is:

```math
\phi_{\text{fog}} = \frac{|\{(r,c) : \text{visible}[r,c] = \text{True}\}|}{H \cdot W} \in [0, 1]
```

This value increases monotonically within an episode, providing the agent with a measure of its own exploration progress.

---

### 4.3 Maze Generation Algorithms

Three distinct maze topologies are generated depending on the curriculum level.

**Recursive Backtracker (DFS).** Produces long winding corridors with few dead-ends. A straight-bias parameter (65%) preferentially continues the current direction, reducing winding while preserving perfect maze topology. Starting from cell (0, 0), the algorithm carves passages to unvisited neighbors via DFS, backtracking when all neighbors are visited. System recursion depth is set to 10,000 to support Level-10 mazes (35×41).

**Prim's Algorithm.** Produces highly branching structures with many dead-ends, directly challenging the agent's junction decision-making. The algorithm maintains a frontier set of candidate walls; it grows the maze by randomly selecting and removing a valid frontier wall, ensuring connectivity to the existing maze portion.

**Wilson's Algorithm.** Produces a **Uniform Spanning Tree (UST)** — a spanning tree of the grid graph drawn uniformly at random from all possible spanning trees. The algorithm uses loop-erased random walks: from an unvisited cell, take a random walk until it hits the existing maze, erasing any loops formed during the walk. This guarantees that every spanning tree is equally likely. The UST is the most theoretically difficult topology for the agent: the absence of spatial bias means no local heuristics reliably predict the global structure.

**Hybrid (Level 10).** Combines all three generators, producing mazes with locally distinct structural regions corresponding to each generator's topology. This is the most heterogeneous configuration, requiring the agent to adapt its navigation strategy within a single episode.

---

### 4.4 Reward Engineering — Potential-Based Shaping

The reward function is designed to be **potential-based**, ensuring optimal policy invariance under shaping (Ng et al., 1999).

**Total Reward Components:**

```math
r = r_{\text{goal}} + r_{\text{progress}} + r_{\text{step}} + r_{\text{wall}} + r_{\text{trap}} + r_{\text{portal}} + r_{\text{timeout}}
```

| Component | Value | Condition |
|-----------|-------|-----------|
| Goal reward | +25.0 | Reached target |
| Distance progress | +3.0 · Δd | Moved closer to target |
| Distance regression | −1.5 · Δd | Moved farther from target |
| Step penalty | −0.05 | Each step |
| Wall penalty | −0.3 | Hit wall |
| Trap penalty | −10.0 | Caught by dynamic trap |
| Portal bonus | +2.0 | Used teleport portal |
| Timeout penalty | −1.0 | Episode timeout |

Final reward is clipped: `r ∈ [−10.0, 15.0]`.

The asymmetric progress coefficients (+3.0 vs −1.5) create a gradient that strongly encourages goal-directed movement without making regression catastrophically penalized, allowing exploratory backtracking when necessary.

**Potential-Based Shaping Theorem (Ng et al., 1999).** With shaping potential `Φ(s) = −d(s, goal)`, the shaped reward:

```math
r'(s, a, s') = r(s, a, s') + \gamma \cdot \Phi(s') - \Phi(s)
```

does not alter the set of optimal policies. The proof relies on the fact that the shaping term telescopes over any complete trajectory, leaving the total return unchanged up to an episode-level constant.

---

### 4.5 A* Shortest Path Solver

The environment includes a complete A* solver using a min-heap priority queue. The cost function is:

```math
f(n) = g(n) + h(n), \quad h(n) = |r_n - r_{\text{goal}}| + |c_n - c_{\text{goal}}|
```

where `g(n)` is the actual path cost from the start and `h(n)` is the Manhattan distance heuristic to the goal. The Manhattan distance is an admissible heuristic for grid navigation (it never overestimates the true cost), guaranteeing optimality of the A* solution.

The optimal path length `L*` is used to compute agent step-efficiency:

```math
\text{eff} = \frac{L^{*}}{t_{\text{steps}}} \in (0, 1]
```

A value of 1.0 indicates the agent found the shortest possible path; values below 1.0 indicate suboptimality proportional to the ratio of wasted steps.

---

## 5. Module: soul.py — Cognitive Architecture

`soul.py` (~717 lines) implements the affective and cognitive layer of A.L.I.V.E. NEXUS. It receives learning statistics from the RL engine and produces behavior observable through the chat interface and Streamlit panels: mood states, personality-modulated responses, episodic memory recalls, and a consciousness stream of inner monologue.

---

### 5.1 Valence-Arousal Emotion Model — Russell's Circumplex

The emotion model is grounded in **Russell's Circumplex Model of Affect** (Russell, 1980), which represents all discrete emotions as points in a two-dimensional space defined by **valence** (pleasantness) and **arousal** (activation level):

```math
e \in \mathbb{R}^2, \quad e = (v, a), \quad v \in [-1, 1],\; a \in [-1, 1]
```

The `EmotionPoint` class represents a point in this space. Blending between emotional states is modeled as an exponential moving average in the valence-arousal plane with blend coefficient `α = 0.3`:

```math
e_{t+1} = e_t + \alpha \cdot (e_{\text{target}} - e_t)
```

This formulation is equivalent to a first-order low-pass filter applied independently to valence and arousal. It prevents abrupt mood transitions while allowing responsive adaptation to new inputs.

**Emotion Intensity.** The magnitude of the emotion vector defines its intensity:

```math
\|e\| = \sqrt{v^2 + a^2} \in [0, \sqrt{2}]
```

Neutral (v = 0, a = 0) has zero intensity. Peak states — such as Excited (v = 0.8, a = 0.9) — have high intensity.

**Quadrant-Based Emotion Labeling.** Discrete emotion labels are assigned by quadrant and magnitude:

| Quadrant (v, a) | High Arousal | Medium Arousal | Low Arousal |
|---|---|---|---|
| v > 0 (positive) | Excited | Happy | Serene |
| v ≈ 0 (neutral) | Tense | Calm/Neutral | Bored |
| v < 0 (negative) | Alarmed | Sad | Depressed |

**Preset Emotion Coordinates.** Named presets for common RL-induced states:

| State | Valence | Arousal |
|---|---|---|
| Excited (goal reached) | 0.8 | 0.9 |
| Happy (moderate reward) | 0.6 | 0.3 |
| Calm (stable navigation) | 0.2 | −0.1 |
| Tense (trap proximity) | −0.1 | 0.7 |
| Sad (failed episode) | −0.6 | −0.2 |
| Confused (high TD error) | 0.0 | 0.5 |
| Depressed (extended failure) | −0.7 | −0.6 |

---

### 5.2 Big Five Personality Traits — OCEAN

The `PersonalityTraits` class implements a **Big Five** (OCEAN) trait model. Each trait is initialized from a seeded Gaussian distribution:

```math
T_k \sim \mathcal{N}(0.5,\; 0.15), \quad k \in \{\text{O, C, E, A, N}\}, \quad T_k \in [0.1, 0.9]
```

The five traits:

- **Openness (O):** Curiosity and openness to new experience; higher O → more exploratory language and behavior
- **Conscientiousness (C):** Organization and goal-directedness; higher C → more methodical navigation, stronger exploitation bias
- **Extraversion (E):** Sociability; higher E → more verbose, energetic responses in the chat interface
- **Agreeableness (A):** Cooperative tendency; higher A → more accommodating responses to positive user input
- **Neuroticism (N):** Emotional instability; higher N → stronger mood swings in response to TD errors and rewards

**Trait Adaptation.** Traits adapt slowly over training based on observed success rate and reward trend:

```math
O \leftarrow O + 0.01 \cdot \text{sgn}(0.5 - \text{success\_rate})
```

```math
N \leftarrow N - 0.01 \cdot \text{sgn}(\text{reward\_trend})
```

This creates a feedback loop: consistent success reduces Neuroticism over episodes; consistent failure increases it. Openness adjusts toward exploration when success rates are low, nudging the agent toward novel strategies.

---

### 5.3 Intent Classification Engine

The `IntentEngine` implements a multi-class intent classifier based on weighted keyword patterns. For an input text, it computes a score for each intent class:

```math
\text{score}(c) = \sum_{k \in K_c} w_k \cdot \mathbb{1}[\text{keyword } k \in \text{text}]
```

where `K_c` is the keyword set for class `c` and `w_k` is a weight. The predicted intent is the argmax class:

```math
\hat{c} = \arg\max_{c} \text{score}(c)
```

Confidence is computed as a softmax over the raw scores:

```math
\text{conf}(\hat{c}) = \frac{\exp(\text{score}(\hat{c}))}{\sum_{c'} \exp(\text{score}(c'))}
```

Intent classes include: `praise`, `criticism`, `question`, `encouragement`, `frustration`, `curiosity`, `neutral`. Each class has a distinct keyword dictionary with associated importance weights.

---

### 5.4 Episodic Emotional Memory — Exponential Decay

The `EmotionalMemory` class stores `MemoryTrace` objects, each encoding a content string, an `EmotionPoint`, a context label, and a timestamp. Memory strength decays exponentially with elapsed time:

```math
\text{strength}(t) = \text{strength}_0 \cdot 2^{-\text{elapsed} / t_{1/2}}
```

with half-life `t_{1/2} = 3600` seconds. This models the **forgetting curve** (Ebbinghaus, 1885): memories fade over time unless reinforced. Reinforcement increases strength:

```math
\text{strength} \leftarrow \min(1.0,\; \text{strength} + 0.1)
```

**Relevance-Weighted Recall.** When recalling memories relevant to a query string, memories are scored by a combination of keyword overlap and current strength:

```math
\text{relevance}(m, q) = \text{keyword\_overlap}(m.\text{content}, q) \cdot m.\text{strength}
```

The top-n memories by relevance are returned for context injection into the response generator.

---

### 5.5 Relationship Dynamics

The `RelationshipEngine` tracks a real-valued relationship score that evolves based on the intent and sentiment of each user-agent interaction:

```math
R_{t+1} = R_t + \Delta R(\text{intent}, \text{sentiment})
```

| Intent | ΔR |
|---|---|
| praise | +5 |
| encouragement | +3 |
| question | +1 |
| neutral | 0 |
| criticism | −5 |
| frustration | −10 |

The relationship score is mapped to a discrete stage with associated behavioral modifiers:

```math
\text{stage} = \begin{cases} \text{Stranger} & R < 10 \\ \text{Acquaintance} & 10 \leq R < 30 \\ \text{Friend} & 30 \leq R < 60 \\ \text{Companion} & R \geq 60 \end{cases}
```

Higher relationship stages activate richer response templates, longer introspective passages in the consciousness stream, and proactively warmer language.

---

### 5.6 RL-to-Emotion Coupling

The `SoulCore.update_from_rl` method receives a statistics dictionary from the RL engine and maps learning signals to emotional state transitions:

```math
\Delta e = f_{\text{rl2emo}}\!\left(\delta_{\text{TD}},\; r_{\text{episode}},\; \epsilon,\; \text{level},\; \text{success}\right)
```

The specific mapping (implemented as a rule-based system):

| Condition | Emotion Target | Blend α |
|---|---|---|
| `δ_TD > 15` (high surprise) | Confused | 0.4 |
| `δ_TD > 5` (active learning) | Curious / Tense | 0.3 |
| `r_episode > 10` (success) | Excited | 0.5 |
| `r_episode < −5` (failure) | Sad | 0.3 |
| Level promotion | Excited → Calm | 0.6 |
| Level demotion | Sad | 0.5 |
| `ε < 0.1` (exploitation phase) | Calm / Serene | 0.2 |

This mapping is not learned — it is a design choice that creates legible, interpretable emotional behavior from purely quantitative RL signals. Whether this constitutes genuine affect or a behavioral simulation of affect is left as an open question.

---

## 6. Module: memory_palace.py — Memory Architecture

`memory_palace.py` (~529 lines) implements a multi-tier memory system inspired by **Complementary Learning Systems (CLS) theory** (McClelland et al., 1995), which proposes that biological brains use fast hippocampal binding for episodic memory and slow cortical consolidation for semantic memory. The module provides working, episodic, and semantic memory layers with JSON-based cross-session persistence.

---

### 6.1 Complementary Learning Systems Theory

CLS theory predicts that efficient learning systems benefit from two complementary memory stores:

- A **fast-learning, pattern-separated** store (hippocampus / working/episodic memory in this module) that records individual experiences with high fidelity
- A **slow-learning, pattern-completing** store (neocortex / semantic memory) that extracts statistical regularities across many experiences

This module implements both: `WorkingMemory` and `EpisodicMemory` serve as the fast store; `SemanticMemory` (the world model) serves as the slow store.

---

### 6.2 Working Memory — Episodic Trace Buffer

`WorkingMemory` provides temporary storage for the current episode context. It maintains a fixed-capacity trace deque:

```math
|\text{trace}| \leq C_{\text{WM}} = 64 \text{ slots}
```

Each trace entry records `(timestamp, key, value_preview)`. The working memory is cleared at episode termination. Its primary function is to provide the response generator with recent interaction context for coherent short-term dialogue.

---

### 6.3 Episodic Memory — Indexed Event Store

`EpisodicMemory` stores full `Episode` records, each containing approximately 18 fields of metadata:

```
Episode = {episode_id, timestamp, maze_seed, maze_alg, maze_h, maze_w,
           curriculum_level, total_steps, max_steps, total_reward, success,
           efficiency, cells_visited, fog_used, traps_used, avg_td_error,
           epsilon_start, epsilon_end, tags}
```

Episodes are indexed by episode ID and can be queried by tag, success/failure, curriculum level, or time range. The episodic store grows unboundedly within a session (subject to JSON persistence), providing a complete longitudinal record of training history.

**Efficiency Calculation:**

```math
\text{eff}_{\text{episode}} = \frac{L^{*}_{\text{A*}}}{t_{\text{steps}}} \cdot \mathbb{1}[\text{success}]
```

where `L*` is the A* optimal path length for that episode's maze.

---

### 6.4 Semantic Memory — Confidence-Weighted Fact Graph

`SemanticMemory` stores `Fact` objects: key-value pairs with associated confidence scores, provenance labels, and access statistics:

```
Fact = {key, value, confidence ∈ [0,1], source ∈ {observation, inference, user}, updated_at, access_count}
```

Confidence strengthens with repeated confirmation:

```math
\text{conf} \leftarrow \min(1.0,\; \text{conf} + 0.05)
```

and weakens without reinforcement:

```math
\text{conf} \leftarrow \max(0.0,\; \text{conf} - 0.02)
```

This models the observation that factual knowledge accumulated through repeated experience is more reliable than knowledge from a single observation. The confidence scores can be used downstream to weight the influence of different world model facts on action selection.

---

### 6.5 Persistence — JSON Serialization Architecture

The full cognitive state is serialized to a ZIP archive containing three JSON files:

```
state_snapshot.zip/
├── weights.json    — Neural network weights (or Q-table for v-IV)
├── config.json     — Hyperparameter snapshot
└── stats.json      — Training statistics and memory contents
```

This format provides API-compatible serialization across all four RIENFoRZe versions. The neural versions store `{online_net, target_net, optimizer_state}` in `weights.json`; RIENFoRZe-IV stores the Q-table as a string-keyed dictionary (see Section 11.8).

---

## 7. Module: analytics.py — Telemetry Engine

`analytics.py` (~414 lines) provides instrumentation for both the RL and cognitive layers. It computes rolling statistics, detects convergence plateaus, and produces the composite capability score displayed in the Streamlit dashboard.

---

### 7.1 Rolling Statistics and Exponential Smoothing

**Rolling Mean and Standard Deviation:**

```math
\bar{x}_w = \frac{1}{w} \sum_{i=n-w}^{n} x_i, \quad \sigma_w = \sqrt{\frac{1}{w} \sum_{i=n-w}^{n} (x_i - \bar{x}_w)^2}
```

**Exponential Moving Average (EMA):**

```math
\text{EMA}_t = \alpha \cdot x_t + (1 - \alpha) \cdot \text{EMA}_{t-1}, \quad \text{EMA}_0 = x_0
```

with smoothing coefficient `α = 0.1`. The EMA is used to produce smooth curves in the reward and loss panels while preserving the ability to detect sudden changes.

**Linear Trend Estimation.** The slope of a linear fit over recent N values is computed via `numpy.polyfit` of degree 1:

```math
\hat{\beta}_1 = \frac{\sum_{i}(x_i - \bar{x})(i - \bar{i})}{\sum_{i}(i - \bar{i})^2}
```

A positive slope indicates improving performance; a negative slope indicates degradation. The trend estimate is used by the plateau detector.

---

### 7.2 Convergence and Plateau Detection

A **convergence detector** monitors whether the rolling average reward has stopped improving. It maintains a `best_average` tracker and a `wait` counter:

```math
\text{converged} = \begin{cases} \text{True} & \text{if } \bar{r}_t \geq \text{best\_average} - \delta_{\text{tol}} \text{ for patience steps} \\ \text{False} & \text{otherwise} \end{cases}
```

with `δ_tol = 0.5` and patience configurable per experiment. When convergence is detected, the analytics engine logs the episode and emits a signal to the Streamlit UI.

The same mechanism drives the LR scheduler (Section 3.11), ensuring that learning rate reduction and convergence detection use consistent plateau criteria.

---

### 7.3 Capability Score — Composite Metric

A composite capability score aggregates multiple performance axes into a single scalar for cross-episode and cross-version comparison:

```math
\text{capability} = w_1 \cdot \text{success\_rate} + w_2 \cdot \bar{\text{eff}} + w_3 \cdot \bar{\text{coverage}} + w_4 \cdot (1 - \bar{\epsilon}) + w_5 \cdot \frac{\text{level}}{10}
```

| Component | Weight | Description |
|---|---|---|
| Success rate | 0.35 | Rolling success fraction |
| Path efficiency | 0.25 | Mean A*-normalized efficiency |
| Exploration coverage | 0.15 | Mean fraction of maze cells visited |
| Exploitation level | 0.15 | `1 − ε` captures how much the agent exploits learned policy |
| Curriculum level | 0.10 | Normalized difficulty tier |

The weights sum to 1.0 and were set to reflect the relative importance of these metrics for a navigation agent. Capability scores are streamed in real time and stored in the episodic memory for longitudinal analysis.

---

## 8. RIENFoRZe-I — Foundational Architecture (17D)

RIENFoRZe-I establishes the complete baseline pipeline. Every subsequent version is an extension or replacement of one or more components defined here.

**State vector:** 17-dimensional, composed as:

```
s ∈ R^17 = [vision(9) | pos(2) | tpos(2) | dist(1) | trap(1) | fog(1) | time(1)]
```

| Index | Dims | Feature | Range |
|-------|------|---------|-------|
| 0–8 | 9 | 3×3 local vision (wall/path/fog) | {0.0, 0.5, 1.0} |
| 9–10 | 2 | Normalized agent position (r/H, c/W) | [0, 1] |
| 11–12 | 2 | Normalized target position (r/H, c/W) | [0, 1] |
| 13 | 1 | Normalized Manhattan distance | [0, 1] |
| 14 | 1 | Normalized distance to nearest trap | [0, 1] |
| 15 | 1 | Fog coverage ratio | [0, 1] |
| 16 | 1 | Time pressure `t/t_max` | [0, 1] |

**Neural network:** Dueling DQN, input 17D, layers (256, 128, 64), output 4 actions. Total 49,124 parameters.

**PER buffer capacity:** 50,000 transitions. Segment tree capacity is rounded up to 65,536 (next power of 2 ≥ 50,000).

**Known limitation (corrected in v-II):** Epsilon decay was applied once per training step rather than once per episode, causing epsilon to decay approximately `min(T, batch_fill_steps)` times faster than the intended schedule. For an episode of T = 200 steps with batch size 64 and buffer filling at step 1000, the per-step decay accumulates ~200 multiplications of 0.997 per episode, yielding an effective per-episode decay rate of 0.997^200 ≈ 0.549 — far more aggressive than intended.

**Key equations reviewed in full above:** Dueling aggregation (Section 3.1), PER sampling (Section 3.4), N-step return (Section 3.5), ICM (Section 3.6), Double DQN target (Section 3.7), Adam (Section 3.8), Polyak averaging (Section 3.9), ACL (Section 3.10).

---

## 9. RIENFoRZe-II — Extended Sensory Architecture (52D)

RIENFoRZe-II introduces five independent modifications to the v-I baseline, each documented separately to support ablation analysis.

**Summary of changes:**

| Modification | v-I | v-II |
|---|---|---|
| State dimension | 17D | 52D |
| Vision field | 3×3 | 5×5 |
| Pheromone channel | None | 13-cell cross |
| Loss function | MSE | Huber (δ=1) |
| Dyna-Q | None | 5 steps/transition |
| Epsilon initialization | 1.0 | 0.7 |
| Epsilon timing | Per train step | Per episode |
| Numerical shields | None | Weight ±100, Q ±10⁶ |
| Breakthrough multiplier | None | 4× on success |

---

### 9.1 Pheromone Channel Engineering

The pheromone system encodes the agent's own historical visit density as a spatial signal, enabling stigmergic self-navigation — reasoning about which regions have been explored without an explicit map.

**Pheromone Grid Accumulation:**

```math
P[r, c] \leftarrow P[r, c] + 1
```

**Normalization to [0, 1]:**

```math
P_{\text{norm}}[r, c] = \frac{P[r, c]}{\max_{r', c'} P[r', c'] + \epsilon}
```

**13-Cell Cross Pattern.** Rather than a full 2D subgrid, the pheromone observation samples 13 specific cells: the center, 4 cardinal neighbors, 4 diagonal neighbors, and 4 extended cardinal positions (range 2). This captures local trail density with 13 values rather than 25, preserving sensitivity to exploration gradients while reducing dimensionality.

**Pheromone Gradient Interpretation.** The agent can implicitly compute a gradient from the 13 cross values. If cardinal pheromones decrease in direction `d`, that direction is less explored. Over training, the agent may learn to bias exploration toward the low-pheromone direction — a form of emergent anti-revisiting behavior derived entirely from the reward signal.

**52D Dimension Accounting:**

```
25 (vision) + 13 (pheromone) + 2 (pos) + 2 (tpos) + 2 (dir) + 1 (dist) + 1 (trap)
+ 1 (fog) + 1 (time) + 4 (momentum) = 52
```

The direction vector `(δr, δc) ∈ {(−1,0),(1,0),(0,−1),(0,1)}` encodes the continuous geometric direction of the last step. The momentum one-hot encodes the categorical action identity. Both are retained: different layers may find different representations more useful, and they interact differently with the dueling head.

---

### 9.2 Dyna-Q Hallucination Planning

RIENFoRZe-II introduces **Dyna-Q** (Sutton, 1991) as a sample amplification mechanism. After each real environmental step, the agent performs K = 5 additional simulated updates using a stored world model.

**World Model Structure:**

```math
M : (s, a) \rightarrow (s', r_{\text{aug}})
```

Each entry is updated at each real step:

```math
M(s, a) \leftarrow (s', r_{\text{aug}})
```

For a deterministic environment, this model is **perfect after a single visit**: the same (s, a) always produces the same (s', r).

**Planning Loop (K = 5 simulated updates):**

```math
(s_{\text{sim}}, a_{\text{sim}}) \sim \text{Uniform}(\text{dom}(M))
```

```math
(s'_{\text{sim}}, r_{\text{sim}}) \leftarrow M(s_{\text{sim}}, a_{\text{sim}})
```

```math
y_{\text{sim}} = r_{\text{sim}} + \gamma (1 - d) \max_{a'} Q(s'_{\text{sim}}, a';\, \theta)
```

```math
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_{\text{Huber}}(y_{\text{sim}},\; Q(s_{\text{sim}}, a_{\text{sim}};\, \theta))
```

**Effective Update Ratios:**

```math
K_{\text{eff}}^{\text{normal}} = 1 + K_{\text{planning}} = 1 + 5 = 6
```

```math
K_{\text{eff}}^{\text{breakthrough}} = 1 + 4K = 1 + 20 = 21 \quad \text{(Super Brain Mode)}
```

**Important design choice:** Dyna-Q simulated transitions are not added to the PER buffer. Only real environment steps (via the N-step buffer) enter PER. This separation prevents model-derived transitions from crowding out high-TD-error real transitions — the PER's primary function.

---

### 9.3 Huber Loss — Motivation and Derivation

RIENFoRZe-I uses MSE loss. RIENFoRZe-II replaces it with **Huber loss** (Huber, 1964), which is more robust to large TD errors in early training.

**Definition (δ = 1.0):**

```math
\mathcal{L}_{\delta}(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & |y - \hat{y}| \leq \delta \\ \delta \cdot |y - \hat{y}| - \frac{1}{2}\delta^2 & |y - \hat{y}| > \delta \end{cases}
```

**Gradient:**

```math
\frac{\partial \mathcal{L}_\delta}{\partial \hat{y}} = \begin{cases} \hat{y} - y & |y - \hat{y}| \leq \delta \\ -\delta \cdot \text{sign}(y - \hat{y}) & |y - \hat{y}| > \delta \end{cases}
```

**Why Huber over MSE.** MSE produces gradients proportional to TD error:

```math
\frac{\partial}{\partial \hat{y}} (y - \hat{y})^2 = -2(y - \hat{y})
```

For large TD errors (common when Q-values are poorly initialized), this produces very large gradients even after clipping. Huber caps the gradient magnitude at `δ` regardless of error size, providing L2 precision for small errors and L1 robustness for large errors.

---

### 9.4 Epsilon Schedule Correction

The v-I epsilon bug (per-train-step decay) is corrected in v-II to **per-episode decay**.

```math
\epsilon_{n+1} = \max(\epsilon_{\min},\; \epsilon_n \cdot \lambda_\epsilon)
```

Applied exactly once at episode termination (`done = True`).

**Theoretical schedule:** with per-episode decay at rate λ = 0.995 from ε₀ = 0.7 to ε_min = 0.05:

```math
\epsilon_n = \max\!\left(0.05,\; 0.7 \cdot 0.995^n\right)
```

Reaches minimum after:

```math
n^{*} = \left\lceil \frac{\ln(0.05 / 0.7)}{\ln(0.995)} \right\rceil = \left\lceil \frac{-2.639}{-0.00501} \right\rceil = 527 \text{ episodes}
```

---

### 9.5 Super Brain Mode

When the agent successfully reaches the goal (`r_episode > 10.0` and `done = True`), the Dyna-Q planning multiplier is activated:

```math
K_{\text{actual}} = \begin{cases} K & \text{normal episode} \\ 4K & \text{if done and } r_{\text{episode}} > 10.0 \end{cases}
```

For v-II with K = 5: breakthrough planning = 20 cycles. The rationale is that a goal-reaching trajectory contains a complete success signal. Intensive replay of this trajectory's world model updates amplifies backward value propagation from the terminal reward — a mechanism analogous to **memory replay consolidation** in biological systems, where high-salience events receive elevated hippocampal replay during quiescent periods.

---

### 9.6 Numerical Stability Shields

RIENFoRZe-I experienced numerical overflow during extended training runs. Two hard clamps are introduced:

**Q-value shield:**

```math
Q(s, a) \leftarrow \text{clip}(Q(s, a),\; -10^6,\; +10^6)
```

**Weight shield (post-Adam update):**

```math
\theta_p \leftarrow \text{clip}(\theta_p,\; -100.0,\; +100.0)
```

The weight bound of ±100 is intentionally generous — it intervenes only in genuine overflow scenarios, not in normal training dynamics. These shields operate independently and on different timescales:

- Q clamp: acts immediately before any Bellman computation
- Weight clamp: acts immediately after each Adam update

---

## 10. RIENFoRZe-III — Full Sensory Architecture (64D)

RIENFoRZe-III is the apex of the gradient-based architectural family. It preserves the entire v-II stack and adds twelve new sensory dimensions through five new sensing modalities, motivated by three failure modes observed in v-II.

| Failure Mode in v-II | New Component | Mechanism |
|---|---|---|
| Tunnel blindness (walls >2 cells away invisible) | Cardinal wall radar (4D) | Raycasts along 4 axes |
| Gradient invisibility (agent cannot compute exploration direction) | Scent gradients (4D) | Log-ratio of visit counts |
| Goal ambiguity (distance loses direction at range) | Target beacon (2D) | Unit vector toward goal |
| Pheromone summarization inadequate | Local flux (1D) | Standard deviation of cross values |
| Curiosity signal not policy-accessible | Curiosity encoding (1D) | ICM bonus directly in state |

**64D Dimension Accounting:**

```
25 (vision) + 13 (pheromone) + 2 (pos) + 2 (tpos) + 2 (dir) + 1 (dist) + 1 (trap)
+ 1 (fog) + 1 (time) + 4 (momentum) + 4 (radar) + 4 (scent) + 2 (beacon) + 1 (flux) + 1 (curiosity) = 64
```

---

### 10.1 Cardinal Wall Radar — Raycast Sensing

For each cardinal direction (N, S, E, W) = `{(−1,0),(1,0),(0,−1),(0,1)}`, the radar casts a ray and returns the normalized distance to the first wall encountered:

```math
d_{\text{radar}}(dr, dc) = \frac{i^{*}}{10}, \quad i^{*} = \min\!\left\{i \in \{1, \ldots, 10\} : \text{maze}[r + i \cdot dr,\; c + i \cdot dc] = 1\right\}
```

capped at `i = 10` if no wall is found. The result is normalized to [0.1, 1.0]: nearest detectable wall is 0.1, clear corridor beyond range is 1.0.

**Geometric interpretation.** The four radar values define an implicit bounding box:

```math
\text{corridor\_length}(\text{dir}) = 10 \cdot d_{\text{radar}}(\text{dir})
```

The aspect ratio of this box (long along current direction vs short perpendicular) is implicitly available to the network. For a Level-10 maze (41 columns), corridors up to 10 cells long are fully visible to the radar — covering 24% of the maze width in a single measurement.

---

### 10.2 Scent Gradient Channels — Logarithmic Visit Differential

Let `visit_grid[r, c]` be the accumulated visit count. The scent gradient in each cardinal direction encodes the log-ratio of neighbor visits to current-cell visits:

```math
g_{\text{scent}}(dr, dc) = \text{clip}\!\left(\log\!\left(\frac{1 + \text{visit}[r + dr, c + dc]}{1 + \text{visit}[r, c]}\right),\; -1,\; 1\right)
```

**Logarithmic rationale.** Raw visit counts are heavily right-skewed: frequently visited cells can accumulate thousands of visits while novel cells have 0–5. The log transformation compresses this dynamic range. The difference of logs produces a log-ratio:

- Positive scent gradient: the neighbor has been visited more (less novel)
- Negative scent gradient: the neighbor has been visited less (more novel)
- Zero: equal exploration history

The clipping to [−1, 1] prevents occasional large gradients from dominating the network input. Note that this gradient is **not** recoverable from the 13 absolute pheromone values: the denominator (current cell's visit count) changes each step and is not separately encoded.

---

### 10.3 Target Beacon — Unit Direction Encoding

v-I and v-II encode only scalar Manhattan distance to the target (magnitude without direction). v-III adds a **unit direction vector** pointing toward the target:

```math
\vec{u}_{\text{beacon}} = \frac{(r_{\text{target}} - r_{\text{agent}},\; c_{\text{target}} - c_{\text{agent}})}{\|(r_{\text{target}} - r_{\text{agent}},\; c_{\text{target}} - c_{\text{agent}})\|_2 + \epsilon}, \quad \epsilon = 10^{-9}
```

The Manhattan distance (retained in `telemetry`) and the beacon vector provide complementary information:

```math
\text{goal\_info} = (d_{\text{manhattan}},\; u_r,\; u_c)
```

The beacon is scale-invariant: a target 5 cells away and a target 50 cells away in the same direction produce the same beacon vector. The distance captures magnitude. Together they span more of the goal-geometry information space than either alone.

---

### 10.4 Local Flux — Pheromone Variance Signal

The 13-cell pheromone cross provides raw density values. The **flux** scalar summarizes their statistical spread:

```math
\text{flux} = \text{std}\!\left(P_{\text{norm}}[c_0], P_{\text{norm}}[c_1], \ldots, P_{\text{norm}}[c_{12}]\right)
```

High flux indicates uneven pheromone distribution — the agent is near a frontier between explored and unexplored territory. Low flux indicates uniform distribution — either fully explored or fully unexplored local neighborhood. The network can learn to treat high-flux states as requiring more deliberate action selection and low-flux states as routine navigation.

---

### 10.5 Curiosity Self-Referential Loop

This is the most structurally novel feature of v-III. In v-I and v-II, the ICM bonus `r_i(s)` influences only the reward signal. In v-III, the ICM bonus is also **directly encoded into the state vector** as the final dimension:

```math
s_{64} = \text{clip}(\text{last\_icm\_bonus},\; 0,\; 1)
```

**Mathematical closed loop.** The curiosity signal at time t is:

```math
\text{icm}(s_t) = \frac{\beta}{\sqrt{N(k(s_t))}}
```

The next state embedding includes this value as its 64th component. Therefore the agent's policy at t+1 is conditioned on the curiosity at t:

```math
\pi(a \mid s_{t+1}) = \pi\!\left(a \mid \bigl[\underbrace{s_{t+1}^{1:63}}_{\text{environment}},\; \underbrace{\text{icm}(s_t)}_{\text{metacognition}}\bigr]\right)
```

If the network learns to use `s_{64}` effectively, the agent can produce **curiosity-seeking behavior** that derives directly from observing its own novelty drive — a primitive form of metacognitive action selection. This is structurally related to meta-reinforcement learning: the agent learns a policy over states that includes its own internal learning signal as an observable.

---

### 10.6 Accelerated Dyna-Q — Instant Breakthrough

v-III dramatically scales the Dyna-Q planning budget:

| Condition | Planning Steps |
|---|---|
| Normal step | 25 |
| Episode success (`r > 20.0`, `done = True`) | 25 × 5 = **125** |

**Effective learning multiplier per environment step:**

```math
K_{\text{eff}}^{\text{normal}} = 1 + 25 = 26
```

```math
K_{\text{eff}}^{\text{breakthrough}} = 1 + 125 = 126
```

**Total Q-updates for a 200-step successful episode:**

```math
\text{updates} = 200 \times 26 + 125 = 5{,}325
```

**Value propagation speed.** With standard Q-learning, value information propagates backward at approximately one cell per episode. With 25-step Dyna-Q, value information can propagate up to 25 steps backward in a single planning phase. For a Level-10 maze (35×41, optimal path ≈ 70 steps), full value propagation requires at minimum 3 real episodes with 25-step planning, compared to approximately 70 without planning.

---

### 10.7 Backpropagation Through the Dueling Architecture

Full gradient derivation for one training step on a batch of size B.

**Dueling head gradient routing.** Let `dQ ∈ R^{B×4}` be the loss gradient with respect to Q-outputs:

```math
\frac{\partial \mathcal{L}}{\partial V(s_i)} = \sum_{a} \frac{\partial \mathcal{L}}{\partial Q(s_i, a)} \cdot \frac{\partial Q}{\partial V} = \sum_{a} dQ_{i,a}
```

```math
\frac{\partial \mathcal{L}}{\partial A(s_i, a)} = dQ_{i,a} - \frac{1}{4} \sum_{a'} dQ_{i,a'} = dQ_{i,a} - \overline{dQ}_i
```

These route through `W_val` and `W_adv` respectively and sum at `h₃`:

```math
\frac{\partial \mathcal{L}}{\partial h_3} = \frac{\partial \mathcal{L}}{\partial V} W_{\text{val}}^T + \frac{\partial \mathcal{L}}{\partial A} W_{\text{adv}}^T
```

**Hidden layer gradients:**

```math
d_{z3} = d_{h3} \odot f'_{\text{leaky}}(z_3)
```

```math
dW_3 = h_2^T d_{z3} / B, \quad db_3 = \text{mean}(d_{z3},\; \text{axis}=0)
```

```math
d_{h2} = d_{z3} W_3^T, \quad d_{z2} = d_{h2} \odot f'_{\text{leaky}}(z_2)
```

```math
dW_2 = h_1^T d_{z2} / B, \quad db_2 = \text{mean}(d_{z2},\; \text{axis}=0)
```

```math
d_{h1} = d_{z2} W_2^T, \quad d_{z1} = d_{h1} \odot f'_{\text{leaky}}(z_1)
```

```math
dW_1 = x^T d_{z1} / B, \quad db_1 = \text{mean}(d_{z1},\; \text{axis}=0)
```

All gradient tensors are element-wise clipped to [−10, 10] before the Adam update. After the Adam update, all weight matrices are clamped to [−100, 100].

The gradient routing enforces interpretability of the dueling decomposition throughout training: `V(s)` is updated by the **sum signal** (total action value), while `A(s, a)` is updated by the **deviation signal** (relative action advantage).

---

### 10.8 Information-Theoretic Analysis of 64D vs 52D

Each new dimension contributes to the state representation insofar as it reduces uncertainty about the optimal action. For new feature `X` given the existing 52D state:

```math
I(A^{*};\; X \mid s_{1:52}) = H(A^{*} \mid s_{1:52}) - H(A^{*} \mid s_{1:52}, X)
```

Features are non-redundant if this quantity is strictly positive.

**Redundancy analysis:**

- **Beacon vs. Manhattan distance:** Distance encodes magnitude only; beacon encodes direction only. These are complementary and nearly orthogonal in information content.
- **Radar vs. vision:** Vision provides dense 2-step coverage; radar provides sparse 10-step coverage along 4 axes. Radar detects structures invisible to the local window.
- **Scent vs. pheromone cross:** Pheromone cross provides absolute density values. Scent provides the spatial gradient (rate of change), which is not recoverable from absolute values alone without knowing the current cell's count (which changes each step).
- **Curiosity encoding vs. ICM reward:** The ICM reward is a scalar summed into episode return. The curiosity state observation is accessible per-step to condition the policy directly, before accumulation.

None of the 12 new dimensions are linearly predictable from the existing 52, supporting their inclusion as non-redundant contributions to the observation space.

---

## 11. RIENFoRZe-IV — Tabular Dyna-Q Architecture (52D, Exact)

RIENFoRZe-IV is a **paradigm shift**: the entire neural network stack — weights, gradients, Adam moments, backpropagation — is discarded. A pure tabular Q-learning agent with a perfect world model takes its place.

> *"The neural architecture remains in the codebase, commented out, as an architectural fossil."*

---

### 11.1 The Approximation Error Argument

A neural Q-function approximates:

```math
Q_\theta(s, a) \approx Q^{*}(s, a)
```

In a deterministic bounded environment, this approximation introduces three classes of instability that need not exist:

**1. Function approximation error.** The network may lack capacity to represent `Q*` exactly for large state spaces. In a deterministic maze, `Q*` is a piecewise function with sharp transitions at wall boundaries — difficult for smooth neural approximators to represent without high capacity.

**2. Gradient interference (Deadly Triad).** Updating `Q(s, a)` via gradient descent may change `Q(s', a')` for nearby states `s'` due to weight sharing. This interference — off-policy training + bootstrapping + function approximation — is the **Deadly Triad** (Sutton & Barto, 2018) and is a fundamental source of DQN instability.

**3. Bootstrapping bias from moving targets.** The Bellman target is computed using the (lagged) target network, which is itself changing. This introduces a moving-target problem that never fully resolves.

None of these issues arise for a tabular agent in a deterministic, bounded environment: Q-table entries are exact, updates are isolated to their specific (s, a) pair, and targets are fixed once computed.

---

### 11.2 Tabular Q-Function and State Hashing

The Q-function is a Python dictionary:

```python
q_table: Dict[Tuple[int, ...], float]
q_table[(s_key, action)] = q_value
```

**State Discretization.** The 52D continuous observation vector is quantized to a hashable integer tuple with `bins = 16` per dimension:

```math
k_i = \left\lfloor \text{clip}(s_i, 0, 1) \cdot (B - 1) \right\rfloor \in \{0, 1, \ldots, 15\}, \quad B = 16
```

The resulting 52-tuple of integers serves as the dictionary key.

**Default Initialization:**

```math
Q(s, a) = 0.0 \quad \text{if } (s, a) \notin \text{dom}(Q_{\text{table}})
```

**Maximum possible key space:**

```math
|\mathcal{K}| \leq 16^{52} \approx 10^{62.7}
```

In practice, the agent visits a small fraction of this space. The Q-table grows monotonically with experience: after N unique (state, action) observations, the table has exactly N entries. Unlike a neural network with fixed parameter count, the tabular Q-function's memory is proportional to experience.

**Tie-Breaking.** When multiple actions share the maximum Q-value (common in early training when the table is sparse with default 0.0 everywhere), a uniform random tie-break is applied:

```math
a^{*} \sim \text{Uniform}\!\left(\left\{ a : Q(k(s), a) = \max_{a'} Q(k(s), a') \right\}\right)
```

Without this, the agent would always select action 0 (up) when all Q-values are equal, introducing a systematic directional bias.

---

### 11.3 Exact Q-Learning Update Rule

For each transition `(s, a, r_aug, s', done)`:

**Step 1 — Lookup:**

```math
Q_{\text{current}} = Q(k(s), a)
```

```math
Q^{*}_{\text{next}} = \max_{a' \in \mathcal{A}} Q(k(s'), a')
```

**Step 2 — Bellman Target:**

```math
y = r_{\text{aug}} + (1 - \mathbb{1}[\text{done}]) \cdot \gamma \cdot Q^{*}_{\text{next}}
```

**Step 3 — TD Error:**

```math
\delta = y - Q_{\text{current}}
```

**Step 4 — Exact Update:**

```math
Q(k(s), a) \leftarrow Q_{\text{current}} + \alpha \cdot \delta, \quad \alpha = 0.3
```

This update modifies exactly one dictionary entry. No other entry is affected — the Deadly Triad is structurally absent.

---

### 11.4 Convergence Guarantees — Tabular vs Approximate RL

**Tabular Q-learning convergence theorem** (Watkins and Dayan, 1992): for a finite MDP, Q-learning converges to `Q*` almost surely provided:

1. Every `(s, a)` pair is visited infinitely often
2. The learning rate satisfies the **Robbins-Monro conditions**:

```math
\sum_{t=0}^{\infty} \alpha_t = \infty, \quad \sum_{t=0}^{\infty} \alpha_t^2 < \infty
```

With constant `α = 0.3`, condition 2 is violated (the sum of constants diverges). This is deliberate: in a deterministic environment with exact targets, the optimal update is a simple overwrite (`α = 1.0` would converge in a single visit per `(s, a)` pair). `α = 0.3` provides a damped average that handles residual noise from the curiosity bonus.

**Neural DQN does not have this guarantee** because: (1) function approximation is not guaranteed to represent `Q*` exactly; (2) the Deadly Triad creates instability; (3) the target network introduces a moving target.

**Tabular + complete model = value iteration.** Once every `(s, a)` pair has been visited at least once, the world model `M` is complete and exact. Subsequent Dyna-Q planning steps perform pure dynamic programming over an exact model — equivalent to value iteration:

```math
Q_{k+1}(s, a) = R(s, a) + \gamma \max_{a'} Q_k(s', a')
```

This is guaranteed to converge to `Q*` for finite MDPs.

---

### 11.5 Perfect World Model — Dyna-Q Without Approximation

The world model stores exact transitions:

```python
model: Dict[Tuple, Tuple]
model[(k(s), action)] = (k(s'), r_aug)
```

For a deterministic MDP:

```math
P(s' \mid s, a) = \mathbb{1}[s' = T(s, a)]
```

The model `M(s, a) = (T(s, a), R(s, a))` is correct with probability 1 after a single visit. This is in sharp contrast to model-based RL in stochastic environments, which requires multiple observations to estimate transition distributions.

**Planning loop (K = 20 steps):**

```math
(s_{\text{sim}}, a_{\text{sim}}) \sim \text{Uniform}(\text{dom}(M))
```

```math
Q(s_{\text{sim}}, a_{\text{sim}}) \leftarrow Q(s_{\text{sim}}, a_{\text{sim}}) + 0.3 \cdot \left(r_{\text{sim}} + \gamma \max_{a'} Q(s'_{\text{sim}}, a') - Q(s_{\text{sim}}, a_{\text{sim}})\right)
```

**Super Brain Mode (on success, `r > 10.0`):**

```math
K_{\text{breakthrough}} = 4 \times 20 = 80 \text{ planning steps}
```

**Effective update ratios:**

| Condition | Q-updates per real step |
|---|---|
| Normal | 1 + 20 = 21 |
| Breakthrough episode (last step) | 1 + 80 = 81 |

**Value propagation probability.** With uniform sampling from model of size |M|:

```math
\Pr[\text{goal's predecessor sampled}] = \frac{|\text{predecessors of G}|}{|M|}
```

Prioritized sweeping (as in the MazE companion module) would improve this probability. The choice of uniform sampling here tests whether raw planning volume (20 steps) compensates for lack of priority ordering.

---

### 11.6 Memory Complexity Analysis

**Q-table (v-IV):**

```math
\text{mem}(Q) = O(|S| \cdot 4 \cdot 8 \text{ bytes}) = O(32|S| \text{ bytes})
```

**World model (v-IV):**

```math
\text{mem}(M) = O(|S| \cdot 4 \cdot (52 \cdot 4 + 8) \text{ bytes}) = O(216 |S| \text{ bytes})
```

**PER buffer (v-I/II/III) — absent in v-IV:**

```math
\text{mem}(\text{PER}) = 50{,}000 \cdot (52 + 1 + 1 + 52 + 1) \cdot 4 \text{ bytes} \approx 20.9 \text{ MB}
```

plus segment tree overhead (~6.4 MB) → total ≈ **27 MB**. Entirely eliminated in v-IV.

**Neural network (v-I/II/III) — absent in v-IV:**

```math
\text{mem}(\theta) = 58{,}117 \cdot 8 \text{ bytes} \approx 465 \text{ KB}
```

Two copies (online + target) plus two Adam moment copies: 4 × 465 KB ≈ **1.86 MB**. Also eliminated.

For small mazes (`|S| ~ 10³` to `10⁴`), the v-IV Q-table and world model together consume well under **1 MB** — at least an order of magnitude less than the neural versions.

---

### 11.7 Why Learning Rate is 0.3, Not 0.001

Neural Adam uses `η = 0.001` because gradient-based updates are noisy (mini-batch variance), the loss landscape has saddle points, and large learning rates cause oscillation.

Tabular Q-learning uses `α = 0.3` because updates are exact (no sampling variance), there is no loss landscape, and the theoretically optimal `α` for a deterministic environment is 1.0.

**Why not α = 1.0:** Even in a deterministic environment, the augmented reward `r_aug = r_ext + r_i(s)` contains stochasticity because `r_i(s)` varies with the curiosity count history. With `α = 1.0`, each update completely overwrites the previous estimate. With `α = 0.3`, the Q-value is a damped average of recent targets.

**Half-life of information under α = 0.3:**

```math
t_{1/2} = \frac{\ln 2}{\ln\!\left(\frac{1}{1 - \alpha}\right)} = \frac{\ln 2}{\ln(1/0.7)} = \frac{0.693}{0.357} \approx 1.94 \text{ visits}
```

Within approximately 2 visits to the same `(s, a)` pair, the Q-value is dominated by the most recent information. Old, potentially stale estimates decay rapidly.

---

### 11.8 JSON Serialization of the Q-Table

Python tuple keys are not JSON-serializable. v-IV implements string-key serialization:

```python
# Save
{"q_table": {str(k): v for k, v in self.q_table.items()}}

# Load (string → tuple via ast.literal_eval)
{ast.literal_eval(k): v for k, v in d["q_table"].items()}
```

The saved state is packaged as a ZIP archive containing `weights.json`, `config.json`, and `stats.json`, mirroring the neural versions for API compatibility. This means all four RIENFoRZe versions can be loaded, inspected, and resumed using the same interface.

---

## 12. MazE Companion Module — SARSA with Prioritized Sweeping

`MazE.py` (referenced as a standalone Streamlit application) implements an independent **SARSA + Prioritized Sweeping** agent for pedagogical comparison and ablation. It is architecturally distinct from the main DQN/tabular agent across all four versions.

**SARSA Update Rule (on-policy TD(0)):**

```math
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]
```

where `a_{t+1}` is the **actually selected** next action (not the greedy maximum). This makes SARSA's policy estimate more conservative near dangerous states (traps): because SARSA accounts for the probability of taking exploratory actions, it implicitly penalizes paths that pass near traps even if the greedy action would avoid them.

**BFS Distance Map.** At initialization, BFS from the goal computes an exact shortest-path distance map `D` over all reachable cells. This serves as a dense reward shaping potential:

```math
r_{\text{shaped}}(s, s') = r(s, s') + \gamma \cdot D(s) - D(s')
```

The shaped reward is dense everywhere in the maze, eliminating the sparse reward problem even in large Level-10 configurations.

**Curiosity-Weighted Exploration:** Visit counts are maintained per cell. The curiosity weight decays exponentially:

```math
r_{\text{curiosity}}(s) = \frac{w_c}{\text{visit}(s) + 1}, \quad w_c \leftarrow 0.99 \cdot w_c \text{ (per episode)}
```

After approximately 459 episodes, `w_c < 0.01`, effectively disabling intrinsic exploration. This schedule naturally transitions from exploration-driven (early training) to reward-driven (late training) behavior.

**Prioritized Sweeping.** After each real step, simulated updates are performed in order of predicted TD error magnitude, focusing compute on the states where the value function is most outdated:

```math
\text{priority}(s, a) = \left| r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right|
```

This is more efficient than the uniform random sampling used in v-IV's Dyna-Q loop, at the cost of maintaining a priority queue over the world model.

---

## 13. Cross-Version Architectural Comparison

| Property | v-I | v-II | v-III | v-IV |
|---|---|---|---|---|
| State dim | 17 | 52 | 64 | 52 |
| Q-function | Neural | Neural | Neural | Tabular |
| Parameters (Q-fn) | 49,124 | 55,049 | 58,117 | O(|S|) |
| Loss | MSE | Huber | Huber | None |
| Convergence guarantee | No | No | No | Yes (finite MDP) |
| PER buffer | 50K | 50K | 50K | None |
| N-step (n=3) | Yes | Yes | Yes | None |
| Target network | Yes (τ=0.005) | Yes | Yes | None |
| Adam optimizer | Yes | Yes | Yes | None |
| Weight clamp | No | ±100 | ±100 | None |
| Dyna-Q steps | 0 | 5 | 25 | 20 |
| Breakthrough (on success) | None | 4× (20) | 5× (125) | 4× (80) |
| Vision field | 3×3 | 5×5 | 5×5 | 5×5 |
| Pheromone channel | No | 13-cell | 13-cell | 13-cell |
| Wall radar | No | No | 4D | No |
| Scent gradients | No | No | 4D | No |
| Target beacon | No | No | 2D unit vec | No |
| Pheromone flux | No | No | 1D std | No |
| Curiosity in state | No | No | 1D | No |
| Epsilon timing | Per step (bug) | Per episode | Per episode | Per episode |
| Epsilon start | 1.0 | 0.7 | 0.7 | 0.7 |
| Memory (est.) | ~29 MB | ~29 MB | ~29 MB | <1 MB |
| Gradient clipping | ±10 | ±10 | ±10 | N/A |

**Effective Q-update multiplier per environment step:**

| Version | Normal | Breakthrough |
|---|---|---|
| v-I | 1× | 1× |
| v-II | 6× | 21× |
| v-III | 26× | 126× |
| v-IV | 21× | 81× |

---

## 14. Consolidated Hyperparameter Reference

### Shared Across All Versions

| Parameter | Value | Description |
|---|---|---|
| `action_size` | 4 | Discrete actions: Up, Down, Left, Right |
| `gamma` | 0.99 | Discount factor |
| `h1, h2, h3` | 256, 128, 64 | Hidden layer widths |
| `alpha_per` | 0.6 | PER priority exponent |
| `beta_start` | 0.4 | IS weight annealing start |
| `beta_frames` | 100,000 | IS annealing duration |
| `n_steps` | 3 | N-step return horizon |
| `tau` | 0.005 | Polyak averaging coefficient |
| `batch_size` | 64 | Training batch size |
| `icm_beta` | 0.05 | Curiosity bonus scale |
| `icm_bins` | 16 | State discretization bins |
| `promote_thresh` | 0.72 | Curriculum promotion threshold |
| `demote_thresh` | 0.25 | Curriculum demotion threshold |
| `curriculum_window` | 20 | Rolling evaluation window |
| `gradient_clip` | 10.0 | Element-wise gradient bound |

### Version-Specific

| Parameter | v-I | v-II | v-III | v-IV |
|---|---|---|---|---|
| `state_size` | 17 | 52 | 64 | 52 |
| `buffer_size` | 50,000 | 50,000 | 50,000 | N/A |
| `lr` | 0.001 | 0.001 | 0.001 | 0.3 (tabular α) |
| `epsilon_start` | 1.0 | 0.7 | 0.7 | 0.7 |
| `epsilon_min` | 0.04 | 0.05 | 0.05 | 0.05 |
| `epsilon_decay` | 0.997 | 0.995 | 0.995 | 0.995 |
| `epsilon_timing` | per step | per episode | per episode | per episode |
| `planning_steps` | 0 | 5 | 25 | 20 |
| `breakthrough_mult` | N/A | 4× | 5× | 4× |
| `breakthrough_threshold` | N/A | r > 10.0 | r > 20.0 | r > 10.0 |
| `loss` | MSE | Huber δ=1 | Huber δ=1 | N/A |
| `q_clip` | N/A | ±10⁶ | ±10⁶ | N/A |
| `weight_clip` | N/A | ±100 | ±100 | N/A |
| `vision_radius` | 1 (3×3) | 2 (5×5) | 2 (5×5) | 2 (5×5) |
| `radar_max_range` | N/A | N/A | 10 cells | N/A |
| `scent_clip` | N/A | N/A | ±1.0 | N/A |
| `beacon_epsilon` | N/A | N/A | 10⁻⁹ | N/A |
| `lr_patience` | 100 | 100 | 100 | N/A |
| `lr_factor` | 0.5 | 0.5 | 0.5 | N/A |
| `lr_min` | 10⁻⁵ | 10⁻⁵ | 10⁻⁵ | N/A |
| `total_net_params` | 49,124 | 55,049 | 58,117 | N/A |

---

## 15. File Architecture

```
Evolving-AI-main/
│
├── brain.py              RL engine: Dueling DDQN, PER, N-Step, ICM,
│                         Curriculum, Adam optimizer, Polyak updates (~746 lines)
│
├── world.py              Environment: maze generation (Backtracker/Prim/Wilson/Hybrid),
│                         fog-of-war, dynamic traps, portals, reward shaping, A* (~744 lines)
│
├── soul.py               Cognitive architecture: Russell emotion model, OCEAN personality,
│                         intent NLP, episodic memory decay, relationship dynamics (~717 lines)
│
├── memory_palace.py      Memory system: working/episodic/semantic memory, CLS-inspired
│                         architecture, JSON persistence, fact confidence tracking (~529 lines)
│
├── analytics.py          Telemetry: rolling statistics, EMA, convergence detection,
│                         capability score, session export (~414 lines)
│
├── RIENFoRZe.py          Primary entry point and orchestration (~1,495 lines)
│
├── RIENFoRZeADv.py       Advanced Streamlit research UI (~2,077 lines)
│
├── requirements.txt      Dependency manifest
│
├── LICENSE               Apache 2.0
│
└── documentation/
    ├── Original_Readme.md        Initial project description
    ├── README_RIENFoRZe_I.md     Foundational architecture reference (17D)
    ├── README_RIENFoRZe_II.md    52D extended architecture reference
    ├── README_RIENFoRZe_III.md   64D full sensory architecture reference
    └── README_RIENFoRZe_IV.md    Tabular Dyna-Q architecture reference
```

**Total codebase: ~6,722 lines across 7 Python files.**

---

## 16. Installation and Usage

**Dependencies:**

```
numpy
streamlit
pandas
```

No deep learning framework is required. All matrix operations are explicit NumPy.

**Installation:**

```bash
git clone https://github.com/Devanik21/Evolving-AI.git
cd Evolving-AI
pip install -r requirements.txt
```

**Launch (standard interface):**

```bash
python RIENFoRZe.py
```

**Launch (advanced Streamlit research UI):**

```bash
streamlit run RIENFoRZeADv.py
```

**Quick orientation in the Streamlit UI:**

1. Select a RIENFoRZe version from the sidebar (I through IV)
2. Toggle **Run Autonomously** to begin training
3. The **Maze** panel shows the procedurally generated environment with the agent's trajectory overlaid
4. The **Learning Curves** panel shows reward, loss (where applicable), epsilon, and Q-value distribution
5. The **Emotion State** panel shows the current valence-arousal position and personality trait summary
6. The **Memory** panel shows episodic records and semantic facts accumulated across sessions
7. The **Research Lab** tab provides an inline reference for all equations and architectural decisions

**Architecture selection guidance:**

- **RIENFoRZe-I:** Baseline. Useful for establishing reference performance and verifying the core DQN pipeline.
- **RIENFoRZe-II:** Recommended for most experimental runs. Balanced state richness, Dyna-Q planning, and stable training via Huber loss and numerical shields.
- **RIENFoRZe-III:** For experiments focused on maximal observation richness, the curiosity self-referential loop, or intensive Dyna-Q planning at 25×/125× multipliers.
- **RIENFoRZe-IV:** For experiments studying tabular methods, convergence guarantees, or memory efficiency. Converges faster in low-complexity mazes (Levels 1–5) but may require more real episodes in high-complexity configurations where the state space exceeds practical tabular coverage.

---

*Project A.L.I.V.E. NEXUS — Master Reference Document*  
*RIENFoRZe Series (I–IV) — April 2026*  
*Devanik · [github.com/Devanik21](https://github.com/Devanik21)*
