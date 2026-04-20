"""
analytics.py — Project A.L.I.V.E. NEXUS
Performance Analytics & Telemetry Engine

Features:
  • Rolling statistics with configurable window sizes
  • Convergence / plateau detection
  • Exploration heatmap analysis
  • Agent capability scoring (composite metric)
  • Learning phase classification
  • Session export & report generation
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
import time
import json
import math


# ============================================================
#  STATISTICAL UTILITIES
# ============================================================
def rolling_mean(data: deque, window: int = None) -> float:
    if not data:
        return 0.0
    arr = list(data)
    if window:
        arr = arr[-window:]
    return float(np.mean(arr))


def rolling_std(data: deque, window: int = None) -> float:
    if len(data) < 2:
        return 0.0
    arr = list(data)
    if window:
        arr = arr[-window:]
    return float(np.std(arr))


def exponential_moving_average(values: List[float], alpha: float = 0.1) -> List[float]:
    if not values:
        return []
    ema = [values[0]]
    for v in values[1:]:
        ema.append(alpha * v + (1 - alpha) * ema[-1])
    return ema


def linear_trend(values: List[float]) -> float:
    """Returns the slope of a linear fit to the last N values."""
    if len(values) < 2:
        return 0.0
    n = len(values)
    x = np.arange(n)
    slope = np.polyfit(x, values, 1)[0]
    return float(slope)


# ============================================================
#  CONVERGENCE DETECTOR
# ============================================================
class ConvergenceDetector:
    """
    Detects whether the learning curve has converged, plateaued,
    is still improving, or has regressed.
    """
    STATES = ['warming_up', 'rapid_learning', 'fine_tuning', 'converged', 'plateau', 'regressing']

    def __init__(self, window: int = 50, tolerance: float = 0.01):
        self.window    = window
        self.tolerance = tolerance
        self.state     = 'warming_up'
        self._history: deque = deque(maxlen=window * 3)
        self._state_duration = 0

    def update(self, value: float) -> str:
        self._history.append(value)
        self._state_duration += 1

        if len(self._history) < self.window:
            self.state = 'warming_up'
            return self.state

        recent = list(self._history)[-self.window:]
        older  = list(self._history)[-self.window*2:-self.window] if len(self._history) >= self.window*2 else recent

        recent_mean = np.mean(recent)
        older_mean  = np.mean(older)
        recent_std  = np.std(recent)
        trend       = linear_trend(recent)

        prev_state = self.state

        if trend > self.tolerance:
            self.state = 'rapid_learning' if trend > self.tolerance * 3 else 'fine_tuning'
        elif trend < -self.tolerance:
            self.state = 'regressing'
        elif recent_std < self.tolerance * 2:
            self.state = 'converged'
        else:
            self.state = 'plateau'

        if self.state != prev_state:
            self._state_duration = 0

        return self.state

    @property
    def state_icon(self) -> str:
        return {
            'warming_up':     '🔥',
            'rapid_learning': '🚀',
            'fine_tuning':    '⚙️',
            'converged':      '✅',
            'plateau':        '📊',
            'regressing':     '⬇️',
        }.get(self.state, '❓')

    @property
    def state_description(self) -> str:
        return {
            'warming_up':     'Filling replay buffer. Learning hasn\'t begun.',
            'rapid_learning': 'Strong positive gradient. Policy improving rapidly.',
            'fine_tuning':    'Gradual improvement. Policy refining.',
            'converged':      'Performance stable. Policy has reached equilibrium.',
            'plateau':        'No clear trend. High variance. Might be stuck.',
            'regressing':     'Negative trend detected. Check hyperparameters.',
        }.get(self.state, '')


# ============================================================
#  EPISODE TRACKER
# ============================================================
class EpisodeTracker:
    """Records and analyzes episode-level statistics."""
    def __init__(self, maxlen: int = 1000):
        self.episodes: List[Dict] = []
        self.maxlen = maxlen

        # Rolling deques for fast stats
        self.rewards   : deque = deque(maxlen=200)
        self.steps     : deque = deque(maxlen=200)
        self.successes : deque = deque(maxlen=200)
        self.losses    : deque = deque(maxlen=500)
        self.td_errors : deque = deque(maxlen=500)
        self.epsilons  : deque = deque(maxlen=500)
        self.optimality: deque = deque(maxlen=200)

        self.convergence = ConvergenceDetector()
        self.total_steps = 0
        self._session_start = time.time()

    def record_episode(self, reward: float, steps: int, success: bool,
                       info: Dict = None):
        info = info or {}
        ep = {
            'reward':      round(reward, 3),
            'steps':       steps,
            'success':     success,
            'timestamp':   time.time(),
            'optimality':  info.get('optimality', 0.0),
            'fog_coverage':info.get('fog_coverage', 1.0),
            'level':       info.get('level', 1),
        }
        self.episodes.append(ep)
        if len(self.episodes) > self.maxlen:
            self.episodes.pop(0)

        self.rewards.append(reward)
        self.steps.append(steps)
        self.successes.append(float(success))
        self.optimality.append(info.get('optimality', 0.0))
        self.total_steps += steps
        self.convergence.update(reward)

    def record_step(self, loss: float, td_error: float, epsilon: float):
        self.losses.append(loss)
        self.td_errors.append(td_error)
        self.epsilons.append(epsilon)

    @property
    def success_rate(self) -> float:
        if not self.successes:
            return 0.0
        return rolling_mean(self.successes, 50)

    @property
    def avg_reward(self) -> float:
        return rolling_mean(self.rewards, 50)

    @property
    def avg_steps(self) -> float:
        return rolling_mean(self.steps, 50)

    @property
    def reward_trend(self) -> float:
        return linear_trend(list(self.rewards)[-30:])

    @property
    def avg_optimality(self) -> float:
        return rolling_mean(self.optimality, 30)

    def get_chart_data(self, n: int = 200) -> Dict:
        """Return data dicts for all performance charts."""
        eps = list(self.episodes)[-n:]
        return {
            'rewards':    [e['reward'] for e in eps],
            'steps':      [e['steps'] for e in eps],
            'successes':  [int(e['success']) for e in eps],
            'optimality': [e.get('optimality', 0) for e in eps],
            'levels':     [e.get('level', 1) for e in eps],
            'losses':     list(self.losses)[-n*5:],
            'td_errors':  list(self.td_errors)[-n*5:],
            'epsilons':   list(self.epsilons)[-n*5:],
            'ema_rewards': exponential_moving_average([e['reward'] for e in eps], alpha=0.1),
        }

    def session_summary(self) -> Dict:
        elapsed = time.time() - self._session_start
        return {
            'total_episodes':  len(self.episodes),
            'total_steps':     self.total_steps,
            'session_duration':round(elapsed, 1),
            'success_rate':    round(self.success_rate, 3),
            'avg_reward':      round(self.avg_reward, 3),
            'avg_steps':       round(self.avg_steps, 1),
            'reward_trend':    round(self.reward_trend, 5),
            'convergence':     self.convergence.state,
            'avg_optimality':  round(self.avg_optimality, 3),
        }


# ============================================================
#  HEATMAP TRACKER
# ============================================================
class HeatmapTracker:
    """
    Tracks agent visits across maze cells across all episodes.
    Generates normalized heatmaps for visualization.
    """
    def __init__(self, max_h: int = 41, max_w: int = 45):
        self.max_h = max_h
        self.max_w = max_w
        self.global_visits = np.zeros((max_h, max_w), dtype=np.float32)
        self.episode_visits = np.zeros((max_h, max_w), dtype=np.float32)

    def record_step(self, r: int, c: int):
        if 0 <= r < self.max_h and 0 <= c < self.max_w:
            self.global_visits[r, c]  += 1.0
            self.episode_visits[r, c] += 1.0

    def new_episode(self):
        self.episode_visits[:] = 0.0

    def get_global_heatmap(self, h: int, w: int) -> np.ndarray:
        """Return normalized heatmap cropped to current maze size."""
        sub = self.global_visits[:h, :w].copy()
        if sub.max() > 0:
            sub /= sub.max()
        return sub

    def get_episode_heatmap(self, h: int, w: int) -> np.ndarray:
        sub = self.episode_visits[:h, :w].copy()
        if sub.max() > 0:
            sub /= sub.max()
        return sub

    def coverage(self, h: int, w: int, maze: np.ndarray = None) -> float:
        """Fraction of passable cells ever visited."""
        visited = (self.global_visits[:h, :w] > 0)
        if maze is not None:
            passable = (maze == 0)
            total = passable.sum()
            return float((visited & passable).sum()) / max(total, 1)
        return float(visited.sum()) / max(h * w, 1)


# ============================================================
#  CAPABILITY SCORE
# ============================================================
class CapabilityScore:
    """
    Composite metric (0-100) combining:
    - Success rate (40%)
    - Path efficiency vs A* (25%)
    - Exploration coverage (15%)
    - Learning convergence (10%)
    - Curriculum level reached (10%)
    """
    def __init__(self):
        self.history: deque = deque(maxlen=100)

    def compute(self, tracker: EpisodeTracker, curriculum_level: int,
                exploration_coverage: float) -> float:
        success_score  = tracker.success_rate * 40.0
        optimality_score = tracker.avg_optimality * 25.0
        exploration_score = min(exploration_coverage, 1.0) * 15.0

        conv_state = tracker.convergence.state
        conv_score = {
            'warming_up': 2.0, 'rapid_learning': 8.0, 'fine_tuning': 7.0,
            'converged': 10.0, 'plateau': 4.0, 'regressing': 0.0,
        }.get(conv_state, 0.0)

        level_score = ((curriculum_level - 1) / 9.0) * 10.0

        total = success_score + optimality_score + exploration_score + conv_score + level_score
        total = float(np.clip(total, 0.0, 100.0))
        self.history.append(total)
        return total

    @property
    def trend(self) -> str:
        if len(self.history) < 5:
            return '→'
        t = linear_trend(list(self.history)[-10:])
        if t > 0.1:   return '↑'
        elif t < -0.1: return '↓'
        return '→'


# ============================================================
#  PERFORMANCE DASHBOARD (Aggregator)
# ============================================================
class PerformanceDashboard:
    """
    The main analytics object. Aggregates all sub-systems and provides
    a unified interface for the Streamlit frontend.
    """
    def __init__(self):
        self.tracker   = EpisodeTracker()
        self.heatmap   = HeatmapTracker()
        self.capability = CapabilityScore()
        self._step_count = 0

    # ----------------------------------------------------------
    def record_step(self, agent_r: int, agent_c: int,
                    loss: float, td_error: float, epsilon: float):
        self.tracker.record_step(loss, td_error, epsilon)
        self.heatmap.record_step(agent_r, agent_c)
        self._step_count += 1

    # ----------------------------------------------------------
    def record_episode(self, reward: float, steps: int, success: bool,
                       info: Dict, curriculum_level: int,
                       h: int, w: int, maze=None):
        self.tracker.record_episode(reward, steps, success, info)
        coverage = self.heatmap.coverage(h, w, maze)
        cap = self.capability.compute(self.tracker, curriculum_level, coverage)
        self.heatmap.new_episode()
        return cap

    # ----------------------------------------------------------
    def get_live_stats(self) -> Dict:
        """Fast stat snapshot for live display."""
        t = self.tracker
        return {
            'total_episodes':  len(t.episodes),
            'success_rate':    round(t.success_rate * 100, 1),
            'avg_reward':      round(t.avg_reward, 2),
            'avg_steps':       round(t.avg_steps, 1),
            'reward_trend':    round(t.reward_trend, 5),
            'convergence':     t.convergence.state,
            'convergence_icon':t.convergence.state_icon,
            'convergence_desc':t.convergence.state_description,
            'capability':      round(self.capability.history[-1], 1) if self.capability.history else 0.0,
            'capability_trend':self.capability.trend,
        }

    # ----------------------------------------------------------
    def get_chart_data(self, n: int = 150) -> Dict:
        return self.tracker.get_chart_data(n)

    # ----------------------------------------------------------
    def get_heatmap(self, h: int, w: int, episode: bool = False) -> np.ndarray:
        if episode:
            return self.heatmap.get_episode_heatmap(h, w)
        return self.heatmap.get_global_heatmap(h, w)

    # ----------------------------------------------------------
    def get_session_report(self) -> str:
        s = self.tracker.session_summary()
        cap = self.capability.history[-1] if self.capability.history else 0.0
        lines = [
            "═══════════════════════════════════════",
            "     A.L.I.V.E. SESSION REPORT         ",
            "═══════════════════════════════════════",
            f"  Duration:        {s['session_duration']}s",
            f"  Total Episodes:  {s['total_episodes']}",
            f"  Total Steps:     {s['total_steps']}",
            f"  Success Rate:    {s['success_rate']*100:.1f}%",
            f"  Avg Reward:      {s['avg_reward']}",
            f"  Avg Steps/Ep:   {s['avg_steps']}",
            f"  Path Efficiency: {s['avg_optimality']*100:.1f}%",
            f"  Convergence:     {s['convergence']}",
            f"  Capability Score:{cap:.1f}/100",
            "═══════════════════════════════════════",
        ]
        return '\n'.join(lines)

    # ----------------------------------------------------------
    def export_json(self) -> str:
        """Export session data as JSON string."""
        data = {
            'session_summary': self.tracker.session_summary(),
            'episodes': self.tracker.episodes[-100:],
            'capability_history': list(self.capability.history),
        }
        return json.dumps(data, indent=2)
