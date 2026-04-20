"""
memory_palace.py — Project A.L.I.V.E. NEXUS
The Memory & Persistence Engine

Memory Architecture:
  ┌───────────────────────────────────────┐
  │  Working Memory  (current episode)    │
  │  Episodic Memory (indexed events)     │
  │  Semantic Memory (world model facts)  │
  │  Persistent Store (JSON cross-session)│
  └───────────────────────────────────────┘

Inspired by:
  - Complementary Learning Systems (CLS) theory
  - Neural episodic control (Pritzel et al., 2017)
  - Human memory consolidation research
"""

import numpy as np
import json
import os
import time
import math
import hashlib
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict


# ============================================================
#  DATA CLASSES
# ============================================================
@dataclass
class Episode:
    """A full episode record with metadata."""
    episode_id:     int
    timestamp:      float
    maze_seed:      int
    maze_alg:       str
    maze_h:         int
    maze_w:         int
    curriculum_level: int
    total_steps:    int
    max_steps:      int
    total_reward:   float
    success:        bool
    efficiency:     float          # astar_optimal / steps taken (0-1)
    cells_visited:  int
    fog_used:       bool
    traps_used:     bool
    avg_td_error:   float
    epsilon_start:  float
    epsilon_end:    float
    tags:           List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'Episode':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Fact:
    """A semantic memory unit: a thing A.L.I.V.E. 'knows'."""
    key:        str
    value:      Any
    confidence: float     # 0→1 how certain this fact is
    source:     str       # 'observation', 'inference', 'user'
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0

    def strengthen(self, delta: float = 0.05):
        self.confidence = min(1.0, self.confidence + delta)
        self.access_count += 1
        self.updated_at = time.time()

    def weaken(self, delta: float = 0.02):
        self.confidence = max(0.0, self.confidence - delta)


# ============================================================
#  WORKING MEMORY
# ============================================================
class WorkingMemory:
    """
    Temporary storage for the current episode context.
    Fast access, limited capacity. Cleared at episode end.
    """
    CAPACITY = 64

    def __init__(self):
        self.slots: Dict[str, Any] = {}
        self.trace: deque = deque(maxlen=self.CAPACITY)
        self.episode_start = time.time()

    def store(self, key: str, value: Any):
        self.slots[key] = value
        self.trace.append((time.time(), key, str(value)[:80]))

    def get(self, key: str, default=None) -> Any:
        return self.slots.get(key, default)

    def record_transition(self, state, action: int, reward: float,
                          next_state, done: bool):
        self.trace.append({
            'type': 'transition', 't': time.time(),
            'action': action, 'reward': round(reward, 3), 'done': done
        })

    def flush(self) -> Dict:
        """Returns working memory contents and clears it."""
        contents = {
            'slots': dict(self.slots),
            'trace_len': len(self.trace),
            'duration': round(time.time() - self.episode_start, 2),
        }
        self.slots.clear()
        self.trace.clear()
        self.episode_start = time.time()
        return contents

    def get_reward_trace(self) -> List[float]:
        return [item.get('reward', 0.0)
                for item in self.trace
                if isinstance(item, dict) and item.get('type') == 'transition']


# ============================================================
#  EPISODIC MEMORY
# ============================================================
class EpisodicMemory:
    """
    Long-term storage for episode records with indexing and retrieval.
    Supports similarity-based retrieval (for strategy transfer).

    Memory Consolidation:
      - Recent memories: full detail
      - Old memories: compressed to summary statistics
      - Very successful episodes: flagged as 'landmarks'
    """
    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.episodes: List[Episode] = []
        self.landmarks: List[Episode] = []   # Best episodes per level
        self._level_bests: Dict[int, float] = defaultdict(lambda: -np.inf)

    def store(self, ep: Episode):
        self.episodes.append(ep)

        # Landmark detection: best reward per curriculum level
        if ep.total_reward > self._level_bests[ep.curriculum_level]:
            self._level_bests[ep.curriculum_level] = ep.total_reward
            # Update or add landmark
            self.landmarks = [l for l in self.landmarks if l.curriculum_level != ep.curriculum_level]
            self.landmarks.append(ep)
            ep.tags.append('landmark')

        # Consolidate if over capacity: drop low-reward old episodes
        if len(self.episodes) > self.capacity:
            self._consolidate()

    def _consolidate(self):
        """Remove weakest memories, keep landmarks and recent episodes."""
        n_keep_recent = 50
        recent = self.episodes[-n_keep_recent:]
        older  = self.episodes[:-n_keep_recent]
        # Keep top-scoring older episodes
        older.sort(key=lambda e: e.total_reward, reverse=True)
        keep_older = older[:self.capacity - n_keep_recent]
        self.episodes = keep_older + recent

    def retrieve_similar(self, query_ep: Dict, n: int = 5) -> List[Episode]:
        """
        Find episodes with similar context (level, success, efficiency).
        Used for meta-learning: 'what worked before in similar situations?'
        """
        if not self.episodes:
            return []

        q_level  = query_ep.get('curriculum_level', 1)
        q_succ   = query_ep.get('success', False)
        q_eff    = query_ep.get('efficiency', 0.0)

        scored = []
        for ep in self.episodes:
            level_sim = 1.0 - abs(ep.curriculum_level - q_level) / 10.0
            succ_sim  = 1.0 if ep.success == q_succ else 0.3
            eff_sim   = 1.0 - abs(ep.efficiency - q_eff)
            score = level_sim * 0.4 + succ_sim * 0.4 + eff_sim * 0.2
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:n]]

    def get_success_episodes(self, level: int = None, n: int = 10) -> List[Episode]:
        eps = [e for e in self.episodes if e.success]
        if level:
            eps = [e for e in eps if e.curriculum_level == level]
        return sorted(eps, key=lambda e: e.efficiency, reverse=True)[:n]

    def get_statistics(self) -> Dict:
        if not self.episodes:
            return {}
        rewards    = [e.total_reward for e in self.episodes]
        successes  = [e.success for e in self.episodes]
        levels     = [e.curriculum_level for e in self.episodes]
        efficiencies = [e.efficiency for e in self.episodes]

        return {
            'total_stored':    len(self.episodes),
            'landmarks':       len(self.landmarks),
            'total_successes': sum(successes),
            'success_rate':    round(np.mean(successes), 3),
            'avg_reward':      round(np.mean(rewards), 3),
            'max_reward':      round(max(rewards), 3),
            'avg_efficiency':  round(np.mean(efficiencies), 3),
            'max_level_reached': max(levels) if levels else 1,
            'level_distribution': dict(zip(*np.unique(levels, return_counts=True))) if levels else {},
        }

    def get_recent(self, n: int = 10) -> List[Dict]:
        return [ep.to_dict() for ep in self.episodes[-n:]]


# ============================================================
#  SEMANTIC MEMORY
# ============================================================
class SemanticMemory:
    """
    The agent's 'world model' — facts it has inferred or been told.
    Supports confidence-weighted knowledge updates.
    """
    def __init__(self):
        self.facts: Dict[str, Fact] = {}
        self.inference_log: deque = deque(maxlen=50)

    def assert_fact(self, key: str, value: Any, confidence: float = 0.7,
                    source: str = 'observation'):
        if key in self.facts:
            existing = self.facts[key]
            # Bayesian-style confidence update
            if existing.value == value:
                existing.strengthen(confidence * 0.1)
            else:
                # Conflicting evidence
                if confidence > existing.confidence:
                    existing.value = value
                    existing.confidence = confidence * 0.8
                else:
                    existing.weaken(0.03)
            existing.updated_at = time.time()
        else:
            self.facts[key] = Fact(key=key, value=value, confidence=confidence,
                                   source=source, updated_at=time.time())

    def query(self, key: str) -> Tuple[Any, float]:
        """Returns (value, confidence) or (None, 0.0)."""
        if key in self.facts:
            f = self.facts[key]
            f.access_count += 1
            return f.value, f.confidence
        return None, 0.0

    def infer_from_episode(self, ep: Episode):
        """Extract semantic facts from an episode."""
        inferences = []

        # Infer algorithm difficulty
        if ep.success:
            key = f'can_solve_{ep.maze_alg}_level_{ep.curriculum_level}'
            self.assert_fact(key, True, confidence=ep.efficiency)
            inferences.append(f"I can solve {ep.maze_alg} at level {ep.curriculum_level}")

        # Infer efficiency benchmarks
        if ep.efficiency > 0.8:
            key = f'best_efficiency_level_{ep.curriculum_level}'
            curr, _ = self.query(key)
            if curr is None or ep.efficiency > curr:
                self.assert_fact(key, ep.efficiency, confidence=0.9, source='inference')
                inferences.append(f"New efficiency record at level {ep.curriculum_level}: {ep.efficiency:.2f}")

        # Infer trap behavior
        if ep.traps_used and ep.success:
            self.assert_fact('can_evade_traps', True, confidence=0.6, source='observation')

        self.inference_log.extend(inferences)
        return inferences

    def get_all(self, min_confidence: float = 0.3) -> List[Dict]:
        return [
            {'key': k, 'value': f.value, 'confidence': round(f.confidence, 3),
             'source': f.source, 'accesses': f.access_count}
            for k, f in sorted(self.facts.items(), key=lambda x: x[1].confidence, reverse=True)
            if f.confidence >= min_confidence
        ]

    def get_summary(self) -> str:
        facts = self.get_all(min_confidence=0.5)
        if not facts:
            return "No established knowledge yet."
        lines = []
        for f in facts[:8]:
            lines.append(f"  [{f['confidence']:.0%}] {f['key']}: {f['value']}")
        return '\n'.join(lines)


# ============================================================
#  PERSISTENT STORE
# ============================================================
class PersistentStore:
    """
    JSON-based cross-session persistence.
    Saves/loads neural network weights, episodic memories,
    semantic facts, and training statistics.
    """
    DEFAULT_PATH = 'alive_memory.json'

    def __init__(self, path: str = None):
        self.path = path or self.DEFAULT_PATH

    def save(self, data: Dict) -> bool:
        try:
            # Convert numpy arrays and other non-serializable types
            serialized = self._serialize(data)
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(serialized, f, indent=2)
            return True
        except Exception as e:
            print(f"[PersistentStore] Save failed: {e}")
            return False

    def load(self) -> Optional[Dict]:
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[PersistentStore] Load failed: {e}")
            return None

    def _serialize(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize(v) for v in obj]
        elif hasattr(obj, 'to_dict'):
            return self._serialize(obj.to_dict())
        return obj

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def size_kb(self) -> float:
        if self.exists():
            return os.path.getsize(self.path) / 1024
        return 0.0


# ============================================================
#  MEMORY PALACE — Unified Interface
# ============================================================
class MemoryPalace:
    """
    The top-level memory system integrating:
    - Working memory (current episode)
    - Episodic memory (long-term episode store)
    - Semantic memory (world model)
    - Persistent store (cross-session JSON)
    """
    def __init__(self, save_path: str = None):
        self.working   = WorkingMemory()
        self.episodic  = EpisodicMemory(capacity=500)
        self.semantic  = SemanticMemory()
        self.store     = PersistentStore(save_path)
        self._episode_id = 0
        self._loaded   = False

    # ----------------------------------------------------------
    def start_episode(self, maze_seed: int, maze_alg: str,
                      maze_h: int, maze_w: int, level: int,
                      epsilon: float, max_steps: int):
        self.working.flush()
        self.working.store('maze_seed',    maze_seed)
        self.working.store('maze_alg',     maze_alg)
        self.working.store('maze_h',       maze_h)
        self.working.store('maze_w',       maze_w)
        self.working.store('level',        level)
        self.working.store('epsilon_start', epsilon)
        self.working.store('max_steps',    max_steps)
        self.working.store('start_time',   time.time())

    # ----------------------------------------------------------
    def record_transition(self, state, action: int, reward: float,
                          next_state, done: bool):
        self.working.record_transition(state, action, reward, next_state, done)

    # ----------------------------------------------------------
    def end_episode(self, total_reward: float, steps: int, success: bool,
                    cells_visited: int, astar_optimal: int,
                    fog: bool, traps: bool, td_error: float, epsilon: float,
                    infer_facts: bool = True) -> Episode:
        self._episode_id += 1

        efficiency = min(1.0, astar_optimal / max(steps, 1)) if success and astar_optimal > 0 else 0.0

        ep = Episode(
            episode_id       = self._episode_id,
            timestamp        = time.time(),
            maze_seed        = self.working.get('maze_seed', 0),
            maze_alg         = self.working.get('maze_alg', 'unknown'),
            maze_h           = self.working.get('maze_h', 0),
            maze_w           = self.working.get('maze_w', 0),
            curriculum_level = self.working.get('level', 1),
            total_steps      = steps,
            max_steps        = self.working.get('max_steps', steps),
            total_reward     = round(total_reward, 3),
            success          = success,
            efficiency       = round(efficiency, 4),
            cells_visited    = cells_visited,
            fog_used         = fog,
            traps_used       = traps,
            avg_td_error     = round(td_error, 5),
            epsilon_start    = round(self.working.get('epsilon_start', 1.0), 4),
            epsilon_end      = round(epsilon, 4),
            tags             = ['success'] if success else ['timeout' if steps >= self.working.get('max_steps', steps) else 'failed'],
        )

        self.episodic.store(ep)

        if infer_facts:
            inferences = self.semantic.infer_from_episode(ep)

        return ep

    # ----------------------------------------------------------
    def save_all(self, brain_weights: Dict, analytics_data: Dict,
                 soul_status: Dict) -> bool:
        payload = {
            'version':      '2.0',
            'saved_at':     time.time(),
            'brain_weights': brain_weights,
            'episodic_stats': self.episodic.get_statistics(),
            'episodic_recent': self.episodic.get_recent(20),
            'semantic_facts': self.semantic.get_all(),
            'analytics': analytics_data,
            'soul': {
                'relationship': soul_status.get('relationship', 50),
                'turns':        soul_status.get('turns', 0),
                'personality':  {
                    'O': soul_status.get('O', 0.5),
                    'C': soul_status.get('C', 0.5),
                    'E': soul_status.get('E', 0.5),
                    'A': soul_status.get('A', 0.5),
                    'N': soul_status.get('N', 0.5),
                }
            },
            'episode_count': self._episode_id,
        }
        return self.store.save(payload)

    # ----------------------------------------------------------
    def load_all(self) -> Optional[Dict]:
        data = self.store.load()
        if data:
            self._loaded = True
            # Restore semantic facts
            for f in data.get('semantic_facts', []):
                self.semantic.assert_fact(
                    f['key'], f['value'], f['confidence'], f.get('source', 'loaded')
                )
            # Restore episode count
            self._episode_id = data.get('episode_count', 0)
        return data

    # ----------------------------------------------------------
    def get_insights(self) -> List[str]:
        """Generate human-readable insights from memory."""
        insights = []
        stats = self.episodic.get_statistics()

        if stats:
            insights.append(f"🧠 I remember {stats['total_stored']} episodes across this lifetime.")
            if stats['success_rate'] > 0:
                insights.append(f"🏆 Success rate: {stats['success_rate']*100:.1f}% across all levels.")
            insights.append(f"⭐ Best level reached: {stats['max_level_reached']}")
            if stats['avg_efficiency'] > 0:
                insights.append(f"⚡ Average path efficiency: {stats['avg_efficiency']*100:.1f}% vs A* optimal.")

        landmarks = self.episodic.landmarks
        if landmarks:
            best = max(landmarks, key=lambda e: e.total_reward)
            insights.append(f"🌟 Landmark episode: Level {best.curriculum_level}, "
                           f"reward={best.total_reward:.1f}, {best.maze_alg} maze.")

        facts = self.semantic.get_all(min_confidence=0.6)
        if facts:
            insights.append(f"📖 Established {len(facts)} high-confidence facts about the world.")

        inferences = list(self.semantic.inference_log)[-3:]
        for inf in inferences:
            insights.append(f"💡 Inferred: {inf}")

        return insights

    # ----------------------------------------------------------
    def get_full_status(self) -> Dict:
        return {
            'episodic_stats':   self.episodic.get_statistics(),
            'episodic_recent':  self.episodic.get_recent(5),
            'landmark_episodes': [ep.to_dict() for ep in self.episodic.landmarks[:5]],
            'semantic_facts':   self.semantic.get_all(min_confidence=0.3),
            'semantic_summary': self.semantic.get_summary(),
            'insights':         self.get_insights(),
            'save_path':        self.store.path,
            'save_size_kb':     round(self.store.size_kb(), 2),
            'total_episodes':   self._episode_id,
            'loaded_from_disk': self._loaded,
        }
