import unittest
import numpy as np
import sys
import os

# Add root directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import memory_palace

class TestMemoryPalace(unittest.TestCase):
    def test_episode_dataclass(self):
        ep = memory_palace.Episode(
            episode_id=1,
            timestamp=12345.6,
            maze_seed=42,
            maze_alg="backtracker",
            maze_h=11,
            maze_w=11,
            curriculum_level=1,
            total_steps=100,
            max_steps=200,
            total_reward=25.0,
            success=True,
            efficiency=0.9,
            cells_visited=50,
            fog_used=False,
            traps_used=False,
            avg_td_error=1.5,
            epsilon_start=0.7,
            epsilon_end=0.05,
            tags=["test"]
        )
        self.assertEqual(ep.episode_id, 1)
        self.assertEqual(ep.success, True)
        self.assertEqual(ep.efficiency, 0.9)

if __name__ == '__main__':
    unittest.main()
