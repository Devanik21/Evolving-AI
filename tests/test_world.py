import unittest
import numpy as np
import sys
import os

# Add root directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import world

class TestWorld(unittest.TestCase):
    def test_maze_generator(self):
        # Test default backtracker
        grid = world.MazeGenerator.generate(11, 11, algorithm="backtracker")
        self.assertEqual(grid.shape, (11, 11))
        self.assertTrue(np.any(grid == world.PATH))
        self.assertTrue(np.any(grid == world.WALL))

    def test_environment_init(self):
        config = {'maze_h': 11, 'maze_w': 11}
        env = world.MazeEnvironment(config=config)
        self.assertEqual(env.maze_h, 11)
        self.assertEqual(env.maze_w, 11)
        state = env.reset()
        self.assertIsNotNone(state)
        # Check initial pos
        self.assertTrue(0 <= env.agent_r < 11)

if __name__ == '__main__':
    unittest.main()
