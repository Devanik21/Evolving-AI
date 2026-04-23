import unittest
import numpy as np
import sys
import os

# Add root directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import brain

class TestBrain(unittest.TestCase):
    def test_segment_tree(self):
        # Create segment tree with capacity 8
        tree = brain.SegmentTree(8, lambda a, b: a + b, 0.0)
        self.assertEqual(len(tree.tree), 16)
        # Verify neutral element initialization
        self.assertEqual(tree.tree[1], 0.0)

    def test_curriculum_manager(self):
        cm = brain.CurriculumManager()
        self.assertEqual(cm.level, 1)
        cm.record(success=True, steps=10, max_steps=100, reward=25.0) # success and eff 0.9 -> score > 0.5
        # If we fill the window with successes it should promote
        for _ in range(7):
            cm.record(success=True, steps=10, max_steps=100, reward=25.0)
        self.assertEqual(cm.level, 2)

if __name__ == '__main__':
    unittest.main()
