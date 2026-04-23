import unittest
import numpy as np
import sys
import os

# Add root directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import soul

class TestSoul(unittest.TestCase):
    def test_emotion_point(self):
        ep = soul.EmotionPoint(0.5, 0.5)
        self.assertEqual(ep.valence, 0.5)
        self.assertEqual(ep.arousal, 0.5)

        # Test bounds
        ep_bounded = soul.EmotionPoint(1.5, -2.0)
        self.assertEqual(ep_bounded.valence, 1.0)
        self.assertEqual(ep_bounded.arousal, -1.0)

    def test_emotion_blend(self):
        ep1 = soul.EmotionPoint(0.0, 0.0)
        ep2 = soul.EmotionPoint(1.0, 1.0)
        ep3 = ep1.blend(ep2, 0.3)
        self.assertAlmostEqual(ep3.valence, 0.3)
        self.assertAlmostEqual(ep3.arousal, 0.3)

if __name__ == '__main__':
    unittest.main()
