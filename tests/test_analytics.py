import unittest
import numpy as np
import sys
import os
from collections import deque

# Add root directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import analytics

class TestAnalytics(unittest.TestCase):
    def test_rolling_mean(self):
        d = deque([1, 2, 3, 4, 5])
        mean = analytics.rolling_mean(d)
        self.assertEqual(mean, 3.0)

        mean_window = analytics.rolling_mean(d, window=2)
        self.assertEqual(mean_window, 4.5)

    def test_exponential_moving_average(self):
        values = [1.0, 2.0, 3.0]
        ema = analytics.exponential_moving_average(values, alpha=0.5)
        self.assertEqual(len(ema), 3)
        self.assertEqual(ema[0], 1.0)
        self.assertEqual(ema[1], 1.5)
        self.assertEqual(ema[2], 2.25)

if __name__ == '__main__':
    unittest.main()
