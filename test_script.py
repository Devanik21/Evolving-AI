import unittest
import sys
import os

# Add tests directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'tests')))

from test_world import TestWorld
from test_brain import TestBrain
from test_soul import TestSoul
from test_memory_palace import TestMemoryPalace
from test_analytics import TestAnalytics

if __name__ == '__main__':
    # Run all tests
    unittest.main()
