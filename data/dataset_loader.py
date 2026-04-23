import json
import os
from typing import List, Dict, Any

class ReplayBufferExporter:
    """
    Utility for serializing Prioritized Experience Replay buffer contents
    to disk for offline RL or imitation learning later.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def export(self, buffer_data: List[Dict[str, Any]], filename: str):
        path = os.path.join(self.data_dir, filename)
        with open(path, 'w') as f:
            json.dump(buffer_data, f)
        print(f"Exported {len(buffer_data)} transitions to {path}")

    def load(self, filename: str) -> List[Dict[str, Any]]:
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            return []
        with open(path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} transitions from {path}")
        return data
