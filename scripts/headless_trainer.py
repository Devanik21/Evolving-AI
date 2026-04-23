import argparse
import sys

def simulate_training(episodes: int):
    """
    Dummy trainer meant to be invoked from automation scripts
    """
    print(f"Initializing model for {episodes} episodes...")
    for e in range(episodes):
        if e % 10 == 0:
            print(f"Episode {e} completed. Loss: 0.05")
    print("Training Complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    simulate_training(args.episodes)
