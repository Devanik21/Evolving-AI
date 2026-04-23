import os
import json
import itertools
import subprocess

def run_hyperparameter_search():
    """
    Automates grid search over defined hyperparameters.
    """
    learning_rates = [0.001, 0.0001, 0.3]
    gammas = [0.99, 0.95]

    base_config = {
        "action_size": 4,
        "batch_size": 64,
        "state_size": 52
    }

    print("Starting hyperparameter search...")
    for lr, gamma in itertools.product(learning_rates, gammas):
        config = base_config.copy()
        config["lr"] = lr
        config["gamma"] = gamma

        config_path = f"config/search_lr_{lr}_gamma_{gamma}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        print(f"Generated config: {config_path}")
        # Normally would execute training script here
        # subprocess.run(["python3", "main_training_script.py", "--config", config_path])

if __name__ == "__main__":
    run_hyperparameter_search()
