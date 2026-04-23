# Empirical Results Log

This document tracks the experimental results across different versions of the RIENFoRZe architecture.

## Version I (17D State, Baseline)
* **Status**: Complete
* **Notes**: Established baseline. Encountered aggressive epsilon decay due to per-step application instead of per-episode.
* **Key Metric**: Path efficiency plateaus at ~0.6.

## Version II (52D State, Huber Loss, Dyna-Q K=5)
* **Status**: Complete
* **Notes**: Added Pheromone Cross and Dyna-Q. Fixed epsilon decay bug. Convergence was significantly faster on Level 5+ mazes.
* **Key Metric**: Success rate improved by 40% on Wilson Algorithm generated mazes compared to v-I.

## Version III (64D State, Accelerated Dyna-Q K=25)
* **Status**: Complete
* **Notes**: Reached maximal sensor configuration with Raycast and Scent Gradients. The loop incorporating Curiosity into state yielded emergent meta-learning behavior.

## Version IV (52D Tabular Dyna-Q K=20)
* **Status**: Ongoing
* **Notes**: Paradigm shift away from Neural function approximation.
* **Hypothesis**: For finite deterministic MDPs, exact Q-updates will bypass the "Deadly Triad" instability seen in v-III. Memory efficiency is expected to improve.
