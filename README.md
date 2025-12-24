# Reinforcement Learning Framework Board Games

This repository contains a comprehensive framework for developing, training, and evaluating autonomous agents in  zero-sum strategy games (currently Connect Four and Tic-Tac-Toe). The system uses **Tabular Q-Learning** (a model-free reinforcement learning algorithm) to learn optimal policies through temporal-difference learning and adversarial self-play.

## Project Overview

The framework provides an architecture for modeling games as **Markov Decision Processes (MDPs)**. By separating the game physics (Board Logic) from the agent cognition (Policy Logic), the system allows for rapid experimentation with different reward structures and state-space optimisation techniques. The primary focus of the implementation is to achieve policy convergence in environments with >million state-action pairs such as Connect Four (Tic-Tac-Toe converges around 16-8k).

## Technical Architecture

### Agent Cognition and Policy
The agents use an **$\epsilon$-greedy exploration strategy** to balance the discovery of new strategies with the exploitation of known optimal moves. The learning process is governed by the Bellman equation which updates the state-action values ($Q$-values) based on observed rewards and the estimated value of subsequent states.

### State-Space Optimisation (Symmetry Reduction)
To address the "curse of dimensionality," the framework implements **State Canonicalization**. By applying rotations and reflections (D4 Symmetry Group for Tic-Tac-Toe and Mirror Symmetry for Connect Four) the agent collapses redundant board configurations into a single canonical representation. This reduces the total state-space volume by up to 8x significantly accelerating the convergence of the Q-table.

### Multi-Factor Reward Shaping
Beyond sparse binary rewards (win/loss), the agents utilize sophisticated reward shaping to internalize intermediate strategic goals:
*   **Adversarial Heuristics:** Positive reinforcement for creating winning threats (forks) and blocking opponent sequences.
*   **Positional Advantage:** Weighting of central occupancy and strategic sub-structures (e.g., vertical stacks in Connect Four).
*   **Penalty Mechanisms:** Negative reinforcement for "suicide moves" that grant immediate winning opportunities to the opponent.

## Evaluation and Benchmarking

### Minimax Baseline Integration
To provide a rigorous performance metric, the framework includes an **Optimal AI** based on the **Minimax algorithm with Alpha-Beta Pruning**. This serves as a theoretical upper bound for performance. RL agents are periodically benchmarked against this baseline to calculate their "optimality gap" and verify the stability of the learned policy.

### Automated Training Analytics
The system includes a logging tool that tracks win/loss/draw ratios over thousands of episodes, providing data-driven insights into agent progress and the impact of hyperparameter tuning (Learning Rate $\alpha$, Discount Factor $\gamma$).

## TODO

*   **Bitboard State Representation:** Transitioning from matrix-based board tracking to 64-bit integer bitboards to utilize bitwise operations for instantaneous win-checking and move validation.
*   **Function Approximation (DQN):** Implementing Deep Q-Networks to replace the Tabular memory, enabling the framework to generalize across the significantly larger state spaces of games like Othello.
*   **Asynchronous Parallel Self-Play:** Utilising Python’s `multiprocessing` library to execute concurrent training environments, increasing the throughput of self-play data.

## Usage Specifications

The toolchain is executed via a Command Line Interface (CLI) supporting various operational modes.

### Training Mode
Initiate a training session between two RL agents to populate the Q-table:
```bash
python main.py --game connect4 --mode train --episodes 10000
```

### Benchmarking Mode
Evaluate a trained model against the Optimal Minimax baseline:
```bash
python main.py --game tictactoe --mode benchmark --model tictactoe_model.pkl
```

### Interactive Play
Execute a match between a human player and a trained agent:
```bash
python main.py --game connect4 --mode play --model c4_v1.pkl
```
