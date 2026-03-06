# Installation

Install the required dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# Gridworld and Value Iteration

This project implements a **Gridworld Markov Decision Process (MDP)**
for a drone navigating a forest fire scenario and solves it using
**Value Iteration**.

------------------------------------------------------------------------

## Configuration

Modify the environment settings in:

    Gridworld/config.yaml

This file contains parameters such as:

-   Grid size\
-   Lake and fire locations\
-   Smoke and boulder positions\
-   Movement probabilities\
-   Rewards and penalties\
-   Discount factor

------------------------------------------------------------------------

## Running the Code

### 1. Inspect Transitions

Run:

``` bash
python3 ./Gridworld/main.py
```

This prints the **transition and reward matrices** for the state
starting at:

    (3, 3)

------------------------------------------------------------------------

### 2. Run Value Iteration

Run:

``` bash
python3 ./Gridworld/optimisation.py
```

This performs **Value Iteration** and generates:

-   **Convergence graph** (value updates vs iterations)
-   **Final value function heatmaps**
    -   with water
    -   without water
-   **Saved value outputs**
    -   `.csv`
    -   `.npy`

All outputs are saved in the `Gridworld/results/` directory.

------------------------------------------------------------------------

### 3. Visualize the Optimal Policy

Run:

``` bash
python3 ./Gridworld/policy.py
```

Outputs:

-   Policy visualization for:
    -   **without water**
    -   **with water**

The policies are rendered as **grid maps with action icons**.

------------------------------------------------------------------------

## Assets

The `./Gridworld/assets/` folder contains **PNG cliparts** used for visualization:

-   Movement directions (N, S, E, W, Hover)
-   Lake
-   Fire
-   Smoke
-   Boulders

These icons are used to render the **policy and environment visually on
the grid**.

------------------------------------------------------------------------

# TD-based Control in Acrobot

This project implements a **Gridworld Markov Decision Process (MDP)**
for a drone navigating a forest fire scenario and solves it using
**Value Iteration**.

------------------------------------------------------------------------

## Configuration

Modify the environment settings in:

    Acrobot/config.py

This file contains parameters such as:

-   Number of bins\
-   Gamma\
-   Number of episodes to run\
-   Number of seeds to use\
-   Best hyperparameters\
-   Window to smooth curves\
-   Bin configuration

------------------------------------------------------------------------

## Running the Code

Run:

``` bash
python3 ./Acrobot/main.py
```

The code required to solve and get the results for the assignment run
sequentially while producing plots and outputs in the terminal.

- Finding the optimal episodes count
- Hyperparameter tuning
- Plot of performance between SARSA vs Q-learning
- Comparison between online and offline performance
- Finding effect of binning

------------------------------------------------------------------------

## File Descriptions

- **main.py** – Entry point that runs hyperparameter search, experiments, and generates plots.
- **config.py** – Stores global configuration such as alpha/epsilon grids and experiment settings.
- **experiments.py** – Runs training experiments for SARSA and Q-Learning with selected parameters.
- **evaluation.py** – Evaluates trained policies and computes performance metrics.
- **hyperparameter_search.py** – Performs grid search over α and ε to find the best configurations.

### algorithms/
- **sarsa.py** – Implementation of the SARSA reinforcement learning algorithm.
- **q_learning.py** – Implementation of the Q-Learning reinforcement learning algorithm.

### utils/
- **discretization.py** – Converts continuous environment states into discrete bins.
- **exploration.py** – Implements ε-greedy exploration strategy.
- **plotting.py** – Generates plots such as reward curves and hyperparameter heatmaps.

------------------------------------------------------------------------

## Plots and images

The `./Acrobot/plots/` folder contains **PNG images** used for visualization

------------------------------------------------------------------------