from main import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os

def value_iteration(theta=1e-6, save_dir="results", save_every=False):

    os.makedirs(save_dir, exist_ok=True)

    V = np.zeros(n_states)

    history = []
    value_snapshots = []

    iteration = 0

    while True:
        delta = 0
        new_V = np.copy(V)

        for i, state in enumerate(states):

            x, y, w = state

            if is_terminal(x, y, w):
                continue

            action_values = []

            for action in ACTIONS:
                total = 0
                for next_state, prob, reward in get_transitions(state, action):
                    j = state_to_idx[next_state]
                    total += prob * (reward + GAMMA * V[j])

                action_values.append(total)

            best_value = max(action_values)
            new_V[i] = best_value
            delta = max(delta, abs(V[i] - best_value))

        V = new_V
        history.append(delta)

        if save_every:
            value_snapshots.append(V.copy())

        iteration += 1

        if delta < theta:
            break

    print(f"Converged in {iteration} iterations")

    # ==============================
    # Save final values
    # ==============================

    np.save(f"{save_dir}/values.npy", V)

    df = pd.DataFrame({
        "state": states,
        "value": V
    })
    df.to_csv(f"{save_dir}/values.csv", index=False)

    # ==============================
    # Save convergence plot
    # ==============================

    plt.figure(figsize=(6,6))
    plt.plot(history)
    plt.tick_params(axis='both', labelsize=18)
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Bellman Residual (delta)", fontsize=18)
    
    
    plt.savefig(f"{save_dir}/convergence_4d.png")
    plt.close()

    # Optionally save all snapshots
    if save_every:
        np.save(f"{save_dir}/value_snapshots.npy",
                np.array(value_snapshots))

    return V, history

    
def save_value_heatmaps(V, save_dir="results"):

    os.makedirs(save_dir, exist_ok=True)

    

    for water in [0, 1]:

        grid = np.zeros((GRID_SIZE, GRID_SIZE))

        # row=y, col=x
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                state = (x, y, water)
                idx = state_to_idx[state]
                grid[y, x] = V[idx]

        fig, ax = plt.subplots(figsize=(6,6))

        

        # Heatmap
        im = ax.imshow(grid, cmap="YlOrRd", alpha=0.9)

       

        # -------------------
        # Add PNG icons
        # -------------------
        add_icon(ax, "assets/lake.png", *LAKE)
        add_icon(ax, "assets/fire.png", *FIRE)

        for pos in SMOKE:
            add_icon(ax, "assets/smoke.png", *pos)

        for pos in BOULDERS:
            add_icon(ax, "assets/rock.png", *pos)

        # Formatting
        ax.set_xticks(np.arange(GRID_SIZE))
        ax.set_yticks(np.arange(GRID_SIZE))
        ax.tick_params(axis='both', labelsize=18)

        ax.set_xlabel("x", fontsize=18)
        ax.set_ylabel("y", fontsize=18)
        ax.invert_yaxis()
        plt.colorbar(im, label="Value")
        plt.tight_layout()

        plt.savefig(f"{save_dir}/value_heatmap_water_{water}_4d.png", dpi=300)
        plt.close()
        

# Create all states
states = []
for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        for w in [0, 1]:
            states.append((x, y, w))

state_to_idx = {s: i for i, s in enumerate(states)}
idx_to_state = {i: s for s, i in state_to_idx.items()}

n_states = len(states)
n_actions = len(ACTIONS)

V, history = value_iteration(theta=1e-16)
save_value_heatmaps(V)