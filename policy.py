from main import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def extract_policy(V):

    policy = {}

    for i, state in enumerate(states):

        if is_terminal(*state):
            policy[state] = "T"
            continue

        action_values = []

        for action in ACTIONS:
            total = 0
            for next_state, prob, reward in get_transitions(state, action):
                j = state_to_idx[next_state]
                total += prob * (reward + config.GAMMA * V[j])
            action_values.append(total)

        best_action = ACTIONS[np.argmax(action_values)]
        policy[state] = best_action

    return policy

def print_policy(policy, water=0):

    grid = np.empty((config.GRID_SIZE, config.GRID_SIZE), dtype=str)

    for x in range(config.GRID_SIZE):
        for y in range(config.GRID_SIZE):

            if (x,y) in config.BOULDERS:
                grid[x,y] = "B"
            elif (x,y) == config.FIRE:
                grid[x,y] = "F"
            elif (x,y) == config.LAKE:
                grid[x,y] = "L"
            else:
                grid[x,y] = policy[(x,y,water)]

    print(grid)


ACTIONS = ["N", "S", "E", "W", "H"]

DIR = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "W": (0, -1)
}

# Create all states
states = []
for x in range(config.GRID_SIZE):
    for y in range(config.GRID_SIZE):
        for w in [0, 1]:
            states.append((x, y, w))

state_to_idx = {s: i for i, s in enumerate(states)}
idx_to_state = {i: s for s, i in state_to_idx.items()}

n_states = len(states)
n_actions = len(ACTIONS)

V = np.load("results/values.npy")
policy = extract_policy(V)
print("Optimal policy in empty phase: ")
print_policy(policy, water=0)
print()
print("---------------------------------")
print()
print("Optimal policy in filled phase: ")
print_policy(policy, water=1)