import numpy as np
import config
import pandas as pd

ACTIONS = ["N", "S", "E", "W", "H"]

DIR = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "W": (0, -1)
}
def in_grid(x, y):
    return 0 <= x < config.GRID_SIZE and 0 <= y < config.GRID_SIZE

def move(x, y, action):
    dx, dy = DIR[action]
    nx, ny = x + dx, y + dy
    if not in_grid(nx, ny):
        return x, y
    return nx, ny

def get_perpendicular(action):
    if action in ["N", "S"]:
        return ["E", "W"]
    return ["N", "S"]
def reward(x, y, nx, ny, water):

    r = config.STEP_COST

    if (nx, ny) in config.SMOKE:
        r += config.SMOKE_PENALTY

    if (nx, ny) in config.BOULDERS:
        r += config.CRASH_COST

    if (nx, ny) == config.FIRE and water == 1:
        r += config.SUCCESS_REWARD

    return r
def is_terminal(x, y, w):
    if (x, y) in config.BOULDERS:
        return True
    if (x, y) == config.FIRE and w == 1:
        return True
    return False
def get_transitions(state, action):
    x, y, w = state

    if is_terminal(x, y, w):
        return [(state, 1.0, 0)]

    if action == "H":
        return [((x, y, w), 1.0, config.STEP_COST)]

    probs = config.SMOKE_MOVE if (x, y) in config.SMOKE else config.NORMAL_MOVE

    outcomes = []

    # Intended
    nx, ny = move(x, y, action)
    outcomes.append((nx, ny, probs["intended"]))

    # Perpendicular
    for p in get_perpendicular(action):
        px, py = move(x, y, p)
        outcomes.append((px, py, probs["perp"]))

    # Stay
    outcomes.append((x, y, probs["stay"]))

    transitions = []

    for nx, ny, prob in outcomes:

        new_w = w
        if (nx, ny) == config.LAKE:
            new_w = 1

        reward = config.STEP_COST

        if (nx, ny) in config.SMOKE:
            reward += config.SMOKE_PENALTY

        if (nx, ny) in config.BOULDERS:
            reward += config.CRASH_COST

        if (nx, ny) == config.FIRE and new_w == 1:
            reward += config.SUCCESS_REWARD

        transitions.append(((nx, ny, new_w), prob, reward))

    return transitions
# Create all states
states = []
for x in range(config.GRID_SIZE):
    for y in range(config.GRID_SIZE):
        for w in [0, 1]:
            states.append((x, y, w))



def visualize_subset(state, action):

    trans = get_transitions(state, action)

    cells = [(i, j) for i in range(2, 5) for j in range(2, 5)]

    # Proper independent matrix
    matrix = np.empty((3, 3), dtype=object)

    # Initialize with (0 prob, 0 reward)
    for i in range(3):
        for j in range(3):
            matrix[i, j] = (0.0, 0.0)

    # Accumulate probabilities and expected rewards
    for (nx, ny, _), prob, r in trans:
        if (nx, ny) in cells:

            i = nx - 2
            j = ny - 2
            matrix[i, j] = (prob,r)

    df = pd.DataFrame(
        matrix,
        index=[2, 3, 4],
        columns=[2, 3, 4]
    )

    print(f"\nTransition-Reward tuples for action {action} from (3,3):")
    print(df)

    return df

if __name__ == "__main__":
    
    for a in ACTIONS : 
        print("Action:", a)
        visualize_subset((3, 3, 0), a)
        print("------------------------------------------------------")
        print()