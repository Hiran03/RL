# Grid parameters
GRID_SIZE = 5

LAKE = (0, 0)
FIRE = (4, 4)

SMOKE = {(1, 2), (3, 2)}
BOULDERS = {(2, 4), (3, 4)}

# Movement probabilities
NORMAL_MOVE = {
    "intended": 0.7,
    "perp": 0.1,
    "stay": 0.1
}

SMOKE_MOVE = {
    "intended": 0.4,
    "perp": 0.1,
    "stay": 0.4
}

STEP_COST = -1
SMOKE_PENALTY = -10
CRASH_COST = -100
SUCCESS_REWARD = 100

GAMMA = 0.95