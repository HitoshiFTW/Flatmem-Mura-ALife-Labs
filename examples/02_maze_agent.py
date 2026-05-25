"""
Example 2: Maze RL agent with flat memory as Q-table.

Agent in a 10x10 grid learns expected rewards per (state, action).
No neural net. No Q-table dict. Flat substrate holds it all.

Run: python examples/02_maze_agent.py
"""

import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from flatmem import MultiRoleMemory, scalar_phasor, position_phasor
from flatmem.encoders import decode_scalar

ACTIONS = ['up', 'down', 'left', 'right']

mem = MultiRoleMemory(
    d=512, M=4096, k=32, seed=42, M_rel=2048,
    roles=('cooccur', 'next') + tuple(f'act_{a}' for a in ACTIONS),
)

def state_slot(x, y): return f's_{x}_{y}'

def update_q(x, y, action, reward, n_writes=5):
    """Write expected reward as superposed phasor (running average via accumulation)."""
    slot = state_slot(x, y)
    hv = scalar_phasor(float(reward), d=512)
    for _ in range(n_writes):
        mem.write_relation(slot, f'act_{action}', hv)

def get_q(x, y, action):
    slot = state_slot(x, y)
    hv = mem.recall(slot, f'act_{action}')
    return decode_scalar(hv)

def choose_action(x, y, epsilon=0.1, rng=None):
    if rng is None: rng = random.Random()
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)
    return max(ACTIONS, key=lambda a: get_q(x, y, a))

# Train: simulate 200 episodes where the goal is at (9, 9).
# Each step: take action, reward = -dist_to_goal + 10 if goal reached.
rng = random.Random(0)
for ep in range(200):
    x, y = rng.randint(0, 9), rng.randint(0, 9)
    for _ in range(20):
        action = choose_action(x, y, epsilon=0.3, rng=rng)
        nx, ny = x, y
        if action == 'up'    and y < 9: ny += 1
        elif action == 'down'  and y > 0: ny -= 1
        elif action == 'left'  and x > 0: nx -= 1
        elif action == 'right' and x < 9: nx += 1
        dist_before = abs(x - 9) + abs(y - 9)
        dist_after  = abs(nx - 9) + abs(ny - 9)
        reward = (dist_before - dist_after) * 0.5 + (10 if (nx, ny) == (9, 9) else 0)
        update_q(x, y, action, reward, n_writes=3)
        x, y = nx, ny
        if (x, y) == (9, 9): break

# Test learned policy from a few states
print('Learned Q-values + chosen action:')
for state in [(0, 0), (5, 5), (8, 8), (9, 8)]:
    qs = {a: get_q(*state, a) for a in ACTIONS}
    best = max(qs, key=qs.get)
    print(f'  state {state}: {", ".join(f"{a}={v:+.2f}" for a, v in qs.items())}  -> {best}')

print(f'\nSubstrate: {mem.substrate_bytes() / 1_048_576:.0f} MB FIXED')
