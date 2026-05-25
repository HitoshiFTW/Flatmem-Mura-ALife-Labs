"""
Example 3: Boid swarm with experiential steering memory.

Each boid encodes its local visual cone (5 nearest neighbor velocities)
into a phasor HV; recalls past optimal steering vectors from those states.
Demonstrates flat memory replacing classic boid rules with experience-based steering.

Run: python examples/03_boids_memory.py
"""

import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from flatmem import MultiRoleMemory, scalar_phasor
from flatmem.encoders import random_projection_encoder, decode_scalar

mem = MultiRoleMemory(
    d=512, M=4096, k=32, seed=42, M_rel=2048,
    roles=('cooccur', 'next', 'cohere', 'align', 'separate'),
)

def visual_cone(neighbors_velocity):
    """Encode flattened neighbor velocity vector as HV."""
    v = np.asarray(neighbors_velocity, dtype=np.float32).ravel()
    return random_projection_encoder(v, d=512, seed=11)

# Train: 100 simulated frames, write learned steering forces for various visual states.
rng = random.Random(0)
for _ in range(100):
    neighbors = [(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(5)]
    state = visual_cone(neighbors)
    # Heuristic steering forces (in a real sim, these would be optimal-per-state)
    cohere   = sum(n[0] for n in neighbors) / 5
    align    = sum(n[1] for n in neighbors) / 5
    separate = -cohere
    # Write each as scalar-encoded HV under its role
    mem.sdm_rel.write(mem._bind(state, mem.roles['cohere']),
                      scalar_phasor(cohere, d=512), word=f'state_{_}|cohere')
    mem.sdm_rel.write(mem._bind(state, mem.roles['align']),
                      scalar_phasor(align, d=512), word=f'state_{_}|align')
    mem.sdm_rel.write(mem._bind(state, mem.roles['separate']),
                      scalar_phasor(separate, d=512), word=f'state_{_}|separate')

# Recall steering for a novel-but-similar state
test_neighbors = [(0.5, -0.3), (0.4, -0.2), (0.6, -0.1), (0.55, -0.25), (0.45, -0.2)]
state = visual_cone(test_neighbors)
for role in ('cohere', 'align', 'separate'):
    out = mem.sdm_rel.read(mem._bind(state, mem.roles[role]))
    val = decode_scalar(out)
    print(f'  recall {role:10s}: {val:+.3f}')

print(f'\nSubstrate: {mem.substrate_bytes() / 1_048_576:.0f} MB FIXED')
print('Note: this demo shows the substrate hosts boid steering history;')
print('  in a real sim, optimal steering vectors would replace the heuristic above.')
