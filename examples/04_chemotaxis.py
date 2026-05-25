"""
Example 4: Chemotaxis organism — gradient-history memory.

An organism observes a stream of chemical concentration values along
its trajectory. Flat memory accumulates them as superposed phasors.
Recall returns the mean concentration — a noise-tolerant running average
in fixed bytes.

Run: python examples/04_chemotaxis.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from flatmem import MultiRoleMemory, scalar_phasor
from flatmem.encoders import decode_scalar

mem = MultiRoleMemory(d=512, M=4096, k=32, seed=42, M_rel=2048,
                      roles=('cooccur', 'next', 'concentration'))

# Simulate organism walking through a gradient: 200 noisy observations
# with mean concentration drifting from 2.0 to 5.0
rng = np.random.default_rng(0)
for t in range(200):
    true_c = 2.0 + 3.0 * (t / 200.0)
    noisy  = true_c + rng.normal(0, 0.3)
    mem.write_relation('agent_A', 'concentration', scalar_phasor(noisy, d=512))

recalled_hv = mem.recall('agent_A', 'concentration')
avg = decode_scalar(recalled_hv)
true_mean = (2.0 + 5.0) / 2
print(f'  recalled mean concentration: {avg:.3f}  (true mean: {true_mean:.3f})')
print(f'  error: {abs(avg - true_mean):.3f}')
print(f'\nSubstrate: {mem.substrate_bytes() / 1_048_576:.0f} MB FIXED for any history length')
print('Try increasing the loop to 10000 -- substrate stays the same.')
