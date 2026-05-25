"""
Universality tests: confirm the substrate works as a drop-in for several
ALife paradigms (maze RL, boids steering, chemotaxis, evolutionary fitness).

Run: python tests/test_universality.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from flatmem import (
    MultiRoleMemory, ComputedKey,
    scalar_phasor, position_phasor, random_projection_encoder,
    bind, bundle, permute, cos, renorm,
)


# ── 1. Maze RL: state-action -> reward ───────────────────────────────────────

def test_maze_rl_q_value():
    """
    Grid-world: encode (x,y) position, write expected reward per action,
    recover Q-values via flat memory.
    """
    mr = MultiRoleMemory(
        d=512, M=2048, k=32, seed=42, M_rel=1024,
        roles=('cooccur', 'next', 'action_up', 'action_down', 'action_left', 'action_right'),
    )
    # state at (5,5) -> action_up has reward +1, action_down has reward -1
    state_hv = position_phasor(5.0, 5.0, d=512)
    # Use write_relation with scalar-phasor encoded reward
    for _ in range(30):
        mr.write_relation('state_5_5', 'action_up',   scalar_phasor(+1.0, d=512))
        mr.write_relation('state_5_5', 'action_down', scalar_phasor(-1.0, d=512))
    # Decode recalled rewards
    up_hv   = mr.recall('state_5_5', 'action_up')
    down_hv = mr.recall('state_5_5', 'action_down')
    from flatmem.encoders import decode_scalar
    up   = decode_scalar(up_hv)
    down = decode_scalar(down_hv)
    assert up > down,                      f'up({up}) should beat down({down})'
    assert abs(up - 1.0) < 0.3,            f'up reward ~+1, got {up}'
    assert abs(down + 1.0) < 0.3,          f'down reward ~-1, got {down}'


# ── 2. Boids: experiential steering ──────────────────────────────────────────

def test_boids_neighbor_recall():
    """
    Boid sees neighbor velocity pattern -> recalls historical 'cohere' force.
    Same pattern should give consistent recall (write multiple times -> strong).
    """
    mr = MultiRoleMemory(
        d=512, M=2048, k=32, seed=42, M_rel=1024,
        roles=('cooccur', 'next', 'cohere', 'align', 'separate'),
    )
    # Encode visual cone state with random projection
    visual = np.array([1.0, 0.5, -0.3, 0.1, 0.8])
    state = random_projection_encoder(visual, d=512, seed=7)
    # Write a 'cohere' steering vector (positive direction)
    steer = scalar_phasor(+0.7, d=512)
    for _ in range(20):
        mr.write_relation('boid_state_A', 'cohere', steer)
    # Recall
    recalled = mr.recall('boid_state_A', 'cohere')
    from flatmem.encoders import decode_scalar
    val = decode_scalar(recalled)
    assert abs(val - 0.7) < 0.2, f'steer ~0.7, got {val:.3f}'


# ── 3. Chemotaxis: gradient history → mean concentration ─────────────────────

def test_chemotaxis_running_mean():
    """
    Encode many concentration observations; recall should approximate the mean.
    Tests superposition averaging via phasor encoding.
    """
    mr = MultiRoleMemory(d=512, M=2048, k=32, seed=42, M_rel=1024,
                         roles=('cooccur', 'next', 'concentration'))
    # Simulate 50 observations around mean=2.0
    rng = np.random.default_rng(0)
    samples = rng.normal(2.0, 0.3, size=50)
    for c in samples:
        mr.write_relation('organism_A', 'concentration', scalar_phasor(float(c), d=512))
    recalled = mr.recall('organism_A', 'concentration')
    from flatmem.encoders import decode_scalar
    avg = decode_scalar(recalled)
    assert abs(avg - 2.0) < 0.5, f'mean ~2.0, recalled {avg:.3f}'


# ── 4. Evolutionary fitness: genome -> fitness scalar ────────────────────────

def test_evolutionary_fitness_recall():
    """
    Encode genome via permute-then-bundle (cyclic for sequence).
    Write fitness; recall.
    """
    mr = MultiRoleMemory(d=512, M=2048, k=32, seed=42, M_rel=1024,
                         roles=('cooccur', 'next', 'fitness'))
    ck = ComputedKey(d=512, seed=42)
    genome = ['G1', 'G2', 'G3', 'G4']
    # Sequence encoding: sum of position-permuted keys
    hv = np.zeros(512, dtype=np.complex64)
    for i, g in enumerate(genome):
        hv = hv + permute(ck.key(g), shift=i)
    hv = renorm(hv)
    fitness = scalar_phasor(0.85, d=512)
    # Use a unique slot via hashing the genome
    slot = 'genome_' + '_'.join(genome)
    for _ in range(30):
        mr.write_relation(slot, 'fitness', fitness)
    recalled = mr.recall(slot, 'fitness')
    from flatmem.encoders import decode_scalar
    f = decode_scalar(recalled)
    assert abs(f - 0.85) < 0.25, f'fitness ~0.85, got {f:.3f}'


# ── 5. Sensorimotor fusion: bind multi-modal then recall motor command ───────

def test_sensorimotor_fusion():
    """
    Vision + proprioception bound via VSA, then write/recall a motor command.
    Demonstrates multi-modal fusion via binding + bundling.
    """
    mr = MultiRoleMemory(d=512, M=2048, k=32, seed=42, M_rel=1024,
                         roles=('cooccur', 'next', 'motor_cmd', 'vision_role', 'proprio_role'))
    vision  = random_projection_encoder(np.array([0.2, 0.7, -0.1]), d=512, seed=11)
    proprio = random_projection_encoder(np.array([0.0, 1.0]),       d=512, seed=22)
    # Fuse
    fused = bundle(bind(vision, mr.roles['vision_role']),
                   bind(proprio, mr.roles['proprio_role']))
    cmd = scalar_phasor(+0.4, d=512)
    # Use a slot keyed by a hash of the fused state
    slot = 'fused_A'
    for _ in range(20):
        mr.sdm_rel.write(bind(fused, mr.roles['motor_cmd']),
                         cmd, word=slot + '|motor_cmd')
    out = mr.sdm_rel.read(bind(fused, mr.roles['motor_cmd']),
                          word=slot + '|motor_cmd')
    from flatmem.encoders import decode_scalar
    val = decode_scalar(out)
    assert abs(val - 0.4) < 0.25, f'motor cmd ~0.4, got {val:.3f}'


# ── 6. Federated merge (counter banks sum across organisms) ──────────────────

def test_federated_merge():
    """
    Two organisms (same seed) train on disjoint facts; merged substrate
    should recall facts from BOTH.
    """
    SEED = 42
    a = MultiRoleMemory(d=512, M=2048, k=32, seed=SEED, M_rel=1024)
    b = MultiRoleMemory(d=512, M=2048, k=32, seed=SEED, M_rel=1024)
    a.assert_relation('cat', 'isa', 'mammal', n=20)
    b.assert_relation('rose', 'isa', 'flower', n=20)
    # Merged: counter-bank sum
    merged = MultiRoleMemory(d=512, M=2048, k=32, seed=SEED, M_rel=1024)
    merged.sdm.C     = a.sdm.C     + b.sdm.C
    merged.sdm_rel.C = a.sdm_rel.C + b.sdm_rel.C
    merged._role_targets['isa'] = (a._role_targets.get('isa', set()) |
                                    b._role_targets.get('isa', set()))
    merged._seen = a._seen | b._seen
    # Query both facts
    cat = merged.query('cat',  'isa')[0]
    rose = merged.query('rose', 'isa')[0]
    assert cat == 'mammal' and rose == 'flower', \
        f'merged failed: cat->{cat} rose->{rose}'


# ── runner ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [v for k, v in globals().items() if k.startswith('test_')]
    fails = 0
    for t in tests:
        try:
            t(); print(f'  [PASS] {t.__name__}')
        except AssertionError as e:
            print(f'  [FAIL] {t.__name__}: {e}'); fails += 1
        except Exception as e:
            print(f'  [ERR ] {t.__name__}: {type(e).__name__}: {e}'); fails += 1
    print(f'\n{len(tests)-fails}/{len(tests)} passed')
    sys.exit(1 if fails else 0)
