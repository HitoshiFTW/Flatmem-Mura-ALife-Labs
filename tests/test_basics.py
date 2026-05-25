"""
Basic correctness tests for flatmem.

Run: python -m pytest tests/test_basics.py -v
Or:  python tests/test_basics.py
"""

import sys, os, pathlib, tempfile, pickle
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from flatmem import (
    ComputedKey, VSASDM, MultiRoleMemory,
    scalar_phasor, decode_scalar, position_phasor,
    random_projection_encoder, bind, unbind, bundle, permute,
    cos, renorm,
)


def test_computed_key_deterministic():
    ck = ComputedKey(d=128, seed=42, word_weight=4.0)
    k1 = ck.key('cat')
    ck2 = ComputedKey(d=128, seed=42, word_weight=4.0)
    k2 = ck2.key('cat')
    assert np.allclose(k1, k2), 'same seed -> same key'
    assert abs(cos(k1, k2, 128) - 1.0) < 1e-5, 'self-similarity = 1.0'


def test_computed_key_discriminates_similar_strings():
    """word_weight=4 must drop cos(little, litttle) well below 0.5."""
    ck = ComputedKey(d=512, seed=42, word_weight=4.0)
    pairs = [('little', 'litttle'), ('the', 'they'), ('was', 'wasted')]
    for a, b in pairs:
        c = cos(ck.key(a), ck.key(b), 512)
        assert c < 0.3, f'cos({a!r}, {b!r}) = {c:.3f} too high'


def test_vsasdm_substrate_flat():
    """Substrate bytes stay constant after many writes."""
    sdm = VSASDM(d=128, M=512, k=8, seed=42)
    sub0 = sdm.substrate_bytes()
    ck = ComputedKey(d=128, seed=42)
    for w in [f'word{i}' for i in range(1000)]:
        sdm.write(ck.key(w), ck.key(f'data_{w}'), word=w)
    sub1 = sdm.substrate_bytes()
    assert sub0 == sub1, f'substrate grew: {sub0} -> {sub1}'


def test_vsasdm_associative_recall():
    """Write key->value, read should reconstruct value (cos close to 1 for one item)."""
    sdm = VSASDM(d=512, M=2048, k=32, seed=42)
    ck = ComputedKey(d=512, seed=42)
    addr, val = ck.key('cat'), ck.key('mammal')
    sdm.write(addr, val, word='cat')
    out = sdm.read(addr, word='cat')
    c = cos(out, val, 512)
    assert c > 0.9, f'self-recall cos={c:.3f}'


def test_multirole_role_orthogonality():
    """Different roles for same item -> orthogonal addresses (|cos| < 0.15)."""
    mr = MultiRoleMemory(d=512, M=2048, k=32, seed=42, M_rel=1024)
    for ra, rb in [('cooccur', 'isa'), ('isa', 'sensory'), ('sensory', 'property')]:
        c = mr.role_orthogonality('cat', ra, rb)
        assert abs(c) < 0.15, f'cos({ra}, {rb}) = {c:.3f} not orthogonal'


def test_multirole_isa_recovery():
    """Inject 6 IS-A facts, query should recover them all."""
    mr = MultiRoleMemory(d=512, M=2048, k=32, seed=42, M_rel=1024)
    facts = {'dog': 'mammal', 'cat': 'mammal', 'rose': 'flower',
             'oak': 'tree', 'apple': 'fruit', 'car': 'vehicle'}
    for h, y in facts.items():
        mr.assert_relation(h, 'isa', y, n=20)
    for h, y in facts.items():
        got, _ = mr.query(h, 'isa')
        assert got == y, f'isa({h}) -> {got}, expected {y}'


def test_multirole_chain_composition():
    """Cross-channel: property(isa(cat)) -> warm."""
    mr = MultiRoleMemory(d=512, M=2048, k=32, seed=42, M_rel=1024)
    for h, y in {'cat': 'mammal', 'dog': 'mammal', 'oak': 'tree',
                 'car': 'vehicle'}.items():
        mr.assert_relation(h, 'isa', y, n=20)
    for h, p in {'mammal': 'warm', 'tree': 'tall', 'vehicle': 'fast'}.items():
        mr.assert_relation(h, 'property', p, n=20)
    hypers = ['mammal', 'flower', 'tree', 'fruit', 'vehicle']
    props  = ['warm', 'cold', 'tall', 'short', 'fast', 'slow']
    expected = {'cat': 'warm', 'dog': 'warm', 'oak': 'tall', 'car': 'fast'}
    ok = 0
    for w, exp in expected.items():
        mid, end = mr.chain(w, 'isa', hypers, 'property', props)
        if end == exp: ok += 1
    assert ok >= 3, f'chain {ok}/4'


def test_multirole_persistence():
    """save+load via pickle preserves recall."""
    mr = MultiRoleMemory(d=256, M=1024, k=16, seed=42, M_rel=512)
    mr.assert_relation('cat', 'isa', 'mammal', n=20)
    pre = mr.query('cat', 'isa')
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        pickle.dump(mr, f)
        path = f.name
    with open(path, 'rb') as f:
        mr2 = pickle.load(f)
    post = mr2.query('cat', 'isa')
    os.unlink(path)
    assert pre == post, f'pre={pre} post={post}'


def test_scalar_encoder_roundtrip():
    """encode then decode recovers value within tolerance."""
    for v in [0.5, -1.0, 3.14, 12.0]:
        hv = scalar_phasor(v, d=512, omega=0.05)
        rec = decode_scalar(hv, omega=0.05)
        assert abs(rec - v) < 0.5, f'scalar {v} -> {rec}'


def test_position_encoder_topology():
    """Nearby positions -> high cosine; far positions -> low cosine."""
    a = position_phasor(0.0, 0.0, d=512)
    b = position_phasor(0.1, 0.1, d=512)   # near
    c = position_phasor(50.0, 50.0, d=512) # far
    assert cos(a, b, 512) > cos(a, c, 512), 'near must be closer than far'


def test_vsa_bind_unbind():
    """unbind(bind(a, b), b) == a (up to FP)."""
    ck = ComputedKey(d=512, seed=42)
    a, b = ck.key('alpha'), ck.key('beta')
    c = bind(a, b)
    a_recovered = unbind(c, b)
    sim = cos(a, a_recovered, 512)
    assert sim > 0.99, f'bind/unbind recovery cos={sim:.4f}'


def test_random_projection_encoder():
    """Same input -> same HV. Close inputs -> close HVs."""
    a = random_projection_encoder(np.array([1.0, 2.0, 3.0]), d=256, seed=7)
    b = random_projection_encoder(np.array([1.0, 2.0, 3.0]), d=256, seed=7)
    c = random_projection_encoder(np.array([1.0, 2.0, 3.001]), d=256, seed=7)
    assert np.allclose(a, b), 'determinism'
    assert cos(a, c, 256) > 0.99, 'small perturbation -> close HV'


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
