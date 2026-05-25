"""
flatmem.encoders -- Helper encoders for converting raw state to phasor HVs.

Domain-agnostic primitives + common patterns for ALife state encoding.
All return component-wise unit-phasor complex64 arrays of dimension d.

VSA ops (bind, bundle, permute) provided as convenience wrappers around
component-wise complex multiply / sum / np.roll.
"""

import hashlib
import numpy as np

from .core import renorm


# ── primitives ───────────────────────────────────────────────────────────────

def _seeded_phasor(tag, d, seed):
    """Deterministic random phasor HV from a tag (string or int)."""
    h = hashlib.sha256(f'{tag}:{seed}'.encode()).digest()
    s = int.from_bytes(h[:8], 'little')
    rng = np.random.default_rng(s)
    ph = rng.uniform(-np.pi, np.pi, d).astype(np.float32)
    return np.exp(1j * ph).astype(np.complex64)


def _qaxis(d, seed=12345):
    """Bipolar random ±1 axis vector for fractional-power scalar encoding."""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)


# ── scalar encoding (Fractional Power Encoding) ──────────────────────────────

def scalar_phasor(value, d=512, omega=0.05, seed=12345, q_axis=None):
    """
    Encode a real scalar as a phasor HV. Phase per component = value * omega * q_axis.

    Reversible: median(angle / (omega * q_axis)) recovers value (modulo 2pi/omega).
    Linear: scalar_phasor(a+b) ≈ scalar_phasor(a) * scalar_phasor(b)  (componentwise).

    Used for continuous quantities (reward, concentration, joint angle, time).
    """
    if q_axis is None:
        q_axis = _qaxis(d, seed)
    phase = float(value) * omega * q_axis
    return np.exp(1j * phase).astype(np.complex64)


def decode_scalar(hv, omega=0.05, seed=12345, q_axis=None):
    """Inverse of scalar_phasor (median across components for noise tolerance)."""
    if q_axis is None:
        q_axis = _qaxis(len(hv), seed)
    phases = np.angle(hv).astype(np.float32)
    cs = phases / (omega * q_axis + 1e-9)
    return float(np.median(cs))


# ── 2D position encoding ─────────────────────────────────────────────────────

def position_phasor(x, y, d=512, omega=0.05, scale=1.0, seed=12345):
    """
    Encode 2D position (x, y) with two independent axes bound together.
        phasor(x, x_axis) ⊙ phasor(y, y_axis)
    Spatial proximity → high cosine. Useful for grid/continuous-position ALife.
    """
    x_axis = _qaxis(d, seed)
    y_axis = _qaxis(d, seed + 1)
    x_hv = np.exp(1j * float(x) * scale * omega * x_axis).astype(np.complex64)
    y_hv = np.exp(1j * float(y) * scale * omega * y_axis).astype(np.complex64)
    return (x_hv * y_hv).astype(np.complex64)


# ── arbitrary vector → HV (random projection) ────────────────────────────────

def random_projection_encoder(vector, d=512, seed=42, project_matrix_cache={}):
    """
    Map an arbitrary ndarray to a phasor HV via fixed random projection.
        hv = renorm( exp(i * (proj_matrix @ vector)) )
    Cache the projection matrix in process for reuse.

    For sensor arrays, embeddings, or any dense numeric state.
    """
    v = np.asarray(vector, dtype=np.float32).ravel()
    key = (v.shape[0], d, seed)
    if key not in project_matrix_cache:
        rng = np.random.default_rng(seed)
        project_matrix_cache[key] = rng.standard_normal((d, v.shape[0])).astype(np.float32)
    P = project_matrix_cache[key]
    phase = (P @ v).astype(np.float32)
    return np.exp(1j * phase).astype(np.complex64)


# ── VSA operations ───────────────────────────────────────────────────────────

def bind(a, b):
    """VSA binding via component-wise complex multiply. Inverse: bind(c, conj(b))."""
    return (a * b).astype(np.complex64)


def unbind(c, b):
    """Inverse of bind: recover a from c=bind(a,b) when b is a unit phasor."""
    return (c * np.conj(b)).astype(np.complex64)


def bundle(*vs):
    """VSA bundling via sum + component-wise renorm. Many HVs into one."""
    acc = np.zeros_like(vs[0])
    for v in vs:
        acc = acc + v
    return renorm(acc)


def permute(hv, shift=1):
    """Cyclic shift for sequence position. permute^i acts like position i."""
    return np.roll(hv, shift).astype(np.complex64)
