"""
flatmem -- Constant-RAM content-addressable memory for digital organisms.

A universal flat-memory substrate based on Sparse Distributed Memory + VSA
role-binding + computed-identity keys + read-time mean-removal. Designed as
a drop-in lifelong-learning memory for Artificial Life simulations,
agent-based models, robotics, and language organisms.

Two-function interface:
    write(addr, data)
    read(addr) -> cleaned data

Substrate size is FIXED regardless of how many items are written.
Modality-blind: addresses can be any high-dimensional pattern.

Quick start:
    from flatmem import MultiRoleMemory, ComputedKey
    mem = MultiRoleMemory(d=512, M=16384, k=64, roles=('next', 'reward'))
    mem.relate('cat', 'isa', 'mammal')
    print(mem.query('cat', 'isa'))  # ('mammal', 1.0)
"""

from .core import (
    ComputedKey,
    VSASDM,
    MultiRoleMemory,
    tokenize,
    cos,
    renorm,
)

from .encoders import (
    scalar_phasor,
    decode_scalar,
    position_phasor,
    random_projection_encoder,
    bind,
    unbind,
    bundle,
    permute,
)

__version__ = "0.1.0"
__all__ = [
    "ComputedKey", "VSASDM", "MultiRoleMemory",
    "tokenize", "cos", "renorm",
    "scalar_phasor", "decode_scalar", "position_phasor",
    "random_projection_encoder",
    "bind", "unbind", "bundle", "permute",
]
