# flatmem

**A brain you can fit in 233 MB that absorbs frontier-scale text and memory never grows.**
**Not a model. An organism.**

`flatmem` is a constant-RAM, content-addressable memory substrate for digital organisms, Artificial Life simulations, agent-based models, robotics, and language organisms. The total memory footprint is **fixed** regardless of how much data is written. Inspired by, and mathematically grounded in, the Marr-Albus-Kanerva cerebellar model.

---

## Why this exists

Traditional memory in AI grows with data: embedding tables (`{id: vector}`), bigram dicts, replay buffers, vector databases. RAM scales linearly with experience. Brains do not — adult human cortex is ~86 billion neurons fixed; lifetime memories live in **synaptic strengths in a fixed matrix**, not in growing dictionaries.

`flatmem` is that fixed matrix in software. Every memory you write superposes into existing counter weights at sparse locations. Recall is reconstructive (gist-based, like human memory). The substrate is **modality-blind**: text, sensor arrays, chemical gradients, robot joint angles — anything that can be encoded as a high-dimensional pattern.

## The 5-piece architecture

1. **Sparse Distributed Memory (SDM)** — fixed bank of `M` random hard-location addresses + counter rows. Writes activate top-`k` nearest by cosine; counters accumulate.
2. **VSA role-binding** — `addr(item, role) = key(item) ⊙ ROLE` (Hadamard complex multiply). Multiple relation types in ONE substrate without interference.
3. **Traffic-class bank separation** — high-traffic channels (co-occurrence) and low-traffic channels (sparse facts) live in separate fixed banks; otherwise dense traffic swamps sparse signal at shared locations.
4. **Computed-identity keys** — zero stored bytes per item. Identity = char-trigram + whole-word random phasor projection. Infinite vocabulary, no embedding table.
5. **Read-time mean-removal** — Arora's all-but-the-top, adapted to live recall. Subtracts the dominant common direction; scale-invariant sensory adaptation.

Each piece has prior art (Kanerva 1988, Plate 1995, Frady-Sommer TPAM, Arora 2017). The **integration** as a working lifelong-learning constant-RAM memory organism is novel.

## Install

```bash
pip install flatmem
# or from source
git clone https://github.com/HitoshiFTW/Flatmem-Mura-ALife-Labs
cd flatmem
pip install -e .
```

Only dependency: `numpy>=1.20`.

## Quick start

```python
from flatmem import MultiRoleMemory

mem = MultiRoleMemory(d=512, M=16384, k=64)

# Inject knowledge with reinforcement
mem.assert_relation('cat', 'isa', 'mammal', n=20)
mem.assert_relation('dog', 'isa', 'mammal', n=20)
mem.assert_relation('mammal', 'property', 'warm', n=20)

# Query
print(mem.query('cat', 'isa'))    # ('mammal', 1.0)
print(mem.query('dog', 'isa'))    # ('mammal', 1.0)

# Cross-channel composition (the property of a cat's hypernym)
mid, end = mem.chain('cat', 'isa', ['mammal', 'flower', 'tree'],
                            'property', ['warm', 'cold', 'tall'])
print(f'{mid} -> {end}')          # 'mammal -> warm'

# Read text — co-occurrence accumulates in fixed substrate
for sentence in ["the cat sat on the mat", "the dog ran fast"] * 50:
    mem.expose_cooccur(sentence)
print(mem.similarity('cat', 'dog'))   # > 0 (semantic neighbors)

# Substrate is FIXED regardless of how much you write
print(f'{mem.substrate_bytes() / 1_048_576:.0f} MB')   # always the same
```

## Universal interface

The substrate exposes **two functions**:

```python
mem.write(addr, data)      # via .relate / .write_relation / .expose_*
mem.read(addr)             # via .recall / .query / .similarity / .neighbors
```

Anything that can be encoded as a `d`-dim phasor HV can address it. `flatmem.encoders` provides helpers for the common ALife encodings:

```python
from flatmem.encoders import (
    scalar_phasor,            # encode a scalar (reward, concentration, joint angle)
    position_phasor,          # encode 2D position with spatial topology
    random_projection_encoder,# encode an arbitrary numeric vector
    bind, unbind, bundle, permute,    # VSA primitives
)
```

## ALife integration patterns

The `examples/` directory shows drop-in patterns for several ALife domains:

| File | Paradigm | What it shows |
|------|----------|---------------|
| `01_text_organism.py` | Language organism | Co-occurrence + IS-A + similarity in one substrate |
| `02_maze_agent.py` | Grid-world RL | Q-values per (state, action) without a Q-table dict |
| `03_boids_memory.py` | Flocking / swarms | Experiential steering from past visual states |
| `04_chemotaxis.py` | Chemical gradients | Running-mean concentration via phasor superposition |

The same substrate object handles all four. No special-case code.

## What it's NOT for

`flatmem` is a **cognitive / episodic / symbolic** memory substrate. It is the **wrong** choice for:
- Mass-cellular-automata grid storage at 60 Hz (use raw VRAM arrays for the grid; put `flatmem` in the agents that navigate it).
- Lookup tables requiring **exact** retrieval (recall is reconstructive / gist).
- Anything where you need 100% precision and have unlimited RAM.

If your problem is "I want to remember the GIST of a lifetime of experience in fixed bytes" — this is the right tool.

## Engineering notes

- **Real-time loops**: top-`k` activation is O(M·d). Cache strategically; don't query per frame at 60 Hz.
- **Multi-agent**: each agent has its own `MultiRoleMemory` (`~192 MB default`). Decentralize.
- **Federated merge**: agents with same `seed` have aligned hard locations; counter banks can be summed (`C_merged = C_a + C_b`) for emergent hive intelligence or generational inheritance. Capacity wall at ~hundreds of agents (noise floor grows as √N).
- **GPU**: not optimized for GPU yet. SDM top-`k` selection causes warp divergence; replace with differentiable softmax for GPU port.

## Run the tests

```bash
python tests/test_basics.py        # core correctness
python tests/test_universality.py  # ALife integration patterns
```

## Research paper

- [PAPER.md](./PAPER.md) — markdown version with full architecture rationale, empirical results, prior-art discussion, and novelty analysis.
- [paper/paper.tex](./paper/paper.tex) — LaTeX source for the arXiv submission, with figures generated from real experimental data (substrate flatness, mean-removal sweep, no-forgetting bars, vision MNIST results). See [paper/README.md](./paper/README.md) for build instructions.

Originally developed inside the **Ikigai** organism project at **Mura ALife Labs** as the constant-RAM memory substrate for a language-grounded digital organism.

## Cite

If you use `flatmem` in research:

```bibtex
@misc{siddhpara2026flatmem,
  author       = {Siddhpara, Prince},
  title        = {flatmem: A Constant-RAM Content-Addressable Memory Substrate for Digital Organisms},
  year         = {2026},
  publisher    = {Mura ALife Labs},
  howpublished = {\url{https://github.com/HitoshiFTW/Flatmem-Mura-ALife-Labs}},
}
```

## License

MIT. See [LICENSE](./LICENSE).

---

<p align="center">
  <b>Mura ALife Labs</b> · 2026<br>
  <i>Building digital organisms, not models.</i>
</p>
