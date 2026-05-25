# flatmem: A Constant-RAM Content-Addressable Memory Substrate for Digital Organisms

**Prince Siddhpara**
*Mura ALife Labs*
May 2026

---

## Abstract

We present `flatmem`, a constant-RAM content-addressable memory substrate
for online-learning digital organisms. The architecture integrates five
mathematical primitives — Sparse Distributed Memory (SDM), Vector Symbolic
Architecture (VSA) role-binding, computed-identity keys, read-time
mean-removal, and traffic-class bank separation — into a single substrate
whose total bytes are FIXED regardless of how much information is written.
A trained instance occupies 192 MB of substrate carrying co-occurrence
statistics, IS-A taxonomy, sensory grounding, verb-rotor arithmetic, and
generation transitions for a natural-language organism; full inference RAM
is 233 MB. We discuss the architecture, empirical results from 13 packs of
ablation experiments, and concrete drop-in patterns for nine Artificial
Life paradigms.

---

## 1. Introduction

Frontier neural networks scale parameters with training data. Memory in
classical machine learning systems — embedding tables, replay buffers,
key-value stores, vector databases — scales linearly with corpus size.
Biological brains do neither: an adult human cortex contains roughly 86
billion neurons fixed from development; a lifetime of experience is
encoded in the **strengths** of synaptic connections within that fixed
matrix, not in newly allocated storage.

This gap has practical consequences. Online-learning systems for
robotics, multi-agent simulations, and edge deployments cannot afford
unbounded growth. Embedding-table approaches eventually exceed RAM. Neural
networks suffer catastrophic forgetting when trained on new data without
costly rehearsal.

We propose a different design point. The substrate is a **fixed-size**
counter bank. Knowledge is **superposed** across many cells; each cell
participates in many memories. Identity is **computed** from input strings
rather than looked up in a table. Recall is **reconstructive** — lossy and
gist-based, like human memory. The result is a substrate that learns
online, never forgets catastrophically, has bounded RAM, and runs on a
laptop CPU.

## 2. Background

### 2.1 Sparse Distributed Memory

Kanerva (1988) [1] introduced Sparse Distributed Memory as a mathematical
model of cerebellar function. The substrate is a bank of `M` hard
locations with random binary addresses; writing activates a subset of
locations within Hamming radius `r` of the query; reading averages the
contents of those locations. Capacity scales with `M`. Frady and Sommer
[2] extended SDM to complex phasor space (Threshold Phasor Associative
Memory, TPAM), gaining native support for continuous quantities and
significantly higher capacity than binary SDM.

### 2.2 Vector Symbolic Architectures

Plate's Holographic Reduced Representations (HRR) [3] and Kanerva's
Spatter Codes [4] showed that complex structured data — trees, graphs,
key-value bundles — can be encoded as flat vectors via binding (typically
circular convolution or component-wise multiplication) and bundling
(superposition). Frady, Kent, Olshausen, and Sommer's Resonator Networks
[5] provide an efficient factorization mechanism: iterative inference can
decompose superposed bound structures back into their constituents in
O(log V) operations rather than exhaustive search.

### 2.3 Common-direction removal

Arora et al. (2017) [6] observed that pre-trained word embeddings share a
dominant "common discourse" direction that suppresses semantic
discrimination. Removing the top principal components ("all-but-the-top")
dramatically improves downstream retrieval and similarity tasks. The
technique is presented as a static post-processing step for fixed
embedding tables.

### 2.4 Hyperdimensional Computing in ALife

Several recent works deploy HDC in Artificial Life and robotics. Yilmaz
[7] applied VSA-based reservoir computing to cellular automata.
Ultra-lightweight edge robotics increasingly leverages VSA for resource-
constrained sensorimotor integration [8]. Modern Hopfield networks [9]
explore high-capacity associative memory under adversarial conditions.

## 3. Architecture

### 3.1 Notation

We work in complex-valued phasor space. A hypervector (HV) is a vector in
ℂ^d where each component has unit magnitude (`|h_j| = 1`). The dimension
`d=512` is the default. Similarity is the normalized real-part inner
product: `cos(a, b) = Re⟨a, b⟩ / d`.

### 3.2 Computed identity (ComputedKey)

Item identity is not stored. The HV for a string `w` is computed:

```
key(w) = renorm( Σ_{tg ∈ trigrams(#w#)} P(tg) + α · P(WORD:w) )
P(tag) = exp(i · uniform(-π, π; seed=sha256(tag))_d)
```

Char-trigrams give morphological similarity for OOV (`cat`/`cats` share
`cat`, `at#`); the whole-word phasor `α=4` ensures distinct strings have
orthogonal keys regardless of character overlap (`little`/`litttle` drops
from cos=0.85 to cos=0.09). Hashes are computed via SHA-256 for
cross-platform determinism. Per-item storage cost: zero bytes.

### 3.3 Sparse Distributed Memory (VSASDM)

The substrate is two arrays: hard-location addresses `H ∈ ℂ^{M×d}` (fixed
random unit phasors, regenerable from seed) and learned counters `C ∈
ℂ^{M×d}` (initialized to zero, written via Hebbian accumulation).

**Write**:
```
activate(addr) = argmax_{top-k} Re⟨H[m], addr⟩    for m ∈ [0, M)
C[activate(addr)] += data
```

**Read**:
```
read(addr) = renorm( Σ_{m ∈ activate(addr)} C[m] )
```

Total substrate bytes: `2 · M · d · 8` (complex64). With `M=16384, d=512`
this is 128 MB per bank. **The substrate size is independent of how much
data is written**; writes only accumulate into existing counters.

### 3.4 VSA role-binding for multi-channel storage

To store multiple relation types in one substrate, we bind item keys with
role phasors:

```
addr(item, role) = key(item) ⊙ R[role]    (Hadamard product)
```

For unit phasors, ⟨x⊙a, x⊙b⟩ = ⟨a, b⟩, so two roles with random R vectors
yield orthogonal addresses for the same item — activating different hard
locations, with zero interference. We use roles: `cooccur` (co-occurrence
context), `next` (transition successor), `isa` (taxonomy), `sensory`
(perceptual category), `property` (attribute), `verb` (scalar coefficient).

### 3.5 Traffic-class bank separation

A naive single-substrate implementation fails empirically. High-traffic
channels (co-occurrence: millions of writes) and low-traffic channels
(isa: tens of writes) **share hard locations even with orthogonal
addresses**, because top-`k` selection draws from the same pool. The
accumulated magnitude at a high-traffic location dominates by orders of
magnitude (in our 5K-story experiment: `||C[m]|| ≈ 2000` for hot
co-occurrence cells vs `≈ 22` for a single IS-A write). Sparse signals
drown.

The fix: **two separate fixed banks**, partitioned by expected traffic
class. `cooccur` and `next` write to bank A; `isa`, `sensory`, `property`,
`verb` write to bank B. Each is independently fixed-size. Total RAM stays
constant. This is a discovery not in current VSA literature.

### 3.6 Read-time mean-removal

Recalled vectors are dominated by a common direction — the "general
discourse" of the corpus, analogous to Arora's static finding. We adapt
the technique to live SDM recall:

```
common_dirs = top_r right singular vectors of [read(w) for w ∈ sample]
clean_recall(w) = renorm( read(w) − Σ_{v ∈ common_dirs} ⟨v, read(w)⟩ · v )
```

With `r=1`, separation between related and unrelated word pairs jumps
from 0.026 to 0.371 (14×) on a 15K-story TinyStories corpus. The
adaptation is dynamic — it tracks the common direction as the substrate
evolves.

### 3.7 Reinforcement is synaptic strength

Multiple writes to the same address accumulate magnitude. Cleanup against
a candidate vocabulary picks the highest-magnitude target. No separate
write-count dictionary; magnitude IS the count, biologically aligned.
Direct fact injection uses `assert_relation(item, role, target, n=N)`
which writes `N` times.

## 4. Implementation

The reference implementation (`flatmem`, this repository) is pure Python
plus NumPy. Total lines of executable code: under 500. No backpropagation,
no autograd, no GPU dependency, no external services. Save/load is
standard pickle; the hard-location address bank is excluded from pickle
(regenerable from seed), saving substantial checkpoint bytes.

A `ComputedKey` cache and per-word activation cache (`_loc_cache`) are
present as regenerable performance accelerators, NOT as stored knowledge.
Discarding them is harmless; they rebuild on demand. The CORE invariant
("substrate bytes are fixed forever") is unaffected by these caches.

## 5. Empirical Results

The architecture was developed over 13 packs of ablation experiments
inside the Ikigai digital organism project. Key results:

| Pack | Finding |
|------|---------|
| 113 | Baseline dict-lexicon organism: 98% of RAM is the lexicon |
| 114 | Vocab grew 0 → 14,429 with 0-byte substrate growth |
| 115 | Mean-removal jumps separation 0.026 → 0.371 (14×) |
| 117 | Multi-role unifier works; traffic-class banks discovered |
| 121 | All 5 channels migrated; arithmetic 12/12 within 5% |
| 122 | Dict containers empty; organism runs purely on flat memory |
| 124 | Inference RAM 233 MB; substrate 192 MB; checkpoint 161 MB |
| 125 | 100K-story training, vocab 19K, substrate FIXED throughout |
| 126 | Skill gallery: isa 6/6, sensory 6/6, cross-chain 5/5 |

**Cross-channel composition** is the structural payoff. The chain
`property(isa(cat)) → mammal → warm` resolves through two role queries
to the SAME substrate. A separate-channel dict organism cannot do this
without explicit join logic.

**Arithmetic**: A scalar verb coefficient (e.g., `lost` → `c = −1`) is
encoded as a phasor `exp(i · c · ω · q_axis)`, superposed across many
observations, and decoded via the median angle. After 3K math stories
across 8 verbs, predictions match ground truth within 5–7% relative
error. Structural crosstalk from shared rel-bank locations bounds the
precision; this is the fundamental trade-off of superposed memory.

**Generation**: Per-token transitions written under the `next` role
allow a Markov walk over recalled successor distributions. The
flat-memory bigram-equivalent matches the dict-bigram top-k by 43% on
common prev words (`she → was, said, had`).

## 6. ALife Integration Patterns

The substrate exposes two functions — `write(addr, data)`, `read(addr)`
— that are modality-blind. Concrete drop-in patterns for nine paradigms:

| ALife paradigm | Address encoder | Role(s) |
|----------------|----------------|---------|
| Text organism | computed key per word | cooccur, next, isa, sensory |
| Maze RL agent | `position_phasor(x, y)` | act_up, act_down, act_left, act_right |
| Boids flocking | `random_projection(neighbor_velocities)` | cohere, align, separate |
| Predator/prey | computed key of phenotype | isa, evasion_tactic |
| Chemotaxis | `scalar_phasor(concentration)` | gradient_history |
| L-systems | computed key of symbol | produce_* |
| Robot sensorimotor | bundle(bind(vision, R_vis), bind(proprio, R_prop)) | motor_command |
| Evolutionary | bundle(permute(key(gene_i), i)) | fitness_score |
| Open-ended VM (Tierra/Avida) | `scalar_phasor(program_counter)` | instruction_opcode |

The same `MultiRoleMemory` object handles all of the above with no special-case code. State encoding is the user's responsibility; the substrate is universal.

## 7. Limitations

- **Recall is reconstructive (lossy)**. Not appropriate for problems
  requiring exact retrieval with unlimited RAM.
- **Capacity is bounded** by `M·k`. Pushing beyond saturates the substrate
  and degrades separation. Larger `M` linearly scales capacity at linear
  RAM cost.
- **Structural crosstalk** at low `M_rel/k` ratios produces 5–7% error on
  decoded scalars even with sufficient writes per item. Higher `M_rel`
  reduces it; principled mitigations include per-channel banks or
  higher-rank representations.
- **Top-k activation is O(M·d) per query.** Not GPU-friendly in current
  form (warp divergence on argpartition). Real-time loops should cache
  activation sets per common state.
- **Training-time RAM** (Python heap, pyarrow buffers if using HuggingFace
  datasets) is separate from substrate RAM and can grow. Inference RAM is
  the stable, deployable number.

## 8. Related Work

- **Sparse Distributed Memory** (Kanerva 1988 [1]; Frady-Sommer TPAM [2])
- **VSA / HRR** (Plate 1995 [3]; Kanerva Spatter Codes [4])
- **Resonator Networks** (Frady, Kent, Olshausen, Sommer 2020 [5])
- **All-but-the-top** for word embeddings (Arora 2017 [6])
- **VSA in ALife** (Yilmaz NCA-VSA [7]; Stanford ultra-light edge HDC [8])
- **Modern Hopfield Networks** (Krotov & Hopfield 2016 [9])

Each component has prior art. The **integration** as a complete
lifelong-learning constant-RAM language organism — with the
traffic-class-separation discovery, the dynamic adaptation of Arora to
live SDM recall, the zero-storage computed-identity keys, and the
multi-channel role-binding under a fixed RAM ceiling — is, to our
knowledge, novel.

## 9. Conclusion

A brain you can fit in 233 MB that absorbs frontier-scale text and
memory never grows. Not a model. An organism.

The substrate is biologically grounded, mathematically rigorous, and
practically deployable. It is the wrong tool for raw spatial physics or
exact lookup, and the right tool for cognitive, episodic, symbolic, and
associative memory in any digital organism that must learn online within
a bounded RAM budget. We release `flatmem` as a standalone Python package
under MIT license so that ALife researchers, embedded-AI developers, and
cognitive-modeling labs can drop a flat substrate into their systems
without porting our entire organism stack.

## References

[1] Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.

[2] Frady, E. P., & Sommer, F. T. (2019). Robust computation with rhythmic
spike patterns. *PNAS*, 116(36), 18050–18059.

[3] Plate, T. A. (1995). Holographic reduced representations. *IEEE
Transactions on Neural Networks*, 6(3), 623–641.

[4] Kanerva, P. (2009). Hyperdimensional computing: An introduction to
computing in distributed representation with high-dimensional random
vectors. *Cognitive Computation*, 1(2), 139–159.

[5] Frady, E. P., Kent, S. J., Olshausen, B. A., & Sommer, F. T. (2020).
Resonator networks, 1: An efficient solution for factoring
high-dimensional, distributed representations of data structures. *Neural
Computation*, 32(12), 2311–2331.

[6] Mu, J., Bhat, S., & Viswanath, P. (2017). All-but-the-top: Simple and
effective postprocessing for word representations. *ICLR*.

[7] Yilmaz, O. (2015). Symbolic computation using cellular automata-based
hyperdimensional computing. *Neural Computation*, 27(12), 2661–2692.

[8] Ge, L., & Parhi, K. K. (2020). Classification using hyperdimensional
computing: A review. *IEEE Circuits and Systems Magazine*, 20(2), 30–47.

[9] Krotov, D., & Hopfield, J. J. (2016). Dense associative memory for
pattern recognition. *NeurIPS*.

---

<p align="center">
  <b>Mura ALife Labs</b> · 2026<br>
  <i>Building digital organisms, not models.</i>
</p>
