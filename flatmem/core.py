"""
flatmem.core -- Core flat-memory substrate.

Three building blocks:
  1. ComputedKey         -- zero-storage identity (char-trigram + whole-word phasor)
  2. VSASDM              -- fixed Sparse Distributed Memory bank
  3. MultiRoleMemory     -- multi-channel substrate via VSA role-binding,
                            with traffic-class bank separation, dense
                            (cooccur/next) vs sparse (isa/sensory/property/verb)

Recall is reconstructive: read returns superposed sum + optional
mean-removal of common directions for scale-invariant adaptation.
"""

import re
import hashlib
import numpy as np


# ── utilities ────────────────────────────────────────────────────────────────

def tokenize(text):
    """Lowercase, strip non-alphanumeric (keep apostrophe), split on whitespace."""
    return [t for t in re.sub(r"[^a-z0-9'\s]", ' ', text.lower()).split() if t]


def renorm(hv):
    """Component-wise unit-phasor normalization for ℂ^d vectors."""
    mags = np.abs(hv)
    mags = np.where(mags > 1e-9, mags, 1.0)
    return (hv / mags).astype(np.complex64)


def cos(a, b, d):
    """Phasor cosine similarity: Re<a,b> / d."""
    return float(np.real(np.vdot(a, b))) / d


# ── 1. Computed (zero-storage) identity ──────────────────────────────────────

class ComputedKey:
    """
    Generates a deterministic phasor HV for any string identifier WITHOUT
    storing it. Identity is reconstructed on demand from the string.

    key(w) = renorm( Σ phasor(char_trigrams(w)) + word_weight · phasor(WORD:w) )

    - char_trigrams give similarity for OOV (typos, morphology)
    - whole-word phasor (word_weight > 0) ensures distinct words are
      orthogonal regardless of character overlap; recommended word_weight=4

    Cross-platform deterministic (hashlib, not Python's salted hash()).
    The _cache dict is a regenerable accelerator, NOT stored knowledge.
    """

    def __init__(self, d=512, seed=114, word_weight=4.0):
        self.d = int(d)
        self.seed = int(seed)
        self.word_weight = float(word_weight)
        self._cache = {}

    def _seeded_phasor(self, tag):
        h = hashlib.sha256(f'{tag}:{self.seed}'.encode()).digest()
        s = int.from_bytes(h[:8], 'little')
        rng = np.random.default_rng(s)
        ph = rng.uniform(-np.pi, np.pi, self.d).astype(np.float32)
        return np.exp(1j * ph).astype(np.complex64)

    def key(self, word):
        """Return the deterministic HV for `word`."""
        if word in self._cache:
            return self._cache[word]
        w = f'#{word}#'
        trigrams = [w[i:i + 3] for i in range(len(w) - 2)] or [w]
        accum = np.zeros(self.d, dtype=np.complex64)
        for tg in trigrams:
            accum += self._seeded_phasor(tg)
        if self.word_weight:
            accum += self.word_weight * self._seeded_phasor(f'WORD:{word}')
        k = renorm(accum)
        self._cache[word] = k
        return k

    def __getstate__(self):
        return {'d': self.d, 'seed': self.seed, 'word_weight': self.word_weight}

    def __setstate__(self, s):
        self.d = s['d']
        self.seed = s['seed']
        self.word_weight = float(s.get('word_weight', 0.0))
        self._cache = {}


# ── 2. Sparse Distributed Memory (Kanerva 1988, complex variant) ────────────

class VSASDM:
    """
    Fixed bank of M hard-location addresses + counter rows.

    write(addr, data, word=None):
        Activate top-k nearest hard locations by Re<H[m], addr>;
        add `data` to counter rows at those k locations.
    read(addr, word=None):
        Sum counter rows at activated locations, return renormalized.

    Substrate size = M · d · 8 bytes (Hconj) + same (C). FIXED.
    The Hconj address bank is regenerable from `seed` — dropped from pickle.
    Per-word activation set is cached in _loc_cache (regenerable).

    consolidate_every (optional, dense banks only): auto-call consolidate()
    after every N writes. Per-row L2 renorm. Off by default (0).
    """

    def __init__(self, d=512, M=16384, k=64, seed=114, consolidate_every=0):
        self.d = int(d); self.M = int(M); self.k = int(k); self.seed = int(seed)
        self.consolidate_every = int(consolidate_every)
        self._writes_since = 0
        self.n_consolidations = 0
        self.Hconj = self._make_Hconj()
        self.C = np.zeros((self.M, self.d), dtype=np.complex64)
        self._loc_cache = {}

    def _make_Hconj(self):
        rng = np.random.default_rng(self.seed + 7)
        ph = rng.uniform(-np.pi, np.pi, (self.M, self.d)).astype(np.float32)
        return np.exp(-1j * ph).astype(np.complex64)  # conj(exp(i*ph))

    def _activate(self, addr):
        sims = (self.Hconj @ addr).real
        return np.argpartition(-sims, self.k)[:self.k]

    def locs(self, addr, word=None):
        if word is not None and word in self._loc_cache:
            return self._loc_cache[word]
        idx = self._activate(addr)
        if word is not None:
            self._loc_cache[word] = idx
        return idx

    def locs_batch(self, addrs, words):
        """Batch-activate many addresses in ONE matmul (speed win for cold cache)."""
        out = [None] * len(words)
        need_rows, need_i = [], []
        for i, wd in enumerate(words):
            cached = self._loc_cache.get(wd)
            if cached is not None:
                out[i] = cached
            else:
                need_rows.append(addrs[i]); need_i.append(i)
        if need_rows:
            A = np.stack(need_rows)                  # (m, d)
            sims = (A @ self.Hconj.T).real           # (m, M)
            for r, i in enumerate(need_i):
                idx = np.argpartition(-sims[r], self.k)[:self.k]
                self._loc_cache[words[i]] = idx
                out[i] = idx
        return out

    def write(self, addr, data, word=None):
        self.C[self.locs(addr, word)] += data
        if self.consolidate_every:
            self._writes_since += 1
            if self._writes_since >= self.consolidate_every:
                self.consolidate()
                self._writes_since = 0
                self.n_consolidations += 1

    def read(self, addr, word=None):
        return renorm(self.C[self.locs(addr, word)].sum(axis=0))

    def consolidate(self):
        """Per-row L2 renormalization. Bounds magnitude drift. Loc cache stays valid."""
        nrm = np.linalg.norm(self.C, axis=1, keepdims=True)
        nrm = np.where(nrm > 1e-9, nrm, 1.0)
        self.C = (self.C / nrm).astype(np.complex64)

    def substrate_bytes(self):
        """Total fixed bytes of the substrate."""
        return self.Hconj.nbytes + self.C.nbytes

    def __getstate__(self):
        s = self.__dict__.copy()
        s['Hconj'] = None
        s['_loc_cache'] = {}
        return s

    def __setstate__(self, s):
        self.__dict__.update(s)
        if getattr(self, 'Hconj', None) is None:
            self.Hconj = self._make_Hconj()
        if not hasattr(self, '_loc_cache') or self._loc_cache is None:
            self._loc_cache = {}


# ── 3. Multi-role substrate ──────────────────────────────────────────────────

class MultiRoleMemory:
    """
    A flat memory holding multiple relation types in ONE substrate via VSA
    role-binding (key(item) ⊙ ROLE_phasor = unique address).

    Two banks separated by traffic class:
      - dense  (high write traffic): cooccur, next   -- one bank
      - sparse (low write traffic):  isa, sensory, property, verb, ...

    Why two banks: shared bank lets a high-traffic channel (millions of
    writes) swamp a low-traffic one (single facts) at shared hard locations.
    Empirically necessary. Both banks are fixed-size; total is still constant.

    Methods:
      relate(item, role, target)        -- store item-role->target (a string)
      write_relation(item, role, hv)    -- store item-role->arbitrary HV
      recall(item, role)                -- raw reconstructive read
      query(item, role, candidates=None)-- recall + cleanup -> best candidate
      chain(item, role_a, c_a, role_b, c_b) -- two-hop cross-channel
      expose_cooccur(text)              -- co-occurrence window write
      expose_transitions(text)          -- per-token (prev,next) write
      next_word_candidates(prev, top_k) -- bigram-equivalent, vectorized cleanup
      similarity(w1, w2)                -- mean-removed cooccur cosine
      neighbors(word, k)                -- nearest seen words
      consolidate()                     -- per-bank consolidation
      expose_verb_observation(verb, c)  -- Channel 2 scalar via phase encoding
      predict_verb_coefficient(verb)    -- decode learned scalar
      assert_relation(item, role, target, n=20) -- reinforced fact injection
    """

    DEFAULT_ROLES = ('cooccur', 'next', 'isa', 'sensory', 'property', 'verb')
    DENSE_ROLES   = {'cooccur', 'next'}

    def __init__(self, d=512, M=16384, k=64, seed=114, window=3, remove_r=1,
                 svd_sample=2000, roles=DEFAULT_ROLES, M_rel=8192,
                 consolidate_every=0, q_omega=0.05, q_seed=12345):
        self.d = int(d); self.window = int(window)
        self.remove_r = int(remove_r); self.svd_sample = int(svd_sample)
        self.ck = ComputedKey(d=d, seed=seed)
        # Two banks: dense (cooccur/next) vs sparse (isa/sensory/...)
        self.sdm     = VSASDM(d=d, M=M,     k=k, seed=seed,     consolidate_every=consolidate_every)
        self.sdm_rel = VSASDM(d=d, M=M_rel, k=k, seed=seed + 1)
        # Role phasors (fixed)
        rng = np.random.default_rng(seed + 999)
        self.roles = {}
        for name in roles:
            ph = rng.uniform(-np.pi, np.pi, self.d).astype(np.float32)
            self.roles[name] = np.exp(1j * ph).astype(np.complex64)
        self._seen          = set()
        self._cooccur_seen  = set()
        self._role_targets  = {}
        self._verb_seen     = set()
        self._dirs          = None
        self._dirty         = True
        # Verb-rotor quantity axis
        self.q_omega = float(q_omega)
        rng_axis = np.random.default_rng(q_seed)
        self.q_axis = (rng_axis.integers(0, 2, size=self.d) * 2 - 1).astype(np.float32)

    # ── routing ───────────────────────────────────────────────────────────
    def _bank(self, role):
        return self.sdm if role in self.DENSE_ROLES else self.sdm_rel

    def _bind(self, a, r):
        return (a * r).astype(np.complex64)

    def _unbind(self, c, r):
        return (c * np.conj(r)).astype(np.complex64)

    def _addr(self, item, role):
        return self._bind(self.ck.key(item), self.roles[role])

    def _slot(self, item, role):
        return f'{item}\x00{role}'

    # ── writing ──────────────────────────────────────────────────────────
    def write_relation(self, item, role, value_hv):
        """Store item-role -> arbitrary HV value."""
        self._bank(role).write(
            self._addr(item, role),
            np.asarray(value_hv, dtype=np.complex64),
            word=self._slot(item, role),
        )
        self._seen.add(item)

    def relate(self, item, role, target):
        """Store item-role -> target (string). Tracks targets for cleanup."""
        self.write_relation(item, role, self.ck.key(target))
        self._role_targets.setdefault(role, set()).add(target)

    def assert_relation(self, item, role, target, n=20):
        """Reinforced injection: writes n times. Use for clean fact injection."""
        for _ in range(n):
            self.relate(item, role, target)

    def targets(self, role):
        return self._role_targets.get(role, set())

    # ── Channel 1: co-occurrence ─────────────────────────────────────────
    def expose_cooccur(self, text):
        """Sliding-window co-occurrence write. Returns token count."""
        tokens = tokenize(text)
        if not tokens:
            return 0
        n = len(tokens)
        K = np.stack([self.ck.key(t) for t in tokens])
        w = self.window
        P = np.empty((n + 1, self.d), dtype=np.complex128)
        P[0] = 0
        P[1:] = np.cumsum(K.astype(np.complex128), axis=0)
        agg, order = {}, []
        for i in range(n):
            lo = i - w if i - w > 0 else 0
            hi = i + w + 1 if i + w + 1 < n else n
            ctx = (P[hi] - P[lo]) - K[i]
            t = tokens[i]
            if t in agg:
                agg[t] = agg[t] + ctx
            else:
                agg[t] = ctx; order.append(t)
                self._seen.add(t); self._cooccur_seen.add(t)
        ukeys = np.stack([self._bind(self.ck.key(t), self.roles['cooccur']) for t in order])
        slots = [self._slot(t, 'cooccur') for t in order]
        locs = self.sdm.locs_batch(ukeys, slots)
        for t, idx in zip(order, locs):
            self.sdm.C[idx] += agg[t].astype(np.complex64)
        self._dirty = True
        return n

    # ── Channel 2: verb rotor (scalar arithmetic) ────────────────────────
    def _encode_scalar(self, c):
        phase = float(c) * self.q_omega * self.q_axis
        return np.exp(1j * phase).astype(np.complex64)

    def _decode_scalar(self, hv):
        phases = np.angle(hv).astype(np.float32)
        cs = phases / (self.q_omega * self.q_axis + 1e-9)
        return float(np.median(cs))

    def expose_verb_observation(self, verb, c_est):
        """Channel 2: encode scalar coefficient as phasor, superpose in (verb,'verb')."""
        self.write_relation(verb, 'verb', self._encode_scalar(c_est))
        self._verb_seen.add(verb)

    def predict_verb_coefficient(self, verb):
        """Decode running-average coefficient. None if unseen."""
        if verb not in self._verb_seen:
            return None
        return self._decode_scalar(self.recall(verb, 'verb'))

    # ── Channel 5: transitions / generation ──────────────────────────────
    def expose_transitions(self, text):
        """Per-token (prev → curr) write under 'next' role."""
        tokens = tokenize(text)
        if len(tokens) < 2:
            return 0
        agg, order = {}, []
        for i in range(len(tokens) - 1):
            prev, curr = tokens[i], tokens[i + 1]
            ck_curr = self.ck.key(curr)
            if prev in agg:
                agg[prev] = agg[prev] + ck_curr
            else:
                agg[prev] = ck_curr.astype(np.complex64).copy(); order.append(prev)
                self._seen.add(prev)
        ukeys = np.stack([self._bind(self.ck.key(p), self.roles['next']) for p in order])
        slots = [self._slot(p, 'next') for p in order]
        locs = self._bank('next').locs_batch(ukeys, slots)
        for p, idx in zip(order, locs):
            self._bank('next').C[idx] += agg[p]
            self._role_targets.setdefault('next', set()).add(p)
        return len(tokens) - 1

    def next_word_candidates(self, prev_word, candidates=None, top_k=20):
        """Vectorized bigram-equivalent: recall + score candidate keys."""
        if candidates is None:
            candidates = self._cooccur_seen
        if not candidates:
            return []
        r = self.recall(prev_word, 'next')
        cands = list(candidates)
        K = np.stack([self.ck.key(c) for c in cands])
        sims = (np.conj(r) @ K.T).real / self.d
        if top_k >= len(cands):
            order = np.argsort(-sims)
        else:
            top = np.argpartition(-sims, top_k)[:top_k]
            order = top[np.argsort(-sims[top])]
        return [(cands[i], float(sims[i])) for i in order]

    # ── reading ──────────────────────────────────────────────────────────
    def recall(self, item, role):
        return self._bank(role).read(self._addr(item, role),
                                     word=self._slot(item, role))

    def query(self, item, role, candidates=None):
        """Recall + cleanup against candidates. Returns (best, score)."""
        if candidates is None:
            candidates = self.targets(role)
        r = self.recall(item, role)
        best, bscore = None, -9.0
        for c in candidates:
            s = cos(r, self.ck.key(c), self.d)
            if s > bscore:
                bscore, best = s, c
        return best, bscore

    def chain(self, item, role_a, cands_a, role_b, cands_b):
        """Two-hop: role_b(role_a(item)). Returns (mid, end)."""
        mid, _ = self.query(item, role_a, cands_a)
        end, _ = self.query(mid, role_b, cands_b)
        return mid, end

    # ── cooccur similarity (Channel 1) with mean-removal ─────────────────
    def _refresh_dirs(self):
        if not self._dirty and self._dirs is not None:
            return
        if self.remove_r <= 0 or not self._cooccur_seen:
            self._dirs = np.zeros((0, self.d), dtype=np.complex64)
            self._dirty = False
            return
        words = list(self._cooccur_seen)
        if len(words) > self.svd_sample:
            rng = np.random.default_rng(0)
            words = [words[i] for i in rng.choice(len(words), self.svd_sample,
                                                  replace=False)]
        Mtx = np.stack([self.recall(w, 'cooccur') for w in words])
        _, _, Vh = np.linalg.svd(Mtx, full_matrices=False)
        self._dirs = Vh[:self.remove_r].astype(np.complex64)
        self._dirty = False

    def cooccur_recall(self, word):
        self._refresh_dirs()
        m = self.recall(word, 'cooccur')
        for v in self._dirs:
            m = m - np.vdot(v, m) * v
        return renorm(m)

    def similarity(self, w1, w2):
        if w1 not in self._cooccur_seen or w2 not in self._cooccur_seen:
            return None
        return cos(self.cooccur_recall(w1), self.cooccur_recall(w2), self.d)

    def neighbors(self, word, k=10, candidates=None):
        """Nearest words by cooccur similarity, against optional candidate pool."""
        if word not in self._cooccur_seen:
            return []
        pool = list(candidates) if candidates is not None else list(self._cooccur_seen)
        target = self.cooccur_recall(word)
        out = [(w, cos(target, self.cooccur_recall(w), self.d))
               for w in pool if w != word]
        return sorted(out, key=lambda x: -x[1])[:k]

    def consolidate(self):
        """Consolidate both banks. Marks dirty so common-directions rebuild."""
        self.sdm.consolidate()
        self.sdm_rel.consolidate()
        self._dirty = True

    # ── introspection ────────────────────────────────────────────────────
    def role_orthogonality(self, item, role_a, role_b):
        a = self._addr(item, role_a)
        b = self._addr(item, role_b)
        return cos(a, b, self.d)

    def substrate_bytes(self):
        return self.sdm.substrate_bytes() + self.sdm_rel.substrate_bytes()

    @property
    def vocab_size(self):
        return len(self._seen)

    def status(self):
        return {
            'roles':         list(self.roles.keys()),
            'vocab':         len(self._seen),
            'cooccur_vocab': len(self._cooccur_seen),
            'verb_vocab':    len(self._verb_seen),
            'dense_mb':      round(self.sdm.substrate_bytes() / 1_048_576, 1),
            'sparse_mb':     round(self.sdm_rel.substrate_bytes() / 1_048_576, 1),
            'substrate_mb':  round(self.substrate_bytes() / 1_048_576, 1),
            'flat':          True,
        }
