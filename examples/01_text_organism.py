"""
Example 1: text organism (Ikigai-style mini).

Reads short sentences. Builds co-occurrence, asserts IS-A facts,
answers similarity queries. Substrate stays fixed.

Run: python examples/01_text_organism.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from flatmem import MultiRoleMemory

mem = MultiRoleMemory(d=512, M=4096, k=32, seed=42, M_rel=2048)

corpus = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "the boy played with the dog",
    "the girl fed the cat",
    "the king sat on his throne",
    "the queen wore a crown",
    "cats and dogs are pets",
] * 30

print('Reading corpus...')
for s in corpus:
    mem.expose_cooccur(s)

print('Asserting IS-A facts...')
mem.assert_relation('cat', 'isa', 'mammal', n=30)
mem.assert_relation('dog', 'isa', 'mammal', n=30)
mem.assert_relation('king', 'isa', 'person', n=30)
mem.assert_relation('queen', 'isa', 'person', n=30)

print('\nSimilarity:')
for a, b in [('cat', 'dog'), ('king', 'queen'), ('cat', 'king')]:
    print(f'  {a} ~ {b} = {mem.similarity(a, b):+.3f}')

print('\nIS-A queries:')
for w in ['cat', 'dog', 'king', 'queen']:
    print(f'  isa({w}) = {mem.query(w, "isa")[0]}')

print(f'\nSubstrate: {mem.substrate_bytes() / 1_048_576:.0f} MB FIXED')
print(f'Vocab seen: {len(mem._cooccur_seen)}')
