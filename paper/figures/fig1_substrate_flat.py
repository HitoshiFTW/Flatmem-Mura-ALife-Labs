"""
Figure 1: substrate-flat vs dict-grows.

Data from Pack 114, 115, 125 (substrate stays 192 MB across 0 to 100K stories;
hypothetical dict-equivalent grows with vocab).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Empirical from Pack 125 (TinyStories 100K run):
# (stories, vocab) at each checkpoint
data = [
    (0,       0),
    (10_000,  9841),    # from Pack 123
    (25_000, 12658),    # Pack 125
    (50_000, 15493),    # Pack 125
    (75_000, 17648),    # Pack 125
    (100_000, 19411),   # Pack 125
]
stories = [s for s, _ in data]
vocab   = [v for _, v in data]

# d=512, complex64 = 8 bytes per component -> 4096 bytes per word
DICT_BYTES_PER_WORD_D512  = 512 * 8
# d=2048 (the ORIGINAL IkigaiBeing): 16384 bytes per word
DICT_BYTES_PER_WORD_D2048 = 2048 * 8
SUBSTRATE_MB = 192   # fixed

dict_mb_d512  = [v * DICT_BYTES_PER_WORD_D512  / 1_048_576 for v in vocab]
dict_mb_d2048 = [v * DICT_BYTES_PER_WORD_D2048 / 1_048_576 for v in vocab]
substrate_mb  = [SUBSTRATE_MB] * len(stories)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(stories, substrate_mb,  'o-', color='#2c7bb6', linewidth=2.5,
        markersize=8, label='flatmem substrate (FIXED)')
ax.plot(stories, dict_mb_d512,  's--', color='#fdae61', linewidth=2,
        markersize=7, label=r'dict equivalent @ $d=512$')
ax.plot(stories, dict_mb_d2048, '^--', color='#d7191c', linewidth=2,
        markersize=7, label=r'dict equivalent @ $d=2048$ (original IkigaiBeing)')
ax.set_xlabel('Stories read', fontsize=11)
ax.set_ylabel('Memory footprint (MB)', fontsize=11)
ax.set_title('Substrate is flat; dict grows with vocabulary', fontsize=12)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig('fig1_substrate_flat.pdf', bbox_inches='tight')
plt.savefig('fig1_substrate_flat.png', bbox_inches='tight', dpi=150)
print('saved fig1_substrate_flat.pdf / .png')
