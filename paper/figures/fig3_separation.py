"""
Figure 3: Mean-removal sharpens discrimination (Pack 115 + 125).

Aggregate separation (mean_related - mean_unrelated) under different r values
(common-direction removal rank). r=0 baseline vs r=1 sharpened; persistent
across scale.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Pack 115 (15K stories, dual substrate RAW vs IDF):
# r-sweep on RAW substrate
r_values = [0, 1, 2, 3, 5, 10]
raw_sep  = [0.026, 0.371, 0.316, 0.233, 0.226, 0.252]

# Pack 125 (100K stories, single substrate, r=1 only at checkpoints):
ck_stories = [25_000, 50_000, 75_000, 100_000]
ck_sep_r1  = [0.163, 0.150, 0.151, 0.147]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# left: r-sweep at 15K
ax1.plot(r_values, raw_sep, 'o-', color='#2c7bb6', linewidth=2, markersize=8)
ax1.axhline(y=raw_sep[0], color='#d7191c', linestyle='--', alpha=0.6,
            label=f'r=0 baseline: {raw_sep[0]:.3f}')
ax1.set_xlabel('Common-direction removal rank (r)', fontsize=11)
ax1.set_ylabel('Separation = mean(related) - mean(unrelated)', fontsize=11)
ax1.set_title('Mean-removal sweep at 15K stories', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# right: sep vs scale
ax2.plot(ck_stories, ck_sep_r1, 's-', color='#1a9850', linewidth=2, markersize=8)
ax2.set_xlabel('Stories read', fontsize=11)
ax2.set_ylabel('Separation (r=1)', fontsize=11)
ax2.set_title('Discrimination holds at scale (r=1)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 0.20)

plt.tight_layout()
plt.savefig('fig3_separation.pdf', bbox_inches='tight')
plt.savefig('fig3_separation.png', bbox_inches='tight', dpi=150)
print('saved fig3_separation.pdf / .png')
