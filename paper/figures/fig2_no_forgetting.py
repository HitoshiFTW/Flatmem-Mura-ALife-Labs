"""
Figure 2: No-forgetting under cross-modal distractors (Pack 129).

5 original IS-A facts retained across 4 waves of distractor writes spanning
different modalities. Bar chart.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

phases = [
    'baseline',
    '+20 IS-A\ndistractors',
    '+5K text\ncooccur',
    '+100 verb\nobservations',
    '+100 vision\nclassifications',
]
originals_kept    = [5, 5, 5, 5, 5]
distractor_facts  = [0, 20, 20, 20, 20]
distractor_kept   = [0, 20, 20, 20, 20]

# What a transformer trained sequentially without replay would do (illustrative,
# approximate — based on catastrophic forgetting literature)
transformer_kept = [5, 3, 1, 0, 0]

x = np.arange(len(phases))
width = 0.35

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.bar(x - width/2, originals_kept,    width,
       color='#2c7bb6', label='flatmem: original 5 retained')
ax.bar(x + width/2, transformer_kept,  width,
       color='#d7191c', alpha=0.7,
       label='naive sequential transformer (illustrative)')
ax.set_xticks(x)
ax.set_xticklabels(phases, fontsize=9)
ax.set_ylabel('Original 5 facts retained (of 5)', fontsize=11)
ax.set_title('No-forgetting under cross-modal distractor floods', fontsize=12)
ax.set_ylim(0, 5.5)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fig2_no_forgetting.pdf', bbox_inches='tight')
plt.savefig('fig2_no_forgetting.png', bbox_inches='tight', dpi=150)
print('saved fig2_no_forgetting.pdf / .png')
