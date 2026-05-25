"""
Figure 4: Vision channel results (Pack 127 / 128).

8x8 digit classification through the same substrate that hosts text channels.
Train/test/save-load accuracy bar chart.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

conditions = ['Train\n(1257 imgs)', 'Test\n(540 imgs)', 'Test after\nsave/load\n(50 imgs)']
accuracies = [100, 90, 86]
colors     = ['#2c7bb6', '#1a9850', '#fdae61']

fig, ax = plt.subplots(figsize=(6.5, 4.5))
bars = ax.bar(conditions, accuracies, color=colors, edgecolor='black')
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{acc}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Classification accuracy (%)', fontsize=11)
ax.set_title('Vision channel: 8x8 digits via random projection encoder\n'
             '(same 192 MB substrate also holds text, IS-A, math, sensory)',
             fontsize=11)
ax.set_ylim(0, 110)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=10, color='gray', linestyle=':', alpha=0.5, label='random chance (10%)')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('fig4_vision_results.pdf', bbox_inches='tight')
plt.savefig('fig4_vision_results.png', bbox_inches='tight', dpi=150)
print('saved fig4_vision_results.pdf / .png')
