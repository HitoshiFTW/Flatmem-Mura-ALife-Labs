"""
Plot continual-learning forgetting matrix from bench results JSON.

Usage: python -m bench.plot results.json
"""
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_forgetting_matrix(results, out_path='forgetting.pdf'):
    """Plot two side-by-side heatmaps showing accuracy[i,j] for each adapter."""
    n_adapters = len(results)
    fig, axes = plt.subplots(1, n_adapters, figsize=(5 * n_adapters, 4.5))
    if n_adapters == 1: axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        M = np.array(r['acc_matrix'])
        n = M.shape[0]
        im = ax.imshow(M, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{M[i, j]:.0%}', ha='center', va='center',
                        fontsize=9, color='black')
        ax.set_xticks(range(n)); ax.set_xticklabels([f'T{j}' for j in range(n)])
        ax.set_yticks(range(n)); ax.set_yticklabels([f'after T{i}' for i in range(n)])
        ax.set_title(f'{name}\nfinal avg {r["final_avg"]:.0%}  BWT {r["bwt"]:+.0%}',
                     fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    print(f'saved {out_path} + .png')


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'permuted_digits_results.json'
    with open(path) as f: results = json.load(f)
    plot_forgetting_matrix(results, out_path='forgetting_matrix.pdf')
