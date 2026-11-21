"""Dataset overview and pipeline figures for the report, README and poster.

Usage: python make_data_figures.py data/rul_steps.csv figures
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

csv, out = sys.argv[1], sys.argv[2]
poster_out = sys.argv[3] if len(sys.argv) > 3 else None
steps = pd.read_csv(csv)
cells = steps.drop_duplicates('cell').copy()
groups = cells.groupby('group')[['c_chg', 'c_dis', 'dod']].median()
life = cells['efc_total']


def conditions_panel(ax, fig, s=22, label_size=None):
    pts = ax.scatter(groups['c_chg'], groups['c_dis'], c=groups['dod'], cmap='viridis', vmin=0, vmax=1,
                     s=s, edgecolors='white', linewidths=0.5)
    ax.set_xlabel('Charge rate (C)')
    ax.set_ylabel('Discharge rate (C)')
    ax.set_xlim(0.3, 3.2); ax.set_ylim(0.3, 2.65)
    cb = fig.colorbar(pts, ax=ax, pad=0.02)
    cb.set_label('Depth of discharge', fontsize=label_size)
    return pts


def lifetime_dod_panel(ax, s=5):
    ax.scatter(cells['dod'] * 100, life, s=s, color='C0', alpha=0.6, edgecolors='none')
    ax.set_xlabel('Depth of discharge (%)')
    ax.set_ylabel('Lifetime (EFC)')
    ax.set_xlim(0, 102); ax.set_ylim(0, 2050)


# ---------------------------------------------------------------- dataset overview (report, README)
plt.rcParams.update({'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7})
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25), gridspec_kw={'width_ratios': [1.2, 1, 1]})
conditions_panel(axes[0], fig)
axes[0].set_title('(a) Usage conditions of the 63 groups')

ax = axes[1]
ax.hist(life, bins=np.arange(0, 2001, 100), color='C0', edgecolor='white', linewidth=0.5)
ax.axvline(life.median(), color='k', ls='--', lw=0.8)
ax.text(life.median() + 40, 33, f'median {life.median():.0f}', fontsize=7)
ax.set_xlabel('Lifetime (EFC to 80% capacity)')
ax.set_ylabel('Number of cells')
ax.set_title('(b) Lifetime of the 245 cells')

lifetime_dod_panel(axes[2])
axes[2].set_title('(c) Lifetime against depth of discharge')
fig.tight_layout(w_pad=0.8)
fig.savefig(f'{out}/dataset_overview.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------- pipeline (report, README)
ROWS = [
    ('Prepare the data', [
        ('1  Raw records', '251 cells, two JSON files\neach (RPT + cycling),\nabout 11 GB'),
        ('2  Per-test table', 'capacity, EFC so far, RUL,\n3 usage conditions;\n3,956 tests of 245 cells'),
        ('3  Features', '+ capacity lost\n+ recent fade slope\n7 inputs, target log RUL'),
        ('4  Sequences', 'one sequence per cell,\npadded to 59 tests:\ntensor 245 x 59 x 7'),
    ]),
    ('Train and test on unseen groups', [
        ('5  Group split', '5-fold GroupKFold;\neach fold tests on\n12-13 unseen groups'),
        ('6  Train', '3 LSTM variants,\nquantile LSTM, 2 RFs;\n5 seeds, early stopping'),
        ('7  Evaluate', 'MAE, RMSE, median error;\nerror by life stage;\ninterval coverage'),
        ('8  Predict a new cell', 'weekly capacities\n+ usage conditions\n= RUL and 10-90% range'),
    ]),
]
W, H = 7.2, 2.62
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
bw, bh, gap, x0 = 1.56, 0.86, 0.24, 0.1
row_y = [1.5, 0.08]
for r, (row_title, boxes) in enumerate(ROWS):
    y = row_y[r]
    right = x0 + 3 * (bw + gap) + bw
    ax.text(x0 if r == 0 else right, y + bh + 0.08, row_title, fontsize=8, fontweight='bold', color='#444444',
            va='bottom', ha='left' if r == 0 else 'right')
    for i, (title, body) in enumerate(boxes):
        x = x0 + i * (bw + gap)
        face = '#FDF4D4' if r == 1 else '#F3F3F3'
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle='round,pad=0,rounding_size=0.05',
                                    fc=face, ec='#9A9A9A', lw=0.7))
        ax.text(x + bw / 2, y + bh - 0.14, title, ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(x + bw / 2, y + bh / 2 - 0.1, body, ha='center', va='center', fontsize=7, linespacing=1.3)
        if i < len(boxes) - 1:
            ax.annotate('', xy=(x + bw + gap - 0.01, y + bh / 2), xytext=(x + bw + 0.01, y + bh / 2),
                        arrowprops=dict(arrowstyle='-|>', color='#555555', lw=0.8, mutation_scale=7))
# elbow connector from box 4 down to box 5
x4 = x0 + 3 * (bw + gap) + bw / 2
x5 = x0 + bw / 2
y_mid = row_y[0] - 0.17
ax.plot([x4, x4, x5], [row_y[0], y_mid, y_mid], color='#555555', lw=0.8)
ax.annotate('', xy=(x5, row_y[1] + bh + 0.01), xytext=(x5, y_mid),
            arrowprops=dict(arrowstyle='-|>', color='#555555', lw=0.8, mutation_scale=7, shrinkA=0))
fig.savefig(f'{out}/pipeline.png', dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- poster version: conditions + lifetime
if poster_out:
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
                         'xtick.labelsize': 10, 'ytick.labelsize': 10})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw={'width_ratios': [1.2, 1]})
    conditions_panel(axes[0], fig, s=45)
    axes[0].set_title('Usage conditions of the 63 groups')
    lifetime_dod_panel(axes[1], s=10)
    axes[1].set_title('Lifetime of the 245 cells')
    fig.tight_layout(w_pad=1.2)
    fig.savefig(f'{poster_out}/conditions_lifetime.png', dpi=250, bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------- numbers quoted in the text
by_group = cells.groupby('group')
ratio = by_group['efc_total'].agg(lambda x: x.max() / x.min() if len(x) > 1 else np.nan)
bins = pd.cut(cells['dod'], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
print('cells per group:', by_group.size().value_counts().to_dict())
print('median life by DoD bin:', cells.groupby(bins, observed=True)['efc_total'].agg(['size', 'median']).round(0).to_dict())
print('within-group longest/shortest: median %.2f, max %.2f' % (ratio.median(), ratio.max()))
print('lifetime range: %.0f-%.0f EFC, ratio %.1f' % (life.min(), life.max(), life.max() / life.min()))
