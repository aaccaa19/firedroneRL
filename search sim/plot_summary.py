"""
Read `search_summary.csv` and create two easy-to-understand summary graphs:
- Heatmap of `fires_found` (scenarios x patterns)
- Heatmap of `squares_viewed` (scenarios x patterns)
Saves images into the `search sim` folder.
Run: python "search sim/plot_summary.py"
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

CSV = Path('search sim') / 'search_summary.csv'
OUT_DIR = Path('search sim')

patterns = []
scenarios = []
rows = []
with open(CSV, newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        pat = row['pattern']
        sc = int(row['scenario'])
        if pat not in patterns:
            patterns.append(pat)
        if sc not in scenarios:
            scenarios.append(sc)
        rows.append(row)

patterns = sorted(patterns)
scenarios = sorted(scenarios)
# build matrices
fires = np.zeros((len(scenarios), len(patterns)), dtype=float)
squares = np.zeros_like(fires)
first_detect = np.full_like(fires, np.nan)

for row in rows:
    sc = int(row['scenario'])
    pat = row['pattern']
    si = scenarios.index(sc)
    pi = patterns.index(pat)
    fires[si, pi] = float(row['fires_found']) if row['fires_found'] != '' else 0.0
    squares[si, pi] = float(row['squares_viewed']) if row['squares_viewed'] != '' else 0.0
    fd = row.get('first_detect_step', '')
    first_detect[si, pi] = float(fd) if fd != '' else np.nan

# Plot heatmap for fires
fig, ax = plt.subplots(figsize=(10,4))
im = ax.imshow(fires, cmap='OrRd', vmin=0, vmax=np.nanmax(fires)+0.1)
ax.set_xticks(np.arange(len(patterns)))
ax.set_yticks(np.arange(len(scenarios)))
ax.set_xticklabels(patterns, rotation=45, ha='right')
ax.set_yticklabels([str(s) for s in scenarios])
for i in range(len(scenarios)):
    for j in range(len(patterns)):
        text = int(fires[i,j])
        ax.text(j, i, text, ha='center', va='center', color='black')
ax.set_title('Fires found (scenarios rows, patterns columns)')
fig.colorbar(im, ax=ax, orientation='vertical', label='fires found')
fig.tight_layout()
fig.savefig(OUT_DIR / 'summary_fires_heatmap.png', dpi=150)
plt.close(fig)

# Plot heatmap for squares viewed
fig, ax = plt.subplots(figsize=(10,4))
im = ax.imshow(squares, cmap='Blues')
ax.set_xticks(np.arange(len(patterns)))
ax.set_yticks(np.arange(len(scenarios)))
ax.set_xticklabels(patterns, rotation=45, ha='right')
ax.set_yticklabels([str(s) for s in scenarios])
for i in range(len(scenarios)):
    for j in range(len(patterns)):
        text = int(squares[i,j])
        ax.text(j, i, text, ha='center', va='center', color='black', fontsize=8)
ax.set_title('Unique grid squares viewed (300 steps)')
fig.colorbar(im, ax=ax, orientation='vertical', label='squares viewed')
fig.tight_layout()
fig.savefig(OUT_DIR / 'summary_squares_heatmap.png', dpi=150)
plt.close(fig)

# Also save a small CSV with first-detect info
out_fd = OUT_DIR / 'first_detect_matrix.csv'
with open(out_fd, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['scenario'] + patterns)
    for i, sc in enumerate(scenarios):
        row = [sc] + [ ('' if np.isnan(first_detect[i,j]) else int(first_detect[i,j])) for j in range(len(patterns))]
        w.writerow(row)

print('Saved summary_fires_heatmap.png, summary_squares_heatmap.png, and first_detect_matrix.csv in search sim folder')
