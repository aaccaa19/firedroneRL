import csv
from pathlib import Path
import importlib.util

BASE = Path(__file__).resolve().parent
module_path = BASE / 'search_sim.py'
spec = importlib.util.spec_from_file_location('search_sim_module', str(module_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

csv_path = BASE / 'search_summary.csv'
if not csv_path.exists():
    print('No CSV found at', csv_path)
    raise SystemExit(1)

# Read CSV into summary dict
summary = {}
scenarios = []
patterns = []
with open(csv_path, 'r', newline='') as f:
    r = csv.reader(f)
    header = next(r)
    for row in r:
        s = int(row[0])
        p = row[1]
        fires = int(row[2]) if row[2] != '' else 0
        squares = int(row[3]) if row[3] != '' else 0
        fd = row[4]
        plot = row[5] if len(row) > 5 else ''
        if s not in summary:
            summary[s] = {}
            scenarios.append(s)
        summary[s][p] = {'fires': fires, 'squares': squares, 'first_detect': fd, 'plot': plot}
        if p not in patterns:
            patterns.append(p)

scenarios = sorted(scenarios)
print('Regenerating summary images for scenarios', scenarios, 'patterns', patterns)
try:
    f1, f2, fd_csv = mod.generate_summary_images(summary, scenarios, patterns, BASE)
    print('Generated:', f1, f2, fd_csv)
except Exception as e:
    print('Failed to generate summaries:', e)
    raise
