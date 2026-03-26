from pathlib import Path
import importlib.util

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('search_sim', str(BASE / 'search_sim.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

res = mod.run_pattern('expanding_square', 3, steps=300, seed=123)
print('fires_count reported:', res['fires_count'])
print('squares_viewed reported:', res['squares_viewed'])
print('first_detect:', res.get('first_detect_step', None))
print('fire_centers:')
for i, fc in enumerate(res['fire_centers']):
    print(i, fc)
# access env to inspect fires_visited per drone
env = res['env']
for i in range(env.n_drones):
    print(f'drone {i} fires_visited indices:', env.fires_visited[i])
# union
union = set().union(*env.fires_visited)
print('union indices:', union, 'len:', len(union))
# also print a quick sample of seen grid sizes
# we can't access seen_grids here; run_pattern returns squares_viewed only
print('Done')
