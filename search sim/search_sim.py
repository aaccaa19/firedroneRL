"""
Copied and adapted search_sim to live inside the `search sim` folder.
This copy updates output paths so plots and summary CSV are written to this folder.
Run: python "search sim/search_sim.py"
"""
import os
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt

# Lightweight local environment that reproduces the detection / view logic
# from the project's DroneEnv but avoids importing torch and gym.
class LocalEnv:
    def __init__(self, curriculum_level=0, scenario=1):
        self.x_max = 20
        self.y_max = 10
        self.z_min = 5
        self.z_max = 10
        self.n_drones = 2
        self.curriculum_level = curriculum_level
        self.scenario = scenario
        self.k = 0.5
        self.grid_size = 0.5
        self.grid_dim = int(self.x_max // self.grid_size)
        self.max_visible_fires = 12
        self.drone_radius = 0.25
        self.base_safety_margin = 0.1
        self.safety_margin = self.base_safety_margin + max(0, (5 - curriculum_level) * 0.1)
        self.max_step_size = 0.5
        # default fire_line same as original
        self.fire_line = np.array(((10,5,0),(10,5,20)))
        # scenario-dependent initialization
        if scenario == 1:
            # By default the original code used a fire_line; we will allow callers to override
            self.fire_centers = []
            self.fire_radius = 0.5
        elif scenario == 2:
            self.fire_centers = [np.array([10.0, 5.0, 5.0])]
            self.fire_radius = 0.25
            self.max_fires = 8
        elif scenario == 3:
            n_fires = 8
            y_positions = np.linspace(1, self.y_max-1, n_fires)
            x_pos = 10.0
            z_pos = 5.0
            self.fire_centers = [np.array([x_pos, y, z_pos]) for y in y_positions]
            self.fire_radius = 0.25
        elif scenario == 4:
            self.fire_centers = [
                np.array([10.0, 5.0, 5.0]),
                np.array([12.0, 3.0, 5.0]),
                np.array([8.0, 6.0, 5.0]),
                np.array([12.0, 6.0, 5.0]),
                np.array([8.0, 3.0, 5.0])
            ]
            self.fire_radius = 0.25
        elif scenario == 5:
            self.fire_wall_x = 10.0
            self.fire_centers = []
            self.fire_radius = 0.25
        elif scenario == 6:
            center = np.array([10.0, 5.0, (self.z_max + self.z_min) / 2])
            self.fire_centers = [center]
            self.fire_radius = 2.5
        # runtime state
        self.drone_pos = np.zeros((self.n_drones, 3))
        self.prev_positions = [None for _ in range(self.n_drones)]
        self.fires_visited = [set() for _ in range(self.n_drones)]
        # cumulative visible fires (indices) seen in view (not necessarily 'discovered')
        self.visible_fires = set()

    def reset(self):
        # deterministic starting positions (edges)
        self.drone_pos = np.array([[0.5, self.y_max/2.0, (self.z_min + self.z_max)/2.0],
                                   [self.x_max - 0.5, self.y_max/2.0, (self.z_min + self.z_max)/2.0]])
        self.prev_positions = [p.copy() for p in self.drone_pos]
        self.fires_visited = [set() for _ in range(self.n_drones)]
        return None, {}

    def view_radius(self, z):
        return self.k * z / 8.0

    def is_occluded(self, drone_xy, target_xy, fire_centers_2d, fire_radius):
        # replicate occlusion check from original code (2D)
        v_target = target_xy - drone_xy
        dist_target = np.linalg.norm(v_target)
        for fb in fire_centers_2d:
            v_fire = fb - drone_xy
            dist_fire = np.linalg.norm(v_fire)
            if dist_fire + 1e-6 < dist_target:
                if dist_fire > 1e-6 and dist_target > 1e-6:
                    cos_angle = np.dot(v_fire, v_target) / (dist_fire * dist_target + 1e-8)
                    if cos_angle > 0.99:
                        proj = np.dot(v_fire, v_target) / (dist_target + 1e-8)
                        closest = drone_xy + (v_target / (dist_target + 1e-8)) * proj
                        if np.linalg.norm(fb - closest) <= (fire_radius + 1e-6):
                            return True
        return False

    def step(self, actions, seen_grids=None):
        # move drones
        for i in range(self.n_drones):
            a = np.array(actions[i], dtype=float)
            # normalize action to unit then scale by max_step_size as original
            norm = np.linalg.norm(a)
            if norm > 1e-8:
                a = a / norm * min(norm, 1.0)
            a = a * self.max_step_size
            self.drone_pos[i] = np.clip(self.drone_pos[i] + a, [0,0,self.z_min], [self.x_max, self.y_max, self.z_max])

        # compute visible fires and seen grid
        fires_seen_union = set()
        fire_centers = self.fire_centers if getattr(self, 'fire_centers', None) else []
        fire_centers_2d = np.array([[fc[0], fc[1]] for fc in fire_centers]) if len(fire_centers) > 0 else np.empty((0,2))

        for i in range(self.n_drones):
            pos = self.drone_pos[i]
            drone_xy = np.array([pos[0], pos[1]])
            z = pos[2]
            view_r = self.view_radius(z)
            for idx, fc in enumerate(fire_centers):
                fc_xy = np.array([fc[0], fc[1]])
                dist = np.linalg.norm(fc_xy - drone_xy)
                if dist <= (view_r + self.fire_radius):
                    # occlusion by other fires
                    blocked = False
                    for jdx, fb in enumerate(fire_centers):
                        if jdx == idx:
                            continue
                        fb_xy = np.array([fb[0], fb[1]])
                        v_fire = fb_xy - drone_xy
                        dist_fire = np.linalg.norm(v_fire)
                        if dist_fire + 1e-6 < dist:
                            if dist_fire > 1e-6 and dist > 1e-6:
                                cos_angle = np.dot(v_fire, (fc_xy - drone_xy)) / (dist_fire * dist + 1e-8)
                                if cos_angle > 0.99:
                                    proj = np.dot(v_fire, (fc_xy - drone_xy)) / (dist + 1e-8)
                                    closest = drone_xy + ((fc_xy - drone_xy) / (dist + 1e-8)) * proj
                                    if np.linalg.norm(fb_xy - closest) <= (self.fire_radius + 1e-6):
                                        blocked = True
                                        break
                    if not blocked:
                        fires_seen_union.add(idx)

            # seen grid cells
            view_r = self.view_radius(pos[2])
            x_min = max(0, int((drone_xy[0] - view_r) // self.grid_size))
            x_max = min(self.grid_dim - 1, int((drone_xy[0] + view_r) // self.grid_size))
            y_min = max(0, int((drone_xy[1] - view_r) // self.grid_size))
            y_max = min(self.grid_dim - 1, int((drone_xy[1] + view_r) // self.grid_size))
            for gx in range(x_min, x_max+1):
                for gy in range(y_min, y_max+1):
                    cx = gx * self.grid_size + self.grid_size/2
                    cy = gy * self.grid_size + self.grid_size/2
                    cell_xy = np.array([cx, cy])
                    if np.linalg.norm(drone_xy - cell_xy) <= view_r:
                        occluded = False
                        if fire_centers_2d.shape[0] > 0:
                            for fc2d in fire_centers_2d:
                                v_fire = fc2d - drone_xy
                                v_cell = cell_xy - drone_xy
                                dist_to_fire = np.linalg.norm(v_fire)
                                dist_to_cell = np.linalg.norm(v_cell)
                                if dist_to_fire < dist_to_cell and dist_to_fire > 1e-6:
                                    cos_angle = np.dot(v_fire, v_cell) / (dist_to_fire * dist_to_cell + 1e-8)
                                    if cos_angle > 0.99:
                                        occluded = True
                                        break
                        if not occluded and seen_grids is not None:
                            seen_grids[i].add((int(gx), int(gy)))

        # discovery threshold similar to original code
        for i in range(self.n_drones):
            pos = self.drone_pos[i]
            for idx, fc in enumerate(fire_centers):
                dist_to_fire = np.linalg.norm(pos - fc)
                if dist_to_fire < (self.fire_radius + self.safety_margin + 1.0):
                    if idx not in self.fires_visited[i]:
                        self.fires_visited[i].add(idx)

        # record visible fires seen this step into cumulative set
        for idx in fires_seen_union:
            self.visible_fires.add(idx)

        # Scenario 2: spreading fire spawn (match original env behavior)
        if self.scenario == 2:
            if len(self.fire_centers) < getattr(self, 'max_fires', 8):
                for center in list(self.fire_centers):
                    if np.random.rand() < 0.02:
                        angle = np.random.uniform(0, 2 * np.pi)
                        height = np.random.uniform(-2, 2)
                        r = np.random.uniform(0.5, getattr(self, 'fire_spawn_radius', 5.0))
                        dx = r * np.cos(angle)
                        dy = r * np.sin(angle)
                        dz = height
                        new_center = center + np.array([dx, dy, dz])
                        new_center = np.clip(new_center, [0, 0, self.z_min], [self.x_max, self.y_max, self.z_max])
                        if all(np.linalg.norm(new_center - fc) > self.fire_radius * 1.5 for fc in self.fire_centers):
                            self.fire_centers.append(new_center)

        return None


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / 'plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Helper: generate waypoints for search patterns

def expanding_square(center, x_max, y_max, spacing=1.0, n_points=500):
    cx, cy = center
    points = []
    # spiral grid growing squares
    step = spacing
    layer = 0
    while len(points) < n_points:
        side = (layer * 2 + 1)
        # top row left->right
        x0 = cx - layer * step
        y0 = cy - layer * step
        for j in range(side):
            x = x0 + j * step
            y = y0
            if 0 <= x <= x_max and 0 <= y <= y_max:
                points.append((x, y))
        # right column top->bottom
        for i in range(1, side):
            x = x0 + (side - 1) * step
            y = y0 + i * step
            if 0 <= x <= x_max and 0 <= y <= y_max:
                points.append((x, y))
        # bottom row right->left
        for j in range(side - 2, -1, -1):
            x = x0 + j * step
            y = y0 + (side - 1) * step
            if 0 <= x <= x_max and 0 <= y <= y_max:
                points.append((x, y))
        # left column bottom->top
        for i in range(side - 2, 0, -1):
            x = x0
            y = y0 + i * step
            if 0 <= x <= x_max and 0 <= y <= y_max:
                points.append((x, y))
        layer += 1
        if layer > max(x_max, y_max) / spacing + 5:
            break
    return points


def sector_search(center, x_max, y_max, n_sectors=8, radius_step=1.0):
    cx, cy = center
    pts = []
    max_r = math.hypot(x_max, y_max)
    for s in range(n_sectors):
        ang = 2 * math.pi * s / n_sectors
        r = radius_step
        while r < max_r:
            x = cx + r * math.cos(ang)
            y = cy + r * math.sin(ang)
            if 0 <= x <= x_max and 0 <= y <= y_max:
                pts.append((x, y))
            r += radius_step
    return pts


def parallel_sweep(x_max, y_max, spacing=1.0):
    # sweep along x with horizontal tracks at different y
    pts = []
    y = 0.5
    direction = 1
    while y <= y_max:
        if direction == 1:
            xs = np.linspace(0.0, x_max, int(max(2, x_max / 0.5)))
        else:
            xs = np.linspace(x_max, 0.0, int(max(2, x_max / 0.5)))
        for x in xs:
            pts.append((float(x), float(y)))
        y += spacing
        direction *= -1
    return pts


def track_line(x_max, y_max, x_line=10.0, n_points=200):
    ys = np.linspace(0.5, y_max - 0.5, n_points)
    return [(x_line, float(y)) for y in ys]


def creeping_line(x_max, y_max, spacing=0.5, segment_length=3.0):
    pts = []
    x = 0.5
    direction = 1
    while x <= x_max:
        # many short segments in y
        y = 0.5 if direction == 1 else y_max - 0.5
        traveled = 0.0
        while traveled < y_max - 1.0:
            # advance by segment_length
            y_next = min(max(0.5, y + direction * segment_length), y_max - 0.5)
            ys = np.linspace(y, y_next, max(2, int(abs(y_next - y) / 0.25) + 1))
            for yy in ys:
                pts.append((float(x), float(yy)))
            traveled += abs(y_next - y)
            y = y_next
        x += spacing
        direction *= -1
    return pts


def box_area_search(x_max, y_max, nx=8, ny=4):
    pts = []
    xs = np.linspace(0.5, x_max - 0.5, nx)
    ys = np.linspace(0.5, y_max - 0.5, ny)
    for j, yy in enumerate(ys):
        if j % 2 == 0:
            for xx in xs:
                pts.append((float(xx), float(yy)))
        else:
            for xx in xs[::-1]:
                pts.append((float(xx), float(yy)))
    return pts


PATTERNS = {
    'expanding_square': expanding_square,
    'sector': sector_search,
    'parallel_sweep': parallel_sweep,
    'track_line': track_line,
    'creeping_line': creeping_line,
    'area_box': box_area_search,
}


def run_pattern(pattern_name, scenario, steps=300, seed=0):
    np.random.seed(seed)
    env = LocalEnv(curriculum_level=0, scenario=scenario)
    # We'll override scenario 1 to be a single fire (user requested)
    if scenario == 1:
        env.fire_centers = [np.array([10.0, 5.0, 5.0])]
        env.fire_radius = 0.5
    # Treat scenario 5 as a single fire for counting as requested
    if scenario == 5:
        env.fire_centers = [np.array([env.fire_line[0][0], env.y_max/2.0, 5.0])]
        # keep fire radius default
    obs, _ = env.reset()
    # set starting positions deterministically at edges
    env.drone_pos = np.array([[0.5, env.y_max/2.0, (env.z_min + env.z_max)/2.0],
                              [env.x_max - 0.5, env.y_max/2.0, (env.z_min + env.z_max)/2.0]])
    env.prev_positions = [p.copy() for p in env.drone_pos]

    # prepare seen grids containers (env.step can record into these)
    seen_grids = [set(), set()]

    # build waypoints for each drone based on pattern
    center = (env.x_max/2.0, env.y_max/2.0)
    kwargs = {'center': center, 'x_max': env.x_max, 'y_max': env.y_max}
    gen = PATTERNS[pattern_name]
    # generate a long list of candidate waypoints and split lanes between drones
    pts = gen(**{k: v for k, v in kwargs.items() if k in gen.__code__.co_varnames})
    if len(pts) == 0:
        pts = [(center[0], center[1])]
    # Partition waypoints by left/right halves so each drone starts on its side
    mid_x = env.x_max / 2.0
    pts_left = [p for p in pts if p[0] <= mid_x]
    pts_right = [p for p in pts if p[0] > mid_x]
    if len(pts_left) == 0 or len(pts_right) == 0:
        # fallback to interleaving if partitioning fails
        pts0 = pts[::2]
        pts1 = pts[1::2]
        if len(pts0) == 0:
            pts0 = pts
        if len(pts1) == 0:
            pts1 = pts
    else:
        pts0 = pts_left
        pts1 = pts_right

    # start each drone at the nearest waypoint on its side to avoid crossing sides
    start0 = env.drone_pos[0][:2]
    start1 = env.drone_pos[1][:2]
    def nearest_index(points, start):
        if len(points) == 0:
            return 0
        dists = [math.hypot(p[0]-start[0], p[1]-start[1]) for p in points]
        return int(np.argmin(dists))
    idx0 = nearest_index(pts0, start0)
    idx1 = nearest_index(pts1, start1)

    trajs = [[], []]

    for step in range(steps):
        actions = np.zeros((env.n_drones, 3), dtype=float)
        for i in range(env.n_drones):
            cur = env.drone_pos[i]
            if i == 0:
                waypoint = np.array([pts0[idx0 % len(pts0)][0], pts0[idx0 % len(pts0)][1], cur[2]])
            else:
                waypoint = np.array([pts1[idx1 % len(pts1)][0], pts1[idx1 % len(pts1)][1], cur[2]])
            delta = waypoint - cur
            dist = np.linalg.norm(delta)
            if dist < 0.25:
                if i == 0:
                    idx0 += 1
                else:
                    idx1 += 1
                actions[i] = np.array([0.0, 0.0, 0.0])
            else:
                # produce action that will move toward waypoint by at most max_step_size
                # action scale: unit action -> move env.max_step_size
                move = delta / env.max_step_size
                # clip magnitude to 1
                mag = np.linalg.norm(move)
                if mag > 1.0:
                    move = move / mag
                actions[i] = move
        obs = env.step(actions, seen_grids=seen_grids)
        for i in range(env.n_drones):
            trajs[i].append(env.drone_pos[i].copy())
        # check first-detect time (either discovered or seen in view)
        discovered_now = set().union(*getattr(env, 'fires_visited', [set(), set()])) if hasattr(env, 'fires_visited') else set()
        visible_now = getattr(env, 'visible_fires', set())
        all_seen = discovered_now.union(visible_now)
        if 'first_detect_step' not in locals() and len(all_seen) > 0:
            first_detect_step = step + 1
    # After run collect stats
    first_detect = locals().get('first_detect_step', None)
    # After run collect stats
    # Count fires that were either "discovered" (fires_visited) or seen in view (visible_fires)
    discovered = set().union(*getattr(env, 'fires_visited', [set(), set()])) if hasattr(env, 'fires_visited') else set()
    visible = getattr(env, 'visible_fires', set())
    fires_union = discovered.union(visible)
    fires_count = len(fires_union)
    # If scenario 1 originally used discretized fire_line, user requested single fire; in that case count is len(env.fire_centers)
    # But actual detection sets are in env.fires_visited (indexes)
    # Squares viewed: union of seen_grids
    squares_viewed = len(seen_grids[0].union(seen_grids[1])) if isinstance(seen_grids, (list, tuple)) else 0
    return {
        'fires_count': fires_count,
        'squares_viewed': squares_viewed,
        'trajs': trajs,
        'fire_centers': [fc.copy() for fc in env.fire_centers],
        'env': env,
        'first_detect_step': first_detect
    }


def plot_run(result, scenario, pattern_name):
    trajs = result['trajs']
    fire_centers = result['fire_centers']
    fig, ax = plt.subplots(figsize=(8,4))
    # plot fires
    if len(fire_centers) > 0:
        xs = [fc[0] for fc in fire_centers]
        ys = [fc[1] for fc in fire_centers]
        ax.scatter(xs, ys, c='red', s=80, label='fires')
    # plot drone trajectories
    colors = ['blue', 'green']
    for i in range(len(trajs)):
        arr = np.array(trajs[i])
        if arr.shape[0] > 0:
            # draw segments with increasing alpha so path becomes darker over time
            n = arr.shape[0]
            for j in range(n-1):
                a1 = arr[j]
                a2 = arr[j+1]
                alpha = float(j+1) / float(max(1, n-1))
                ax.plot([a1[0], a2[0]], [a1[1], a2[1]], c=colors[i], alpha=0.15 + 0.85*alpha, linewidth=1.5)
            ax.scatter(arr[0,0], arr[0,1], c=colors[i], marker='o')
            ax.scatter(arr[-1,0], arr[-1,1], c=colors[i], marker='x')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    fd = result.get('first_detect_step', None)
    fd_str = str(fd) if fd is not None else 'None'
    ax.set_title(f'scenario{scenario}_{pattern_name}\nfires={result["fires_count"]} squares={result["squares_viewed"]} first_detect={fd_str}')
    ax.set_aspect('equal')
    ax.legend()
    fname = OUT_DIR / f'search_s{scenario}_{pattern_name}.png'
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def main():
    print('Running deterministic search patterns (no RL agent loaded).')
    scenarios = [1,2,3,4,5,6]
    patterns = list(PATTERNS.keys())
    steps = 300
    summary = {}
    for s in scenarios:
        summary[s] = {}
        for p in patterns:
            print(f'Running scenario {s} pattern {p}...')
            res = run_pattern(p, s, steps=steps, seed=42)
            img = plot_run(res, s, p)
            summary[s][p] = {'fires': res['fires_count'], 'squares': res['squares_viewed'], 'first_detect': res.get('first_detect_step', ''), 'plot': str(img)}
            # regenerate summary images after each run so user sees progress; log errors
            try:
                generate_summary_images(summary, scenarios, patterns, BASE_DIR)
            except Exception as e:
                print('generate_summary_images failed during runs:', e)
    # Save summary CSV
    import csv
    outcsv = BASE_DIR / 'search_summary.csv'
    with open(outcsv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scenario','pattern','fires_found','squares_viewed','first_detect_step','plot'])
        for s in scenarios:
            for p in patterns:
                row = [s, p, summary[s][p]['fires'], summary[s][p]['squares'], summary[s][p].get('first_detect', ''), summary[s][p]['plot']]
                w.writerow(row)
    print('Done. summary saved to', outcsv)
    # Ensure final summary images are up-to-date
    try:
        f1, f2, fd_csv = generate_summary_images(summary, scenarios, patterns, BASE_DIR)
        print('Saved summary images:', f1, f2)
    except Exception as e:
        print('generate_summary_images failed at end:', e)


def generate_summary_images(summary, scenarios, patterns, outdir):
    """Create heatmaps for fires found and squares viewed and save to outdir."""
    import numpy as _np
    import matplotlib.pyplot as _plt
    # build matrices
    S = len(scenarios)
    P = len(patterns)
    fires_mat = _np.zeros((S, P), dtype=float)
    squares_mat = _np.zeros((S, P), dtype=float)
    first_detect = _np.full((S, P), _np.nan)
    s_idx = {s: i for i, s in enumerate(scenarios)}
    p_idx = {p: j for j, p in enumerate(patterns)}
    for s in scenarios:
        for p in patterns:
            v = summary.get(s, {}).get(p, None)
            if v is None:
                continue
            fires_mat[s_idx[s], p_idx[p]] = v.get('fires', 0)
            squares_mat[s_idx[s], p_idx[p]] = v.get('squares', 0)
            fd = v.get('first_detect', '')
            try:
                first_detect[s_idx[s], p_idx[p]] = float(fd) if fd != '' else _np.nan
            except Exception:
                first_detect[s_idx[s], p_idx[p]] = _np.nan

    # Save fires heatmap
    fig, ax = _plt.subplots(figsize=(max(4, P), max(3, S)))
    im = ax.imshow(fires_mat, cmap='Reds', aspect='auto')
    ax.set_xticks(range(P))
    ax.set_xticklabels(patterns, rotation=45, ha='right')
    ax.set_yticks(range(S))
    ax.set_yticklabels([str(s) for s in scenarios])
    ax.set_title('Fires Found')
    fig.colorbar(im, ax=ax)
    f1 = outdir / 'summary_fires_heatmap.png'
    try:
        fig.savefig(f1, dpi=150, bbox_inches='tight')
    except Exception:
        fig.savefig(f1, dpi=120)
    _plt.close(fig)

    # Save squares heatmap
    fig2, ax2 = _plt.subplots(figsize=(max(4, P), max(3, S)))
    im2 = ax2.imshow(squares_mat, cmap='Blues', aspect='auto')
    ax2.set_xticks(range(P))
    ax2.set_xticklabels(patterns, rotation=45, ha='right')
    ax2.set_yticks(range(S))
    ax2.set_yticklabels([str(s) for s in scenarios])
    ax2.set_title('Squares Viewed')
    fig2.colorbar(im2, ax=ax2)
    f2 = outdir / 'summary_squares_heatmap.png'
    try:
        fig2.savefig(f2, dpi=150, bbox_inches='tight')
    except Exception:
        fig2.savefig(f2, dpi=120)
    _plt.close(fig2)

    # Save first detect matrix csv
    import csv as _csv
    fd_csv = outdir / 'first_detect_matrix.csv'
    with open(fd_csv, 'w', newline='') as cf:
        writer = _csv.writer(cf)
        header = ['scenario/pattern'] + patterns
        writer.writerow(header)
        for s in scenarios:
            row = [s]
            for p in patterns:
                val = summary.get(s, {}).get(p, {}).get('first_detect', '')
                row.append(val)
            writer.writerow(row)

    return f1, f2, fd_csv

if __name__ == '__main__':
    main()
