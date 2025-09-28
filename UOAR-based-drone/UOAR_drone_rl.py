import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import csv
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- TUNABLE CONSTANTS (centralized) ---
# Reward/penalty scaling
# Split terminal penalties so collisions (hard failures) can be stronger than timeouts
COLLISION_PENALTY = -80.0
TIMEOUT_PENALTY = -20.0
# Backwards-compatible alias (some code paths may still reference TERMINAL_PENALTY)
TERMINAL_PENALTY = COLLISION_PENALTY

# Energy: smaller coefficient and tighter clipping to avoid domination
ENERGY_COEF = 0.002   # smaller coefficient for per-step energy magnitude
# Per-step clip for energy-based penalty (post-baseline subtraction)
ENERGY_CLIP = 1.0     # tighter clip to avoid huge single-step penalties
REWARD_CLIP = 50.0
NORMALIZED_REWARD_CLIP = 5.0
PROXIMITY_SCALE = 18.0
OPTIMAL_BONUS = 12.0
# Per-component weights to balance reward components (prox, exploration, energy)
PROX_WEIGHT = 1.5
EXPL_WEIGHT = 0.6
# Lower energy weight significantly so it no longer drowns other signals
ENERGY_WEIGHT = 0.00008
# Small L2 regularizer on policy actions to discourage saturation (in actor loss)
ACT_REG = 1e-2

# Replay / normalization warm-up
REPLAY_WARMUP = 20000  # env steps before training/normalizing rewards
# Increase warmup transitions so normalization and critic updates are stable
N_WARMUP_TRANSITIONS = 10000  # transitions to collect before using normalization

# Optional small actor-side energy regularizer (penalize large action magnitudes)
ENERGY_ACT_REG = 2e-3

# Training / TD3 defaults
TARGET_CLIP = 4.0     # slightly relaxed target clipping to preserve learning signal
GRAD_CLIP_NORM = 1.0
# Actor regularization to discourage action saturation
ACT_REG = 5e-3

# --- ENVIRONMENT ---
class DroneEnv(gym.Env):
    def __init__(self, area_size=10, drone_radius=0.25, fire_line=((5,5,0),(5,5,10)), fire_radius=0.25, safety_margin=0.1, max_step_size=0.5, curriculum_level=0, scenario=1):
        super().__init__()
        self.area_size = area_size
        # Halve drone radius and safety margin (bounding box) for all scenarios
        self.drone_radius = drone_radius
        self.fire_line = np.array(fire_line)
        # Halve fire_radius for all scenarios except scenario 1
        if scenario == 1:
            self.fire_radius = 0.5  # original fire size for scenario 1
        else:
            self.fire_radius = fire_radius  # default is 0.25 (halved)
        self.base_safety_margin = safety_margin
        # Curriculum learning: start with larger safety margin, gradually reduce
        self.curriculum_level = curriculum_level
        self.safety_margin = safety_margin + max(0, (5 - curriculum_level) * 0.1)
        self.max_step_size = max_step_size
        self.n_drones = 2
        self.scenario = scenario
        # Always define self.fire_centers for all scenarios
        self.fire_centers = []
        # For scenario 2: fire spread state
        if scenario == 2:
            self.fire_centers = [np.mean(self.fire_line, axis=0)]  # List of fire centers
            # fire_radius already set above
            self.fire_spread_rate = 0.5  # Not used, but kept for compatibility
            self.fire_spawn_radius = 5.0  # New fires spawn within this radius from any fire
            self.max_fires = 8  # Limit number of fires for performance
        # Scenario 3: line of fires (cylinders) along x axis
        elif scenario == 3:
            # Place fires at regular intervals along x axis at y=area_size/2, z=5 (on ground)
            n_fires = 10
            x_positions = np.linspace(1, self.area_size-1, n_fires)
            y_pos = self.area_size / 2
            z_pos = 5.0
            self.fire_centers = [np.array([x, y_pos, z_pos]) for x in x_positions]
            # fire_radius already set above
        # Scenario 5: impassable fire wall along x axis
        elif scenario == 5:
            self.fire_wall_x = self.area_size / 2
            # fire_radius already set above
            self.fire_centers = []  # Ensure attribute exists for visualization
        # Scenario 4: 5 randomly placed fires
        elif scenario == 4:
            n_fires = 5
            self.fire_centers = []
            for _ in range(n_fires):
                # Place fires randomly in the area, at random z between 5 and 10
                x = np.random.uniform(1, self.area_size-1)
                y = np.random.uniform(1, self.area_size-1)
                z = np.random.uniform(5, 10)
                self.fire_centers.append(np.array([x, y, z]))
            # fire_radius already set above
        # Scenario 6: large central fire
        elif scenario == 6:
            center = np.array([self.area_size / 2, self.area_size / 2, self.area_size / 2])
            self.fire_centers = [center]
            self.fire_radius = 2.5  # Large fire size for scenario 6
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_drones,3), dtype=np.float32)
        # Each drone: [x, y, z, fire_below, other_x, other_y, other_z]
        low = np.array([0,0,0,0,0,0,0]*self.n_drones).reshape((self.n_drones,7))
        high = np.array([area_size,area_size,area_size,1,area_size,area_size,area_size]*self.n_drones).reshape((self.n_drones,7))
        self.observation_space = spaces.Box(low=low, high=high, shape=(self.n_drones,7), dtype=np.float32)
        # --- Add cell_rewards for regenerating reward ---
        grid_size = 0.5
        grid_dim = int(self.area_size // grid_size)
        self.grid_size = grid_size
        self.grid_dim = grid_dim
        # Keep per-cell rewards small to avoid runaway episode sums
        self.max_cell_reward = 10
        self.cell_regen_rate = 2
        self.cell_rewards = np.full((grid_dim, grid_dim), self.max_cell_reward, dtype=np.int32)
        self.reset()

    def reset(self, seed=None, options=None):
        def point_to_segment_dist(p, a, b):
            ap = p - a
            ab = b - a
            t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-8)
            t = np.clip(t, 0, 1)
            closest = a + t * ab
            return np.linalg.norm(p - closest)
        safe = False
        max_attempts = 10
        for _ in range(max_attempts):
            pos1 = np.random.uniform(0.5, self.area_size - 0.5, size=3)
            pos2 = np.random.uniform(0.5, self.area_size - 0.5, size=3)
            # Force z to be between 5 and 10
            pos1[2] = np.random.uniform(5, 10)
            pos2[2] = np.random.uniform(5, 10)
            dist1 = point_to_segment_dist(pos1, self.fire_line[0], self.fire_line[1])
            dist2 = point_to_segment_dist(pos2, self.fire_line[0], self.fire_line[1])
            min_dist = self.drone_radius + self.fire_radius + self.safety_margin
            # Drones must not collide with fire or each other
            if (dist1 > min_dist and dist2 > min_dist and np.linalg.norm(pos1-pos2) > 2*self.drone_radius + self.safety_margin):
                safe = True
                break
        if not safe:
            pos1 = np.array([1.0, 1.0, 7.5])  # z in [5,10]
            pos2 = np.array([8.0, 8.0, 7.5])
        self.drone_pos = np.stack([pos1, pos2])
        self.done = [False, False]
        self.cell_rewards = np.full((self.grid_dim, self.grid_dim), self.max_cell_reward, dtype=np.int32)  # Reset cell rewards to max
        if self.scenario == 2:
            self.fire_centers = [np.mean(self.fire_line, axis=0)]
        if self.scenario == 3:
            n_fires = 10
            x_positions = np.linspace(1, self.area_size-1, n_fires)
            y_pos = self.area_size / 2
            z_pos = 5.0
            self.fire_centers = [np.array([x, y_pos, z_pos]) for x in x_positions]
        if self.scenario == 4:
            n_fires = 5
            self.fire_centers = []
            for _ in range(n_fires):
                x = np.random.uniform(1, self.area_size-1)
                y = np.random.uniform(1, self.area_size-1)
                z = np.random.uniform(5, 10)
                self.fire_centers.append(np.array([x, y, z]))
        if self.scenario == 5:
            # No fire centers, just a wall at x=area_size/2
            self.fire_centers = []  # Ensure attribute exists for visualization
        return self._get_obs(), {}

    def _get_obs(self):
        # Each drone sees a cone on the ground; the higher it is, the larger the cone
        def point_to_segment_dist_2d(p, a, b):
            # 2D distance from point p to segment ab
            ap = p - a
            ab = b - a
            t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-8)
            t = np.clip(t, 0, 1)
            closest = a + t * ab
            return np.linalg.norm(p - closest)
        def is_blocked_by_fire(drone_xy, fire_centers_2d, fire_radius, target_xy):
            # Returns True if a fire is between drone_xy and target_xy
            for fc in fire_centers_2d:
                # Ensure fc is 2D for all calculations
                fc2d = fc[:2] if fc.shape[0] > 2 else fc
                # Vector from drone to fire and drone to target
                v_fire = fc2d - drone_xy
                v_target = (target_xy[:2] if target_xy.shape[0] > 2 else target_xy) - drone_xy
                dist_to_fire = np.linalg.norm(v_fire)
                dist_to_target = np.linalg.norm(v_target)
                if dist_to_fire < dist_to_target and dist_to_fire > 1e-6:
                    # Angle between vectors
                    cos_angle = np.dot(v_fire, v_target) / (dist_to_fire * dist_to_target + 1e-8)
                    if cos_angle > 0.99:  # ~8 deg cone, adjust as needed
                        # Check if fire center is close to the line from drone to target
                        proj = np.dot(v_fire, v_target) / (np.linalg.norm(v_target) + 1e-8)
                        closest = drone_xy + v_target / np.linalg.norm(v_target) * proj
                        if np.linalg.norm(fc2d - closest) <= fire_radius:
                            return True
            return False
        obs = []
        k = 0.5  # scaling factor for cone radius (reduced by half)
        fire_a_2d = np.array([self.fire_line[0][0], self.fire_line[0][1]])
        fire_b_2d = np.array([self.fire_line[1][0], self.fire_line[1][1]])
        for i in range(self.n_drones):
            pos = self.drone_pos[i]
            other = self.drone_pos[1-i]
            drone_xy = np.array([pos[0], pos[1]])
            z = pos[2]
            view_radius = k * z / 8
            fire_in_view = 0.0
            # For all scenarios, check occlusion by fire (cannot see behind fire)
            fire_centers_2d = np.array([[fc[0], fc[1]] for fc in self.fire_centers]) if self.fire_centers else np.empty((0,2))
            # For scenario 1 and 5, treat the fire line or wall as a set of closely spaced fire centers for occlusion
            if (self.scenario == 1 or self.scenario == 5) and fire_centers_2d.shape[0] == 0:
                # Discretize the fire line/wall into points for occlusion
                n_points = 20
                if self.scenario == 1:
                    a, b = self.fire_line[0], self.fire_line[1]
                else:
                    # Wall at x=fire_wall_x, spanning y and z
                    a = np.array([self.fire_wall_x, 0])
                    b = np.array([self.fire_wall_x, self.area_size])
                fire_centers_2d = np.array([a + (b - a) * t for t in np.linspace(0, 1, n_points)])
            # Check if any fire is visible (not blocked by another fire)
            for fc in fire_centers_2d:
                dist = np.linalg.norm(drone_xy - fc[:2])
                if dist <= (view_radius + self.fire_radius):
                    blocked = is_blocked_by_fire(drone_xy, fire_centers_2d, self.fire_radius, fc)
                    if not blocked:
                        fire_in_view = 1.0
                        break
            obs.append(np.concatenate([pos, [fire_in_view], other]))
        return np.stack(obs)

    def step(self, actions, seen_grids=None):
        actions = np.clip(actions, -1, 1)
        rewards = [0, 0]
        dones = [False, False]
        # per-step bookkeeping for logging: exploration and proximity contributions
        exploration_contrib = [0.0 for _ in range(self.n_drones)]
        proximity_contrib = [0.0 for _ in range(self.n_drones)]
        # Move both drones (with scenario 5 wall and random crash logic applied inline)
        for i in range(self.n_drones):
            a = actions[i]
            a = a / (np.linalg.norm(a) + 1e-8) * min(np.linalg.norm(a), 1.0)
            a = a * self.max_step_size
            # Scenario 5: 1% random crash chance
            if self.scenario == 5 and np.random.rand() < 0.006:
                rewards[i] = COLLISION_PENALTY
                dones[i] = True
                continue
            # Scenario 5: impassable fire wall at x=area_size/2
            if self.scenario == 5:
                x0 = self.drone_pos[i][0]
                x1 = np.clip(self.drone_pos[i][0] + a[0], 0, self.area_size)
                if (x0 < self.fire_wall_x and x1 >= self.fire_wall_x) or (x0 > self.fire_wall_x and x1 <= self.fire_wall_x):
                    rewards[i] = COLLISION_PENALTY
                    dones[i] = True
                    continue
            # Normal movement if not crashing or hitting wall
            self.drone_pos[i] = np.clip(self.drone_pos[i] + a, [0, 0, 5], [self.area_size, self.area_size, 10])

        # Scenario 2: fire spreads each step
        if self.scenario == 2:
            # With some probability, add a new fire near an existing fire
            if len(self.fire_centers) < self.max_fires:
                for center in list(self.fire_centers):
                    if np.random.rand() < 0.02:  # 1% chance per fire per step
                        angle = np.random.uniform(0, 2 * np.pi)
                        height = np.random.uniform(-2, 2)
                        r = np.random.uniform(0.5, self.fire_spawn_radius)
                        dx = r * np.cos(angle)
                        dy = r * np.sin(angle)
                        dz = height
                        new_center = center + np.array([dx, dy, dz])
                        # Keep within bounds
                        new_center = np.clip(new_center, [0, 0, 5], [self.area_size, self.area_size, 10])
                        # Only add if not too close to existing fires
                        if all(np.linalg.norm(new_center - fc) > self.fire_radius * 1.5 for fc in self.fire_centers):
                            self.fire_centers.append(new_center)

        # Check collisions and apply reward/penalty system for all scenarios
        def point_to_segment_dist(p, a, b):
            ap = p - a
            ab = b - a
            t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-8)
            t = np.clip(t, 0, 1)
            closest = a + t * ab
            return np.linalg.norm(p - closest)

        min_safe_dist = self.drone_radius + self.fire_radius + self.safety_margin
        collision_dist = self.drone_radius + self.fire_radius  # Actual collision boundary

        alpha = 1.2  # Much gentler exponential curve
        # Reduced proximity scale to avoid huge Q values and unstable training
        proximity_scale = PROXIMITY_SCALE

        for i in range(self.n_drones):
            # For scenarios 2, 3, 4, 6: use fire_centers; for 1 and 5: use fire_line
            if self.scenario in [2, 3, 4, 6]:
                dists = [np.linalg.norm(self.drone_pos[i] - fc) for fc in self.fire_centers]
                min_dist = min(dists)
                collision_dist = self.drone_radius + self.fire_radius
                min_safe_dist = self.drone_radius + self.fire_radius + self.safety_margin
                # If inside the fire (distance to center < fire_radius), apply same penalty everywhere
                if min_dist <= self.fire_radius:
                    rewards[i] = COLLISION_PENALTY
                    dones[i] = True
                    continue
                # --- Proximity reward logic for keeping average fire distance near 1 ---
                optimal_dist = 1.0
                distance_error = abs(min_dist - optimal_dist)
                # Small bounded proximal encouragement scaled by PROXIMITY_SCALE
                prox = PROXIMITY_SCALE * max(0, 1 - distance_error)
                rewards[i] += PROX_WEIGHT * prox
                proximity_contrib[i] += PROX_WEIGHT * prox
                if min_dist > optimal_dist + 1.0:
                    pen = - (PROXIMITY_SCALE / 2.0) * (min_dist - optimal_dist)
                    rewards[i] += PROX_WEIGHT * pen
                    proximity_contrib[i] += PROX_WEIGHT * pen
            else:
                min_dist = point_to_segment_dist(self.drone_pos[i], self.fire_line[0], self.fire_line[1])
                collision_dist = self.drone_radius + self.fire_radius
                min_safe_dist = self.drone_radius + self.fire_radius + self.safety_margin
            if dones[i]:
                continue  # Already crashed (e.g., scenario 5 random crash or wall)
            if min_dist <= collision_dist:
                # Hard collision - terminal state
                rewards[i] = TERMINAL_PENALTY
                dones[i] = True
            elif min_dist <= min_safe_dist:
                # Danger zone - steep penalty but not terminal, encouraging learning
                danger_factor = (min_safe_dist - min_dist) / (min_safe_dist - collision_dist)
                penalty = - (PROXIMITY_SCALE * 2.0) * (danger_factor ** 2)  # Quadratic penalty in danger zone
                rewards[i] += penalty
                proximity_contrib[i] += penalty
            else:
                # Safe zone - exponential reward for proximity
                safe_distance = min_dist - min_safe_dist
                exp_reward = proximity_scale * np.exp(-alpha * safe_distance)
                rewards[i] += PROX_WEIGHT * exp_reward
                proximity_contrib[i] += PROX_WEIGHT * exp_reward
                # Additional bonus for being in optimal range (just outside safety margin)
                optimal_range = 0.3  # Distance beyond safety margin that's considered optimal
                if safe_distance <= optimal_range:
                    optimal_bonus = OPTIMAL_BONUS * (1 - safe_distance / optimal_range)
                    rewards[i] += PROX_WEIGHT * optimal_bonus
                    proximity_contrib[i] += PROX_WEIGHT * optimal_bonus

        # Drone-drone collision
        if np.linalg.norm(self.drone_pos[0] - self.drone_pos[1]) < 2*self.drone_radius + self.safety_margin:
            rewards = [COLLISION_PENALTY, COLLISION_PENALTY]
            dones = [True, True]

        # Exploration reward (reduced impact)
        grid_size = self.grid_size
        grid_dim = self.grid_dim
        max_reward = self.max_cell_reward
        regen_rate = self.cell_regen_rate
        # Build a mask of which cells are seen by any drone this step, accounting for occlusion by fire
        seen_mask = np.zeros((grid_dim, grid_dim), dtype=bool)
        for i in range(self.n_drones):
            pos = self.drone_pos[i]
            drone_xy = np.array([pos[0], pos[1]])
            z = pos[2]
            view_radius = 1.0 * z / 8
            x_min = max(0, int((drone_xy[0] - view_radius) // grid_size))
            x_max = min(grid_dim - 1, int((drone_xy[0] + view_radius) // grid_size))
            y_min = max(0, int((drone_xy[1] - view_radius) // grid_size))
            y_max = min(grid_dim - 1, int((drone_xy[1] + view_radius) // grid_size))
            # For all scenarios, use fire occlusion
            fire_centers_2d = np.array([[fc[0], fc[1]] for fc in self.fire_centers]) if self.fire_centers else np.empty((0,2))
            if (self.scenario == 1 or self.scenario == 5) and fire_centers_2d.shape[0] == 0:
                n_points = 20
                if self.scenario == 1:
                    a, b = self.fire_line[0], self.fire_line[1]
                else:
                    a = np.array([self.fire_wall_x, 0])
                    b = np.array([self.fire_wall_x, self.area_size])
                fire_centers_2d = np.array([a + (b - a) * t for t in np.linspace(0, 1, n_points)])
            for gx in range(x_min, x_max+1):
                for gy in range(y_min, y_max+1):
                    cx = gx * grid_size + grid_size/2
                    cy = gy * grid_size + grid_size/2
                    cell_xy = np.array([cx, cy])
                    if np.linalg.norm(drone_xy - cell_xy) <= view_radius:
                        occluded = False
                        for fc in fire_centers_2d:
                            v_fire = fc[:2] - drone_xy
                            v_cell = cell_xy - drone_xy
                            dist_to_fire = np.linalg.norm(v_fire)
                            dist_to_cell = np.linalg.norm(v_cell)
                            if dist_to_fire < dist_to_cell and dist_to_fire > 1e-6:
                                cos_angle = np.dot(v_fire, v_cell) / (dist_to_fire * dist_to_cell + 1e-8)
                                if cos_angle > 0.99:
                                    occluded = True
                                    break
                        if not occluded:
                            seen_mask[gx, gy] = True
        # Give reduced reward for seen cells, reset their reward to 0
        exploration_scale = 0.05  # Reduced impact of exploration (smaller to avoid large totals)
        # Iterate cells and distribute clamped per-cell reward to drones equally
        for gx in range(grid_dim):
            for gy in range(grid_dim):
                if seen_mask[gx, gy]:
                    base = float(self.cell_rewards[gx, gy])
                    # Removed additive constant and clamp to tighter range
                    reward = base * exploration_scale
                    reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))
                    if reward != 0:
                        per_drone = reward / float(self.n_drones)
                        for i in range(self.n_drones):
                            rewards[i] += per_drone
                            exploration_contrib[i] += per_drone
                        self.cell_rewards[gx, gy] = 0
                else:
                    # Regenerate reward for unseen cells
                    if self.cell_rewards[gx, gy] < max_reward:
                        self.cell_rewards[gx, gy] = min(max_reward, self.cell_rewards[gx, gy] + regen_rate)

        # Reduced edge penalty to not interfere with fire proximity
        edge_penalty_scale = 50.0  # Much lower penalty
        edge_penalty_alpha = 2.0   # Less steep penalty curve
        for i in range(self.n_drones):
            pos = self.drone_pos[i]
            # Distance to each wall (x=0, x=max, y=0, y=max, z=5, z=10)
            dists = [pos[0], self.area_size - pos[0], pos[1], self.area_size - pos[1], pos[2] - 5, 10 - pos[2]]
            min_dist = min(dists)
            # Only apply edge penalty when very close to walls
            if min_dist < 1.0:  # Only within 1 unit of walls
                edge_penalty = edge_penalty_scale * np.exp(-edge_penalty_alpha * min_dist)
                rewards[i] -= edge_penalty

    # --- Custom penalties and exploration bonus ---
        # Track previous positions for each drone
        if not hasattr(self, 'prev_positions'):
            self.prev_positions = [None for _ in range(self.n_drones)]
        if not hasattr(self, 'fires_visited'):
            self.fires_visited = [set() for _ in range(self.n_drones)]

        for i in range(self.n_drones):
            pos = self.drone_pos[i]
            # Penalty for staying in the same place for too long
            if self.prev_positions[i] is not None:
                movement = np.linalg.norm(pos - self.prev_positions[i])
                # Reduced movement penalty to avoid large accumulated negatives
                if movement < 0.15:  # threshold for 'staying still'
                    rewards[i] -= 1.0

            self.prev_positions[i] = pos.copy()
            # Exploration bonus for discovering new fires
            for idx, fc in enumerate(self.fire_centers):
                dist_to_fire = np.linalg.norm(pos - fc)
                if dist_to_fire < (self.fire_radius + self.safety_margin + 1.0):
                    if idx not in self.fires_visited[i]:
                        bonus = 20.0
                        rewards[i] += EXPL_WEIGHT * bonus
                        exploration_contrib[i] += EXPL_WEIGHT * bonus
                        self.fires_visited[i].add(idx)

        # Small per-step energy penalty computed from change in action (Δa) to encourage smooth control
        # This reduces penalizing necessary baseline thrust and avoids driving agent to trivially low-action policies.
        energy_coeff = ENERGY_COEF
        # maintain EMA baseline for energy if not present
        if not hasattr(self, 'energy_ema'):
            self.energy_ema = 0.0
            # faster-moving baseline so typical energy magnitudes are tracked quickly
            self.energy_ema_alpha = 0.02
        # Ensure info exists before we write into it
        if not isinstance(locals().get('info', None), dict):
            info = {}
        if 'energy' not in info:
            info['energy'] = [0.0 for _ in range(self.n_drones)]
        # Track per-step energy components separately so logging can be precise
        energy_components = [0.0 for _ in range(self.n_drones)]
        for i in range(self.n_drones):
            try:
                a = np.clip(actions[i], -1.0, 1.0)
                # If prev action exists, penalize change in action; else use current magnitude as proxy
                if hasattr(self, 'prev_actions') and self.prev_actions[i] is not None:
                    da = a - self.prev_actions[i]
                else:
                    da = a
                # energy measured as mean absolute delta (lower variance than squared)
                energy_raw = energy_coeff * float(np.mean(np.abs(da)))
                # update EMA baseline
                self.energy_ema = (1 - self.energy_ema_alpha) * self.energy_ema + self.energy_ema_alpha * energy_raw
                energy_rel = energy_raw - self.energy_ema
                # Clip to avoid huge single-step penalties
                energy_rel = float(np.clip(energy_rel, -ENERGY_CLIP, ENERGY_CLIP))
                # Subtract weighted penalty
                energy_penalty = ENERGY_WEIGHT * energy_rel
                rewards[i] -= energy_penalty
                # record for logging (store raw and relative versions)
                energy_components[i] = float(energy_raw)
                # Keep legacy 'energy' key as raw magnitude for downstream logging
                info['energy'][i] = float(energy_raw)
            except Exception:
                energy_components[i] = float('nan')
                info['energy'][i] = float('nan')

        # persist prev_actions for next step
        if not hasattr(self, 'prev_actions'):
            self.prev_actions = [None for _ in range(self.n_drones)]
        for i in range(self.n_drones):
            self.prev_actions[i] = np.clip(actions[i], -1.0, 1.0)

        self.done = dones
        # Attach the component breakdown for downstream logging: exploration, proximity, energy
        info.update({'exploration_contrib': exploration_contrib, 'proximity_contrib': proximity_contrib, 'energy': energy_components})
        return self._get_obs(), rewards, dones, False, info

    def render(self, drone_pos=None, fire_centers=None):
        # 3D visualization using matplotlib
        fig = plt.figure(1)
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([0, self.area_size])
        ax.set_ylim([0, self.area_size])
        ax.set_zlim([0, self.area_size])
        # Draw fire line or fires
        if self.scenario == 5:
            # Draw fire wall as a red plane at x=area_size/2
            fire_x = self.fire_wall_x
            y = np.linspace(0, self.area_size, 2)
            z = np.linspace(0, self.area_size, 2)
            Y, Z = np.meshgrid(y, z)
            X = np.full_like(Y, fire_x)
            ax.plot_surface(X, Y, Z, color='red', alpha=0.4)
        elif self.scenario in [2, 3, 4, 6]:
            # Draw all fire centers as vertical cylinders attached to the floor
            fire_height = self.area_size  # Cylinder height (floor to ceiling)
            n_cylinder = 24
            for fc in (fire_centers if fire_centers is not None else self.fire_centers):
                # Cylinder base center at (fc[0], fc[1], 0), top at (fc[0], fc[1], fire_height)
                theta = np.linspace(0, 2 * np.pi, n_cylinder)
                x = fc[0] + self.fire_radius * np.cos(theta)
                y = fc[1] + self.fire_radius * np.sin(theta)
                z_bottom = np.zeros_like(theta)
                z_top = np.full_like(theta, fire_height)
                # Draw side faces
                for i in range(n_cylinder - 1):
                    verts = [
                        [x[i], y[i], 0],
                        [x[i+1], y[i+1], 0],
                        [x[i+1], y[i+1], fire_height],
                        [x[i], y[i], fire_height]
                    ]
                    poly = Poly3DCollection([verts], color='red', alpha=0.4)
                    ax.add_collection3d(poly)
                # Draw top and bottom faces
                verts_top = list(zip(x, y, z_top))
                verts_bottom = list(zip(x, y, z_bottom))
                poly_top = Poly3DCollection([verts_top], color='red', alpha=0.4)
                poly_bottom = Poly3DCollection([verts_bottom], color='red', alpha=0.4)
                ax.add_collection3d(poly_top)
                ax.add_collection3d(poly_bottom)
        else:
            p1, p2 = self.fire_line
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='r', linewidth=4, label='Fire Line')
        # Draw drones as spheres
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        for pos in (drone_pos if drone_pos is not None else self.drone_pos):
            x = pos[0] + self.drone_radius * np.cos(u) * np.sin(v)
            y = pos[1] + self.drone_radius * np.sin(u) * np.sin(v)
            z = pos[2] + self.drone_radius * np.cos(v)
            ax.plot_surface(x, y, z, color='b', alpha=0.5)
        # Draw cone base (field of view) for each drone
        k = 0.5  # must match _get_obs (reduced by half)
        for idx, pos in enumerate(drone_pos if drone_pos is not None else self.drone_pos):
            drone_xy = np.array([pos[0], pos[1]])
            z = pos[2]
            view_radius = k * z / 8
            theta = np.linspace(0, 2 * np.pi, 200)
            circle_x = drone_xy[0] + view_radius * np.cos(theta)
            circle_y = drone_xy[1] + view_radius * np.sin(theta)
            circle_z = np.zeros_like(theta)
            # Mask the field of view if blocked by a fire (scenarios 2/3)
            if self.scenario == 2 or self.scenario == 3:
                fire_centers_plot = fire_centers if fire_centers is not None else self.fire_centers
                fire_centers_2d = np.array([[fc[0], fc[1]] for fc in fire_centers_plot])
                mask = np.ones_like(theta, dtype=bool)
                for fc in fire_centers_2d:
                    v_fire = fc - drone_xy
                    dist_to_fire = np.linalg.norm(v_fire)
                    if dist_to_fire <= (view_radius + self.fire_radius):
                        angle_fire = np.arctan2(v_fire[1], v_fire[0])
                        # Block a sector behind the fire (shadow)
                        block_angle = np.arcsin(self.fire_radius / (dist_to_fire + 1e-8)) if dist_to_fire > self.fire_radius else np.pi
                        # For each theta, check if it is behind the fire
                        for j, t in enumerate(theta):
                            # Vector from drone to this theta
                            v_theta = np.array([np.cos(t), np.sin(t)])
                            angle_diff = np.arctan2(np.sin(t - angle_fire), np.cos(t - angle_fire))
                            # Block the sector behind the fire (opposite direction)
                            if abs(angle_diff) < block_angle:
                                # Only block if the point is further than the fire
                                pt = drone_xy + view_radius * v_theta
                                if np.linalg.norm(pt - drone_xy) > dist_to_fire:
                                    mask[j] = False
                # Only plot visible part of the disk
                circle_x_masked = circle_x[mask]
                circle_y_masked = circle_y[mask]
                circle_z_masked = circle_z[mask]
                if len(circle_x_masked) > 2:
                    ax.plot(circle_x_masked, circle_y_masked, circle_z_masked, color='orange', alpha=0.6, linewidth=1.5)
                    verts = [list(zip(circle_x_masked, circle_y_masked, circle_z_masked))]
                    poly = Poly3DCollection(verts, color='orange', alpha=0.15)
                    ax.add_collection3d(poly)
            else:
                # No occlusion
                ax.plot(circle_x, circle_y, circle_z, color='orange', alpha=0.6, linewidth=1.5)
                verts = [list(zip(circle_x, circle_y, circle_z))]
                poly = Poly3DCollection(verts, color='orange', alpha=0.15)
                ax.add_collection3d(poly)
        # Draw safety AABB as transparent cubes (12 edges each)
        for pos in (drone_pos if drone_pos is not None else self.drone_pos):
            aabb_min = pos - (self.drone_radius + self.safety_margin)
            aabb_max = pos + (self.drone_radius + self.safety_margin)
            # 8 corners of the cube
            corners = np.array([
                [aabb_min[0], aabb_min[1], aabb_min[2]],
                [aabb_max[0], aabb_min[1], aabb_min[2]],
                [aabb_max[0], aabb_max[1], aabb_min[2]],
                [aabb_min[0], aabb_max[1], aabb_min[2]],
                [aabb_min[0], aabb_min[1], aabb_max[2]],
                [aabb_max[0], aabb_min[1], aabb_max[2]],
                [aabb_max[0], aabb_max[1], aabb_max[2]],
                [aabb_min[0], aabb_max[1], aabb_max[2]],
            ])
            # List of edges as pairs of corner indices
            edges = [
                [0,1],[1,2],[2,3],[3,0], # bottom
                [4,5],[5,6],[6,7],[7,4], # top
                [0,4],[1,5],[2,6],[3,7]  # sides
            ]
            for s, e in edges:
                ax.plot([corners[s][0], corners[e][0]],
                        [corners[s][1], corners[e][1]],
                        [corners[s][2], corners[e][2]], color="g", alpha=0.3)
        ax.set_title(f"Drone positions: {np.array(drone_pos if drone_pos is not None else self.drone_pos).round(2)}")
        plt.draw()
        plt.pause(0.01)

# --- TD3 AGENT WITH aPE2 ACTION SELECTION ---
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh()  # output in [-1,1]
        )
    def forward(self, obs):
        return self.net(obs)

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


# Simple per-component running normalizer (Welford) for proximity/exploration/energy
class RewardNormalizer:
    def __init__(self, warmup=N_WARMUP_TRANSITIONS, clip=3.0):
        self.names = ['prox', 'expl', 'energy']
        self.counts = {n: 0 for n in self.names}
        self.means = {n: 0.0 for n in self.names}
        self.M2 = {n: 0.0 for n in self.names}
        self.warmup = warmup
        self.clip = clip

    def update(self, prox, expl, energy):
        vals = {'prox': float(prox), 'expl': float(expl), 'energy': float(energy)}
        for n in self.names:
            x = vals[n]
            self.counts[n] += 1
            if self.counts[n] == 1:
                self.means[n] = x
                self.M2[n] = 0.0
            else:
                delta = x - self.means[n]
                self.means[n] += delta / self.counts[n]
                delta2 = x - self.means[n]
                self.M2[n] += delta * delta2

    def ready(self):
        # ready when all components have at least warmup samples
        return all(self.counts[n] >= self.warmup for n in self.names)

    def normalize(self, prox, expl, energy):
        vals = {'prox': float(prox), 'expl': float(expl), 'energy': float(energy)}
        out = {}
        for n in self.names:
            if self.counts[n] > 1:
                var = self.M2[n] / max(1, (self.counts[n] - 1))
                std = (var ** 0.5) if var > 0 else 1.0
                z = (vals[n] - self.means[n]) / (std + 1e-8)
                z = float(max(-self.clip, min(self.clip, z)))
            else:
                z = float(vals[n])
            out[n] = z
        return out['prox'], out['expl'], out['energy']

class ReplayBuffer:
    def __init__(self, size, obs_dim, act_dim):
        self.size = size
        self.ptr = 0
        self.full = False
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros((size, 1), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)
    def add(self, o, a, r, no, d):
        self.obs[self.ptr] = o
        self.act[self.ptr] = a
        self.rew[self.ptr] = r
        self.next_obs[self.ptr] = no
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True
    def sample(self, batch_size):
        max_i = self.size if self.full else self.ptr
        if max_i == 0:
            raise ValueError("Replay buffer is empty")
        idx = np.random.randint(0, max_i, size=batch_size)
        return (
            torch.tensor(self.obs[idx], dtype=torch.float32),
            torch.tensor(self.act[idx], dtype=torch.float32),
            torch.tensor(self.rew[idx], dtype=torch.float32),
            torch.tensor(self.next_obs[idx], dtype=torch.float32),
            torch.tensor(self.done[idx], dtype=torch.float32)
        )

class TD3Agent:
    def save(self, filename):
        import pickle
        # Save only the state_dicts and relevant parameters
        data = {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critics': [c.state_dict() for c in self.critics],
            'critics_target': [c.state_dict() for c in self.critics_target],
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opts': [opt.state_dict() for opt in self.critic_opts],
            'buffer': self.buffer.__dict__,
            'params': {
                'area_size': self.area_size,
                'drone_radius': self.drone_radius,
                'fire_line': self.fire_line,
                'fire_radius': self.fire_radius,
                'safety_margin': self.safety_margin,
                'max_step_size': self.max_step_size,
                'obs_dim': self.obs_dim,
                'act_dim': self.act_dim,
                'n_candidates': self.n_candidates,
                'noise_levels': self.noise_levels,
                'gamma': self.gamma,
                'tau': self.tau,
                'policy_noise': self.policy_noise,
                'noise_clip': self.noise_clip,
                'policy_delay': self.policy_delay,
                'batch_size': self.batch_size,
                'total_it': self.total_it
            }
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filename):
        import pickle
        data = None
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.actor.load_state_dict(data['actor'])
        self.actor_target.load_state_dict(data['actor_target'])
        for c, state in zip(self.critics, data['critics']):
            c.load_state_dict(state)
        for c, state in zip(self.critics_target, data['critics_target']):
            c.load_state_dict(state)
        self.actor_opt.load_state_dict(data['actor_opt'])
        for opt, state in zip(self.critic_opts, data['critic_opts']):
            opt.load_state_dict(state)
        self.buffer.__dict__.update(data['buffer'])
        # Optionally update params if needed
        for k, v in data['params'].items():
            setattr(self, k, v)
    def decay_noise(self, decay_rate=0.99, min_noise=0.01):
        """Decay the noise levels by decay_rate, not going below min_noise."""
        self.noise_levels = [max(n * decay_rate, min_noise) for n in self.noise_levels]
    def __init__(self, area_size, drone_radius, fire_line, fire_radius, safety_margin, max_step_size, obs_dim=7, act_dim=3, n_critics=2, n_candidates=30, noise_levels=None, gamma=0.95, tau=0.005, policy_noise=0.12, noise_clip=0.25, policy_delay=2, buffer_size=200000, batch_size=2048, hidden_size=384):
        if noise_levels is None:
            # stronger initial candidate noise for exploration
            noise_levels = [0.1, 0.2, 0.3, 0.5]
        self.area_size = area_size
        self.drone_radius = drone_radius
        self.fire_line = np.array(fire_line)
        self.fire_radius = fire_radius
        self.safety_margin = safety_margin
        self.max_step_size = max_step_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # Actor / Critic networks
        self.actor = Actor(self.obs_dim, self.act_dim, hidden=hidden_size)
        self.actor_target = Actor(self.obs_dim, self.act_dim, hidden=hidden_size)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critics = nn.ModuleList([Critic(self.obs_dim, self.act_dim, hidden=hidden_size) for _ in range(n_critics)])
        self.critics_target = nn.ModuleList([Critic(self.obs_dim, self.act_dim, hidden=hidden_size) for _ in range(n_critics)])
        for i in range(n_critics):
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())
        # Reduce actor/critic learning rates for stability
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        # Lower critic LR to stabilize training further
        self.critic_opts = [torch.optim.Adam(c.parameters(), lr=5e-5) for c in self.critics]
        self.buffer = ReplayBuffer(buffer_size, self.obs_dim, self.act_dim)
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.n_candidates = n_candidates
        self.noise_levels = noise_levels
        self.total_it = 0
        # Running reward normalization (Welford's algorithm)
        self.r_count = 0
        self.r_mean = 0.0
        self.r_M2 = 0.0
    def select_action(self, obs_np):
        # default: stochastic candidate-based selection
        return self._select_action_candidates(obs_np, stochastic=True)

    def select_action(self, obs_np, deterministic=False):
        """Public API: return action. If deterministic=True, use raw actor output without candidate sampling."""
        if deterministic:
            obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a = self.actor(obs).cpu().numpy()[0]
            return np.clip(a, -1, 1)
        else:
            return self._select_action_candidates(obs_np, stochastic=True)

    def _select_action_candidates(self, obs_np, stochastic=True):
        obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            base_action = self.actor(obs).cpu().numpy()[0]
        
        # Enhanced candidate generation for fire-line following
        candidates = []
        
        # 1. Base policy action (no noise)
        candidates.append(base_action)
        
        # 2. Random noise variations
        for noise in self.noise_levels:
            for _ in range(self.n_candidates // len(self.noise_levels)):
                perturbed = base_action + np.random.normal(0, noise, size=self.act_dim)
                perturbed = np.clip(perturbed, -1, 1)
                candidates.append(perturbed)
        
        # 3. Fire-directed actions (biased towards fire line)
        current_pos = obs_np[:3]
        fire_start, fire_end = self.fire_line[0], self.fire_line[1]
        
        # Find closest point on fire line to current position
        def point_to_segment_dist_and_closest(p, a, b):
            ap = p - a
            ab = b - a
            t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-8)
            t = np.clip(t, 0, 1)
            closest = a + t * ab
            return np.linalg.norm(p - closest), closest
        
        dist_to_fire, closest_fire_point = point_to_segment_dist_and_closest(current_pos, fire_start, fire_end)
        
        # Generate actions that move towards optimal distance from fire
        min_safe_dist = self.drone_radius + self.fire_radius + self.safety_margin
        optimal_dist = min_safe_dist + 0.15  # Slightly outside safety margin
        
        if dist_to_fire > optimal_dist + 0.5:
            # Too far - move towards fire
            direction_to_fire = closest_fire_point - current_pos
            direction_to_fire = direction_to_fire / (np.linalg.norm(direction_to_fire) + 1e-8)
            for intensity in [0.3, 0.6, 0.9]:
                fire_action = direction_to_fire * intensity
                fire_action = np.clip(fire_action, -1, 1)
                candidates.append(fire_action)
        elif dist_to_fire < min_safe_dist + 0.1:
            # Too close - move away from fire
            direction_away = current_pos - closest_fire_point
            direction_away = direction_away / (np.linalg.norm(direction_away) + 1e-8)
            for intensity in [0.4, 0.7, 1.0]:
                away_action = direction_away * intensity
                away_action = np.clip(away_action, -1, 1)
                candidates.append(away_action)
        else:
            # Good distance - move parallel to fire line
            fire_direction = fire_end - fire_start
            fire_direction = fire_direction / (np.linalg.norm(fire_direction) + 1e-8)
            for direction in [1, -1]:  # Both directions along fire line
                for intensity in [0.3, 0.6]:
                    parallel_action = fire_direction * direction * intensity
                    parallel_action = np.clip(parallel_action, -1, 1)
                    candidates.append(parallel_action)
        
        # Enhanced scoring with more sophisticated reward prediction
        scores = []
        for a in candidates:
            # Simulate next state (only position part)
            sim_pos = np.clip(obs_np[:3] + a * self.max_step_size, [0, 0, 5], [self.area_size, self.area_size, 10])
            
            # Compute fire_below for simulated pos
            def point_to_segment_dist(p, a, b):
                ap = p - a
                ab = b - a
                t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-8)
                t = np.clip(t, 0, 1)
                closest = a + t * ab
                return np.linalg.norm(p - closest)
            
            drone_xy0 = np.array([sim_pos[0], sim_pos[1], 0.0])
            fire_a = np.array([self.fire_line[0][0], self.fire_line[0][1], 0.0])
            fire_b = np.array([self.fire_line[1][0], self.fire_line[1][1], 0.0])
            dist_below = point_to_segment_dist(drone_xy0, fire_a, fire_b)
            fire_below = 1.0 if dist_below <= (self.fire_radius + 2) else 0.0
            sim_obs = np.concatenate([sim_pos, [fire_below], obs_np[4:7]])
            
            # Simulate immediate reward using the new reward function
            dist = point_to_segment_dist(sim_pos, self.fire_line[0], self.fire_line[1])
            min_safe_dist = self.drone_radius + self.fire_radius + self.safety_margin
            collision_dist = self.drone_radius + self.fire_radius

            if dist <= collision_dist:
                immediate_reward = COLLISION_PENALTY
            elif dist <= min_safe_dist:
                danger_factor = (min_safe_dist - dist) / (min_safe_dist - collision_dist)
                immediate_reward = - (PROXIMITY_SCALE * 2.0) * (danger_factor ** 2)
            else:
                safe_distance = dist - min_safe_dist
                alpha = 1.2
                proximity_scale = PROXIMITY_SCALE
                immediate_reward = proximity_scale * np.exp(-alpha * safe_distance)

                # Add small optimal range bonus
                optimal_range = 0.3
                if safe_distance <= optimal_range:
                    optimal_bonus = 20 * (1 - safe_distance / optimal_range)
                    immediate_reward += optimal_bonus
            
            # Long-term Q (average over critics)
            obs_t = torch.tensor(sim_obs, dtype=torch.float32).unsqueeze(0)
            act_t = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
            q_vals = [critic(obs_t, act_t).item() for critic in self.critics]
            avg_q = np.mean(q_vals)
            
            # Estimate candidate energy (squared delta of action magnitude relative to zero)
            est_energy = float(np.sum(a ** 2) / max(1, a.size))
            # Weighted score: immediate reward + discounted Q - small energy penalty
            energy_penalty_est = ENERGY_WEIGHT * est_energy * 1.0  # stronger estimated penalty
            score = immediate_reward + self.gamma * avg_q - energy_penalty_est
            scores.append(score)
        
        best_idx = int(np.argmax(scores))
        # Pick best candidate (avoid averaging that cancels directional components)
        try:
            return np.clip(np.array(candidates[best_idx], dtype=np.float32), -1, 1)
        except Exception:
            return np.clip(candidates[best_idx], -1, 1)
    def store(self, o, a, r, no, d):
        # Store raw clipped reward in buffer. We will normalize on-the-fly in train()
        try:
            raw = float(r)
            clipped = float(np.clip(raw, -REWARD_CLIP, REWARD_CLIP))
        except Exception:
            clipped = float(r)

        # Add raw clipped reward to replay buffer
        self.buffer.add(o, a, clipped, no, d)

        # If we have not yet initialized running stats and the buffer now has enough
        # transitions, initialize Welford moments from buffer contents
        max_i = self.buffer.size if self.buffer.full else self.buffer.ptr
        if self.r_count == 0 and max_i >= N_WARMUP_TRANSITIONS:
            data = np.array(self.buffer.rew[:max_i], dtype=np.float64)
            if data.size == 0:
                # fallback to the single clipped sample
                data = np.array([clipped], dtype=np.float64)
            self.r_count = int(data.size)
            self.r_mean = float(np.mean(data))
            self.r_M2 = float(np.sum((data - self.r_mean) ** 2))

        # If running stats are initialized, update them with the new raw clipped reward
        if self.r_count > 0:
            x = float(clipped)
            self.r_count += 1
            if self.r_count == 1:
                self.r_mean = x
                self.r_M2 = 0.0
            else:
                delta = x - self.r_mean
                self.r_mean += delta / self.r_count
                delta2 = x - self.r_mean
                self.r_M2 += delta * delta2
            if self.r_count > 1:
                r_std = (self.r_M2 / (self.r_count - 1)) ** 0.5
            else:
                r_std = 1.0
            try:
                last_norm = float((clipped - self.r_mean) / (r_std + 1e-8))
                last_norm = float(np.clip(last_norm, -NORMALIZED_REWARD_CLIP, NORMALIZED_REWARD_CLIP))
                self.last_normalized_reward = last_norm
            except Exception:
                self.last_normalized_reward = float('nan')
        else:
            # Not normalizing yet; report clipped raw reward for logging
            self.last_normalized_reward = float(clipped)
    def train(self):
        # Require sufficient buffer fill and warmup transitions before training
        max_i = self.buffer.size if self.buffer.full else self.buffer.ptr
        if max_i < max(self.batch_size * 10, N_WARMUP_TRANSITIONS):
            return
        self.total_it += 1
        o, a, r_raw, no, d = self.buffer.sample(self.batch_size)
        # Normalize rewards on-the-fly if running stats are available
        with torch.no_grad():
            if self.r_count > 1:
                r_mean = torch.tensor(self.r_mean, dtype=torch.float32)
                r_std = torch.tensor((self.r_M2 / (self.r_count - 1)) ** 0.5, dtype=torch.float32)
                r = (r_raw - r_mean) / (r_std + 1e-8)
                r = torch.clamp(r, -float(NORMALIZED_REWARD_CLIP), float(NORMALIZED_REWARD_CLIP))
            else:
                r = r_raw
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_a = self.actor_target(no)
            next_a = (next_a + noise).clamp(-1, 1)
            # Target Q
            target_qs = [ct(no, next_a) for ct in self.critics_target]
            min_target_q = torch.min(torch.stack(target_qs, dim=0), dim=0)[0]
            target = r + self.gamma * (1 - d) * min_target_q
            # Clamp targets to avoid extreme updates
            target = torch.clamp(target, -float(TARGET_CLIP), float(TARGET_CLIP))
        # record last normalized reward for diagnostics
        try:
            self.last_normalized_reward = float(r.mean().item())
        except Exception:
            self.last_normalized_reward = float('nan')
        # Critic update
        critic_losses = []
        td_errors = []
        q_mins = []
        q_maxs = []
        critic_grad_norms = []
        for i, critic in enumerate(self.critics):
            q = critic(o, a)
            # TD-error per sample (target - q)
            with torch.no_grad():
                td = (target - q)
                # store mean absolute td-error for this critic
                td_errors.append(float(td.abs().mean().item()))
                q_mins.append(float(q.min().item()))
                q_maxs.append(float(q.max().item()))
            # Use Huber (Smooth L1) for robustness to outliers
            loss = nn.SmoothL1Loss()(q, target)
            critic_losses.append(loss.item())
            self.critic_opts[i].zero_grad()
            loss.backward()
            # Gradient clipping to avoid huge updates and record grad norm
            try:
                gn = torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP_NORM)
                critic_grad_norms.append(float(gn))
            except Exception:
                critic_grad_norms.append(float('nan'))
            self.critic_opts[i].step()
        # store last critic/td diagnostics for logging
        try:
            self.last_critic_loss = float(sum(critic_losses) / len(critic_losses))
        except Exception:
            self.last_critic_loss = float('nan')
        try:
            self.last_td_error = float(sum(td_errors) / len(td_errors))
        except Exception:
            self.last_td_error = float('nan')
        try:
            self.last_q_min = float(sum(q_mins) / len(q_mins))
            self.last_q_max = float(sum(q_maxs) / len(q_maxs))
        except Exception:
            self.last_q_min = float('nan')
            self.last_q_max = float('nan')
        try:
            # Average critic grad norm for logging
            self.last_critic_grad_norm = float(np.nanmean(critic_grad_norms)) if critic_grad_norms else float('nan')
        except Exception:
            self.last_critic_grad_norm = float('nan')
        # Delayed policy update
        if self.total_it % self.policy_delay == 0:
            # Actor loss with small L2 penalty on actions to discourage saturation
            pred_actions = self.actor(o)
            actor_loss = -self.critics[0](o, pred_actions).mean()
            actor_loss = actor_loss + ACT_REG * (pred_actions ** 2).mean()
            # Small actor-side energy regularizer (mean absolute action magnitude)
            try:
                actor_loss = actor_loss + ENERGY_ACT_REG * torch.mean(torch.abs(pred_actions))
            except Exception:
                pass
            self.actor_opt.zero_grad()
            actor_loss.backward()
            # Clip actor grads too
            try:
                agn = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP_NORM)
                self.last_actor_grad_norm = float(agn)
            except Exception:
                self.last_actor_grad_norm = float('nan')
            self.actor_opt.step()
            # store last actor loss
            try:
                self.last_actor_loss = float(actor_loss.item())
            except Exception:
                self.last_actor_loss = float('nan')
            # Polyak averaging
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for i in range(len(self.critics)):
                for param, target_param in zip(self.critics[i].parameters(), self.critics_target[i].parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# --- MAIN LOOP ---
def main():
    import os
    scenario_names = {1: "Static Fire Line", 3: "Line of Fires", 2: "Spreading Fire", 4: "Random Fires", 5: "Impassable Fire Wall (w/ Random Crash)", 6: "Central Fire"}
    all_scenarios = [1, 3, 2, 4, 5, 6]
    print("Available scenarios:")
    for s in all_scenarios:
        print(f"  {s}: {scenario_names[s]}")
    user_input = input("Enter scenario numbers to run, separated by commas (e.g., 1,3,4,1): ").strip()
    if user_input:
        try:
            scenarios = [int(x) for x in user_input.split(',') if int(x) in all_scenarios]
            if not scenarios:
                print("No valid scenarios selected. Running all by default.")
                scenarios = all_scenarios
        except Exception:
            print("Invalid input. Running all scenarios by default.")
            scenarios = all_scenarios
    else:
        scenarios = all_scenarios
    # Ask user if they want to see visualization at the end of each scenario
    vis_input = input("Show visualization at the end of each scenario? (y/n): ").strip().lower()
    show_vis_each = vis_input == 'y'
    curriculum_episodes = 1  # Episodes per curriculum level zzz

    episode_rewards = [[], []]  # Store total points per episode for each drone
    avg_fire_distances = [[], []]  # Track average distance to fire per episode

    # Initialize agents ONCE so they retain learning across scenarios
    env = DroneEnv(curriculum_level=0, scenario=1)
    obs_dim = env.observation_space.shape[1]
    act_dim = env.action_space.shape[1]
    agents = [TD3Agent(
        area_size=env.area_size,
        drone_radius=env.drone_radius,
        fire_line=env.fire_line,
        fire_radius=env.fire_radius,
        safety_margin=env.safety_margin,
        max_step_size=env.max_step_size,
        obs_dim=obs_dim,
        act_dim=act_dim
    ) for _ in range(env.n_drones)]

    # Per-component normalizer for logging and optional normalization
    reward_normalizer = RewardNormalizer(warmup=N_WARMUP_TRANSITIONS, clip=3.0)

    # Prepare logs directory and CSV files (overwrite existing logs for a fresh run)
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    training_csv_path = logs_dir / "training_metrics.csv"
    steps_csv_path = logs_dir / "steps.csv"
    # Open and write headers
    training_fp = open(training_csv_path, 'w', newline='')
    training_writer = csv.writer(training_fp)
    training_writer.writerow(["episode", "total_reward", "length", "success", "collisions", "avg_goal_dist", "crash_types", "proximity_component", "exploration_component", "energy_component", "normalized_total", "norm_prox_mean", "norm_expl_mean", "norm_energy_mean", "action_saturation_fraction", "actor_loss", "critic_loss", "td_error"])

    steps_fp = open(steps_csv_path, 'w', newline='')
    steps_writer = csv.writer(steps_fp)
    # Add per-step components: exploration_reward, proximity_reward and per-step td_error
    steps_writer.writerow(["episode", "step", "x", "y", "z", "action0", "action1", "action2", "reward", "exploration_reward", "proximity_reward", "energy", "exploration_norm", "proximity_norm", "energy_norm", "normalized_reward", "td_error", "action_saturation", "rule_override"])

    # Per-update diagnostics (agent training updates)
    updates_csv_path = logs_dir / "updates.csv"
    updates_fp = open(updates_csv_path, 'w', newline='')
    updates_writer = csv.writer(updates_fp)
    # include normalized reward and grad norms for debugging
    updates_writer.writerow(["episode", "step", "agent_id", "total_it", "actor_loss", "critic_loss", "td_error", "q_min", "q_max", "normalized_reward", "actor_grad_norm", "critic_grad_norm", "replay_fill"])

    global_episode_counter = 0
    # global step counter for warm-up and scheduling
    global_step = 0
    # number of environment steps to populate replay buffer with random actions before training
    start_timesteps = 5000
    # reward scaling for training (we now use agent-side normalization, so set to 1.0)
    reward_scale = 1.0

    # Ask user if they want to load previous training or start from scratch
    load_choice = input("Load previous training from pickle file? (y/n): ").strip().lower()
    if load_choice == 'y' and os.path.exists("td3_agent_0.pkl"):
        agents[0].load("td3_agent_0.pkl")
        # Copy agent 0's weights to agent 1
        agents[1].actor.load_state_dict(agents[0].actor.state_dict())
        agents[1].actor_target.load_state_dict(agents[0].actor_target.state_dict())
        for i in range(len(agents[1].critics)):
            agents[1].critics[i].load_state_dict(agents[0].critics[i].state_dict())
            agents[1].critics_target[i].load_state_dict(agents[0].critics_target[i].state_dict())
        agents[1].actor_opt.load_state_dict(agents[0].actor_opt.state_dict())
        for i in range(len(agents[1].critic_opts)):
            agents[1].critic_opts[i].load_state_dict(agents[0].critic_opts[i].state_dict())
        agents[1].buffer.__dict__.update(agents[0].buffer.__dict__)
        print("Loaded agent 0 and copied weights to agent 1.")
    else:
        print("Training will start from scratch.")

    for scenario_idx, scenario in enumerate(scenarios):
        print(f"\n=== SCENARIO {scenario}: {scenario_names[scenario]} ===")
        last_episode_states = None
        last_episode_fire_centers = None
        for curriculum_level in range(5):  # 5 curriculum levels per scenario
            print(f"  --- CURRICULUM LEVEL {curriculum_level + 1}/5 ---")
            env = DroneEnv(curriculum_level=curriculum_level, scenario=scenario)
            print(f"  Safety margin for this level: {env.safety_margin:.2f}")
            for episode in range(curriculum_episodes):
                obs, _ = env.reset()
                done = [False, False]
                total_reward = [0, 0]
                fire_distances = [[], []]  # Track fire distances during episode
                steps = 0
                crash_type = [None, None]  # Track crash reason for each drone

                # Track seen areas for each drone (list of (x, y, radius))
                seen_areas = [[], []]
                # Track seen grid cells for exploration reward
                seen_grids = [set(), set()]

                # --- Store episode-level and step-level data for logging/visualization ---
                episode_states = []
                episode_fire_centers = []
                episode_rewards_per_step = []
                episode_dones_per_step = []
                episode_actions = []
                rewards = [0, 0]  # Initialize rewards before first step

                while not all(done) and steps < 300:
                    actions = np.zeros((env.n_drones, 3))
                    for i in range(env.n_drones):
                        if not done[i]:
                            # During warmup choose random actions to fill buffer
                            if global_step < start_timesteps:
                                actions[i] = np.random.uniform(-1, 1, size=(3,))
                            else:
                                actions[i] = agents[i].select_action(obs[i])

                    # Store action and state before step for logging/visualization
                    episode_actions.append(np.copy(actions))
                    episode_states.append(np.copy(obs[:, :3]))

                    next_obs, rewards, done, _, info = env.step(actions, seen_grids=seen_grids)

                    # Store post-step info
                    if env.scenario == 2:
                        episode_fire_centers.append([fc.copy() for fc in env.fire_centers])
                    else:
                        episode_fire_centers.append(None)
                    episode_rewards_per_step.append(list(rewards))
                    episode_dones_per_step.append(list(done))

                    # Track fire distances
                    def point_to_segment_dist(p, a, b):
                        ap = p - a
                        ab = b - a
                        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-8)
                        t = np.clip(t, 0, 1)
                        closest = a + t * ab
                        return np.linalg.norm(p - closest)

                    for i in range(env.n_drones):
                        if not done[i]:
                            if env.scenario == 2:
                                # Closest fire center
                                dists = [np.linalg.norm(next_obs[i][:3] - fc) for fc in env.fire_centers]
                                dist_to_fire = min(dists)
                            else:
                                dist_to_fire = point_to_segment_dist(next_obs[i][:3], env.fire_line[0], env.fire_line[1])
                            fire_distances[i].append(dist_to_fire)

                    # Record seen area for each drone
                    k = 1.0  # must match _get_obs and render
                    for i in range(env.n_drones):
                        pos = next_obs[i]
                        drone_xy = np.array([pos[0], pos[1]])
                        z = pos[2]
                        view_radius = k * z / 8
                        seen_areas[i].append((drone_xy[0], drone_xy[1], view_radius))

                    for i in range(env.n_drones):
                        # Clamp raw reward and pass clipped raw reward to agent (agent will normalize)
                        raw_r = float(rewards[i])
                        r_clamped = float(np.clip(raw_r, -REWARD_CLIP, REWARD_CLIP))
                        r_for_buffer = r_clamped * reward_scale
                        # Update per-component normalizer if env returned breakdown
                        try:
                            prox_val = float(info.get('proximity_contrib', [float('nan')])[i]) if info is not None else float('nan')
                            expl_val = float(info.get('exploration_contrib', [float('nan')])[i]) if info is not None else float('nan')
                            energy_val = float(info.get('energy', [float('nan')])[i]) if info is not None else float('nan')
                        except Exception:
                            prox_val = expl_val = energy_val = float('nan')
                        # Update running stats
                        try:
                            reward_normalizer.update(prox_val, expl_val, energy_val)
                        except Exception:
                            pass
                        # If the per-component normalizer is ready, use normalized components to form a scaled reward for buffer
                        normalized_total = None
                        exploration_norm = proximity_norm = energy_norm = float('nan')
                        try:
                            if reward_normalizer.ready():
                                p_n, e_n, en_n = reward_normalizer.normalize(prox_val, expl_val, energy_val)
                                # Keep consistent naming: prox, expl, energy
                                proximity_norm = float(p_n)
                                exploration_norm = float(e_n)
                                energy_norm = float(en_n)
                                # Recompose normalized total according to component weights
                                # Add small shaped bonus proportional to proximity (encourages moving closer to safe/optimal band)
                                shaped_goal_bonus = 0.8 * max(0.0, proximity_norm)
                                normalized_total = PROX_WEIGHT * proximity_norm + EXPL_WEIGHT * exploration_norm - ENERGY_WEIGHT * energy_norm + shaped_goal_bonus
                                # Clip normalized total to avoid extreme values
                                normalized_total = float(np.clip(normalized_total, -float(NORMALIZED_REWARD_CLIP), float(NORMALIZED_REWARD_CLIP)))
                        except Exception:
                            normalized_total = None

                        # Decide what to store in buffer: prefer normalized_total when available, else raw clipped reward
                        if normalized_total is not None:
                            r_buffer = float(normalized_total)
                        else:
                            r_buffer = r_for_buffer

                        # Store the reward used for training in the agent's buffer
                        agents[i].store(obs[i], actions[i], r_buffer, next_obs[i], float(done[i]))
                        # Attempt a training step; agent.train() will gate on warmup internally
                        agents[i].train()
                        # Write per-update diagnostics if the agent produced them
                        try:
                            updates_writer.writerow([
                                global_episode_counter,
                                steps,
                                i,
                                agents[i].total_it,
                                getattr(agents[i], 'last_actor_loss', float('nan')),
                                getattr(agents[i], 'last_critic_loss', float('nan')),
                                getattr(agents[i], 'last_td_error', float('nan')),
                                getattr(agents[i], 'last_q_min', float('nan')),
                                getattr(agents[i], 'last_q_max', float('nan')),
                                getattr(agents[i], 'last_normalized_reward', float('nan')),
                                getattr(agents[i], 'last_actor_grad_norm', float('nan')),
                                getattr(agents[i], 'last_critic_grad_norm', float('nan')),
                                getattr(agents[i].buffer, 'ptr', float('nan'))
                            ])
                            try:
                                updates_fp.flush()
                            except Exception:
                                pass
                        except Exception:
                            pass
                        # Use the clamped raw reward for human-readable episode totals, but also track normalized totals
                        total_reward[i] += r_clamped
                        if normalized_total is not None:
                            # track a separate normalized episode total (create attribute if needed)
                            if not hasattr(agents[i], 'episode_normalized_total'):
                                agents[i].episode_normalized_total = 0.0
                            agents[i].episode_normalized_total += float(normalized_total)
                        # store per-step normalized comps for later CSV logging
                        try:
                            if 'episode_normalized_components' not in locals():
                                episode_normalized_components = [[] for _ in range(env.n_drones)]
                        except Exception:
                            episode_normalized_components = [[] for _ in range(env.n_drones)]
                        try:
                            episode_normalized_components[i].append((exploration_norm, proximity_norm, energy_norm, normalized_total))
                        except Exception:
                            episode_normalized_components[i].append((float('nan'), float('nan'), float('nan'), float('nan')))
                        # If env provided component breakdown in info, record for logging
                        try:
                            # info may contain 'exploration_contrib', 'proximity_contrib', and 'energy'
                            if isinstance(info, dict):
                                if 'exploration_contrib' in info:
                                    # per-drone contributions list
                                    exploration_val = float(info['exploration_contrib'][i]) if len(info['exploration_contrib']) > i else float('nan')
                                else:
                                    exploration_val = float('nan')
                                if 'proximity_contrib' in info:
                                    proximity_val = float(info['proximity_contrib'][i]) if len(info['proximity_contrib']) > i else float('nan')
                                else:
                                    proximity_val = float('nan')
                                if 'energy' in info:
                                    energy_val = float(info['energy'][i]) if len(info['energy']) > i else float('nan')
                                else:
                                    energy_val = float('nan')
                            else:
                                exploration_val = proximity_val = energy_val = float('nan')
                        except Exception:
                            exploration_val = proximity_val = energy_val = float('nan')
                        # Detect crash type with clearer messages
                        if done[i] and crash_type[i] is None:
                            if env.scenario == 5:
                                if rewards[i] <= COLLISION_PENALTY:
                                    # Check if random crash or wall
                                    prev_x = obs[i][0]
                                    new_x = next_obs[i][0]
                                    wall_x = env.fire_wall_x
                                    # If drone is on opposite side of wall after step, it's a wall crash
                                    if (prev_x < wall_x and new_x >= wall_x) or (prev_x > wall_x and new_x <= wall_x):
                                        crash_type[i] = 'crashed into fire wall (x={:.2f})'.format(wall_x)
                                    else:
                                        crash_type[i] = 'random ground crash (1% chance)'
                            else:
                                if rewards[i] <= COLLISION_PENALTY:
                                    # Check if crashed into fire or other drone
                                    if done[0] and done[1] and np.linalg.norm(next_obs[0][:3] - next_obs[1][:3]) < 2*env.drone_radius + env.safety_margin + 1e-3:
                                        crash_type[0] = crash_type[1] = 'drone-drone collision'
                                    else:
                                        crash_type[i] = 'fire collision'

                    obs = next_obs
                    steps += 1
                    global_step += 1

                # Save last episode's states for visualization
                if curriculum_level == 4 and episode == curriculum_episodes - 1:
                    last_episode_states = episode_states
                    last_episode_fire_centers = episode_fire_centers

                # Calculate average fire distances for this episode
                for i in range(env.n_drones):
                    if crash_type[i] is None and done[i]:
                        crash_type[i] = 'timeout'
                    episode_rewards[i].append(total_reward[i])
                    if fire_distances[i]:
                        avg_fire_distances[i].append(np.mean(fire_distances[i]))
                    else:
                        avg_fire_distances[i].append(float('inf'))  # No distance recorded

                # Episode number for this scenario/curriculum
                scenario_episode = curriculum_level * curriculum_episodes + episode + 1
                # Increment global episode counter and log
                global_episode_counter += 1

                # Periodic deterministic evaluation (every 5 episodes)
                if global_episode_counter % 5 == 0:
                    # Run a single deterministic episode for agents and log
                    eval_env = DroneEnv(curriculum_level=curriculum_level, scenario=scenario)
                    eval_obs, _ = eval_env.reset()
                    eval_done = [False, False]
                    eval_steps = 0
                    eval_total = [0.0, 0.0]
                    while not all(eval_done) and eval_steps < 300:
                        eval_actions = np.zeros((eval_env.n_drones, 3))
                        for ai in range(eval_env.n_drones):
                            if not eval_done[ai]:
                                eval_actions[ai] = agents[ai].select_action(eval_obs[ai], deterministic=True)
                        eval_next_obs, eval_rewards, eval_done, _, _ = eval_env.step(eval_actions)
                        for ai in range(eval_env.n_drones):
                            eval_total[ai] += float(np.clip(eval_rewards[ai], -REWARD_CLIP, REWARD_CLIP))
                        eval_obs = eval_next_obs
                        eval_steps += 1
                    # Write eval metrics
                    eval_path = logs_dir / "eval_metrics.csv"
                    write_header = not eval_path.exists()
                    with open(eval_path, 'a', newline='') as efp:
                        ew = csv.writer(efp)
                        if write_header:
                            ew.writerow(["episode", "scenario", "curriculum_level", "eval_steps", "eval_total_reward", "eval_avg_reward", "eval_success"])
                        ev_success = 1 if sum(1 for r in eval_total if r > COLLISION_PENALTY) == eval_env.n_drones else 0
                        ew.writerow([global_episode_counter, scenario, curriculum_level, eval_steps, float(np.mean(eval_total)), float(np.mean(eval_total)), ev_success])

                # Determine success & collisions heuristically. Use crash_type where available.
                collisions = 0
                for i in range(env.n_drones):
                    ct = crash_type[i]
                    if ct is not None and ('collision' in ct or 'crash' in ct or 'fire' in ct or 'wall' in (ct or '')):
                        collisions += 1
                    else:
                        # Fallback: if total_reward very low below collision penalty, count as collision
                        if total_reward[i] <= COLLISION_PENALTY:
                            collisions += 1
                success = 1 if collisions == 0 else 0
                # Average goal distance (use avg_fire_distances if available)
                try:
                    avg_goal_dist = np.nanmean([fire_distances[i][-1] if fire_distances[i] else np.nan for i in range(env.n_drones)])
                except Exception:
                    avg_goal_dist = float('nan')

                # If the episode ended due to timeout, apply the timeout penalty once to the total reward
                for i in range(env.n_drones):
                    if crash_type[i] == 'timeout':
                        # Subtract timeout penalty once (timeouts are less severe than collisions)
                        episode_rewards[i] = [r for r in episode_rewards[i]]
                        total_reward[i] += TIMEOUT_PENALTY

                # Include last actor/critic/td diagnostics if available
                actor_loss_log = float('nan')
                critic_loss_log = float('nan')
                td_error_log = float('nan')
                try:
                    # Use agent 0 diagnostics as representative
                    actor_loss_log = agents[0].last_actor_loss if hasattr(agents[0], 'last_actor_loss') else float('nan')
                    critic_loss_log = agents[0].last_critic_loss if hasattr(agents[0], 'last_critic_loss') else float('nan')
                    td_error_log = agents[0].last_td_error if hasattr(agents[0], 'last_td_error') else float('nan')
                except Exception:
                    pass

                # Aggregate last-step approximate components for logging (best-effort)
                prox_comp = float(np.nan)
                expl_comp = float(np.nan)
                try:
                    # If we recorded proximity/exploration contributions per step in env info, use last
                    last_info = None
                    # Some code paths append per-step info to episode_rewards_per_step as tuples
                    if episode_rewards_per_step and isinstance(episode_rewards_per_step[-1][0], (list, tuple)):
                        prox_comp = float(episode_rewards_per_step[-1][0][0])
                        expl_comp = float(episode_rewards_per_step[-1][0][1])
                except Exception:
                    pass
                # Energy component: best-effort extract from last info stored in episode arrays if available
                energy_comp = float(np.nan)
                try:
                    if episode_rewards_per_step and len(episode_rewards_per_step[-1]) >= 3:
                        # If last entry contains breakdown (prox, expl, energy), try to use it
                        last_entry = episode_rewards_per_step[-1][0]
                        if isinstance(last_entry, (list, tuple)) and len(last_entry) >= 3:
                            energy_comp = float(last_entry[2])
                except Exception:
                    energy_comp = float('nan')

                # Aggregate normalized components per episode if available
                try:
                    norm_prox_sum = float('nan')
                    norm_expl_sum = float('nan')
                    norm_energy_sum = float('nan')
                    norm_total = float('nan')
                    if 'episode_normalized_components' in locals():
                        # Sum across steps for drone 0 only (consistent with other per-step logging)
                        vals = episode_normalized_components[0]
                        if vals:
                            arr = np.array(vals, dtype=np.float32)
                            # columns: exploration_norm, proximity_norm, energy_norm, normalized_total
                            norm_expl_sum = float(np.nansum(arr[:, 0]))
                            norm_prox_sum = float(np.nansum(arr[:, 1]))
                            norm_energy_sum = float(np.nansum(arr[:, 2]))
                            norm_total = float(np.nansum(arr[:, 3]))
                except Exception:
                    norm_prox_sum = norm_expl_sum = norm_energy_sum = norm_total = float('nan')

                # Compute per-episode action saturation fraction (fraction of steps where any action dim is near ±1)
                try:
                    sats = []
                    for t in range(len(episode_actions)):
                        act = episode_actions[t][0] if t < len(episode_actions) else np.array([0.0, 0.0, 0.0])
                        sats.append(float(np.mean(np.abs(act) > 0.98)))
                    action_saturation_fraction = float(np.nanmean(sats)) if sats else float('nan')
                except Exception:
                    action_saturation_fraction = float('nan')

                # Compute mean normalized components over episode for drone 0
                try:
                    norm_prox_mean = float('nan')
                    norm_expl_mean = float('nan')
                    norm_energy_mean = float('nan')
                    if 'episode_normalized_components' in locals():
                        vals = episode_normalized_components[0]
                        if vals:
                            arr = np.array(vals, dtype=np.float32)
                            norm_expl_mean = float(np.nanmean(arr[:, 0]))
                            norm_prox_mean = float(np.nanmean(arr[:, 1]))
                            norm_energy_mean = float(np.nanmean(arr[:, 2]))
                except Exception:
                    norm_prox_mean = norm_expl_mean = norm_energy_mean = float('nan')

                training_writer.writerow([global_episode_counter, float(np.mean(total_reward)), int(steps), success, int(collisions), float(avg_goal_dist), ";".join([str(ct) for ct in crash_type]), prox_comp, expl_comp, energy_comp, norm_total, norm_prox_mean, norm_expl_mean, norm_energy_mean, action_saturation_fraction, actor_loss_log, critic_loss_log, td_error_log])

                # Write per-step rows (log only drone 0 for compactness)
                # When writing per-step rows, attempt to split reward into components if possible.
                for t in range(len(episode_states)):
                    st = episode_states[t][0]
                    act = episode_actions[t][0] if t < len(episode_actions) else np.array([0.0, 0.0, 0.0])
                    reward_step = episode_rewards_per_step[t][0] if t < len(episode_rewards_per_step) else 0.0
                    # Try to estimate exploration vs proximity component if we recorded seen grids or computed them
                    exploration_comp = float('nan')
                    proximity_comp = float('nan')
                    # If env provided a breakdown in episode_rewards_per_step as tuples, handle that
                    try:
                        # Some code paths may have stored a tuple (prox, expl)
                        if isinstance(episode_rewards_per_step[t][0], (list, tuple)) and len(episode_rewards_per_step[t][0]) >= 2:
                            proximity_comp = float(episode_rewards_per_step[t][0][0])
                            exploration_comp = float(episode_rewards_per_step[t][0][1])
                    except Exception:
                        pass

                    # As a safety, clamp the recorded per-step reward to a reasonable range to avoid huge episode sums
                    reward_step_clamped = float(np.clip(reward_step, -REWARD_CLIP, REWARD_CLIP))
                    # Fill td_error column from agent diagnostics if available (use agent 0 as representative)
                    td_error_val = float('nan')
                    try:
                        td_error_val = agents[0].last_td_error if hasattr(agents[0], 'last_td_error') else float('nan')
                    except Exception:
                        td_error_val = float('nan')
                    # extract per-step energy component if env provided it in episode arrays (info not stored per-step by default)
                    energy_val = float('nan')
                    try:
                        # if episode_rewards_per_step stored tuples (prox, expl) or info was preserved, try to extract energy
                        if t < len(episode_rewards_per_step):
                            # attempt to read from episode_rewards_per_step's associated info if available
                            # otherwise, fallback to NaN
                            pass
                        # compute action_saturation metric for this step (fraction of action dims near ±1)
                        action_saturation = float(np.mean(np.abs(act) > 0.98))
                    except Exception:
                        action_saturation = float('nan')
                    # Pull per-step normalized components if available
                    expl_norm = prox_norm = energy_norm = norm_reward_step = float('nan')
                    try:
                        if 'episode_normalized_components' in locals() and t < len(episode_normalized_components[0]):
                            en = episode_normalized_components[0][t]
                            expl_norm = float(en[0])
                            prox_norm = float(en[1])
                            energy_norm = float(en[2])
                            norm_reward_step = float(en[3])
                    except Exception:
                        expl_norm = prox_norm = energy_norm = norm_reward_step = float('nan')

                    steps_writer.writerow([global_episode_counter, t, float(st[0]), float(st[1]), float(st[2]), float(act[0]), float(act[1]), float(act[2]), reward_step_clamped, exploration_comp, proximity_comp, energy_val, expl_norm, prox_norm, energy_norm, norm_reward_step, td_error_val, action_saturation, 0])

                print(f"Episode {scenario_episode}/{5 * curriculum_episodes} (Scenario {scenario}, Level {curriculum_level+1}) Total Rewards: {[round(r, 1) for r in total_reward]}")
                for i in range(env.n_drones):
                    avg_dist = avg_fire_distances[i][-1] if avg_fire_distances[i] else float('inf')
                    print(f"  Drone {i+1}: Total Points = {total_reward[i]:.1f}, Avg Fire Distance = {avg_dist:.2f}, Ended by: {crash_type[i]}")

                # Decay noise for all agents after each episode
                for agent in agents:
                    agent.decay_noise()

        # --- Play episode visualization after scenario ends ---
        is_last_scenario = (scenario_idx == len(scenarios) - 1)
        if (show_vis_each) or is_last_scenario:
            input("Press Enter to play episode visualization...")
            print("Playing episode visualization...")
            if last_episode_states is not None:
                for t in range(len(last_episode_states)):
                    print(f"Step {t}: Drone positions: {np.array(last_episode_states[t]).round(2)}")
                for t in range(len(last_episode_states)):
                    env.render(drone_pos=last_episode_states[t], fire_centers=last_episode_fire_centers[t])
    
    # Remove non-blocking show here to avoid closing the stats plot
    # Plot total points per episode for each drone
    plt.ioff()  # Disable interactive mode for the stats plot
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Total rewards
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards[0], label='Drone 1', alpha=0.7)
    plt.plot(episode_rewards[1], label='Drone 2', alpha=0.7)
    # Add vertical lines for curriculum level changes
    for i in range(1, 5):
        plt.axvline(x=i*curriculum_episodes, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Total Points')
    plt.title('Total Points per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Average fire distances
    plt.subplot(1, 3, 2)
    # Filter out infinite values for plotting
    dist1_filtered = [d for d in avg_fire_distances[0] if d != float('inf')]
    dist2_filtered = [d for d in avg_fire_distances[1] if d != float('inf')]
    plt.plot(range(len(dist1_filtered)), dist1_filtered, label='Drone 1', alpha=0.7)
    plt.plot(range(len(dist2_filtered)), dist2_filtered, label='Drone 2', alpha=0.7)
    # Add horizontal line for optimal distance
    optimal_dist = min(env.drone_radius + env.fire_radius + env.base_safety_margin + 0.15, 1.0)
    plt.axhline(y=optimal_dist, color='green', linestyle=':', label=f'Optimal Distance ({optimal_dist:.2f})')
    # Add vertical lines for curriculum level changes
    for i in range(1, 5):
        plt.axvline(x=i*curriculum_episodes, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Average Distance to Fire')
    plt.title('Average Fire Distance per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Combined performance metric
    plt.subplot(1, 3, 3)
    # Calculate a combined score: high reward + low distance = good performance
    combined_scores = []
    for i in range(len(episode_rewards[0])):
        reward_norm = (episode_rewards[0][i] + episode_rewards[1][i]) / 2  # Average reward
        if i < len(avg_fire_distances[0]) and avg_fire_distances[0][i] != float('inf'):
            dist_penalty = max(0, avg_fire_distances[0][i] - optimal_dist) * 50  # Penalty for being too far
            combined_score = reward_norm - dist_penalty
        else:
            combined_score = reward_norm - 100  # High penalty for crashes
        combined_scores.append(combined_score)
    
    plt.plot(combined_scores, color='purple', linewidth=2, label='Combined Performance')
    # Add vertical lines for curriculum level changes
    for i in range(1, 5):
        plt.axvline(x=i*curriculum_episodes, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Combined Performance Score')
    plt.title('Combined Performance (Reward - Distance Penalty)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()  # Blocking show for the stats plot

    # Plot union of all seen areas for each drone (last episode)
    colors = ['blue', 'green']
    plt.figure()
    ax = plt.gca()
    for i in range(env.n_drones):
        first = True
        for (x, y, r) in seen_areas[i]:
            if first:
                circle = plt.Circle((x, y), r, color=colors[i], alpha=0.1, label=f'Drone {i+1}')
                first = False
            else:
                circle = plt.Circle((x, y), r, color=colors[i], alpha=0.1)
            ax.add_patch(circle)
    plt.xlim(0, env.area_size)
    plt.ylim(0, env.area_size)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Union of All Seen Areas (Last Episode)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    # --- Plot Exploration Reward Table (cell_rewards) ---
    plt.figure(figsize=(6, 5))
    plt.imshow(env.cell_rewards.T, origin='lower', cmap='YlOrRd', vmin=0, vmax=env.max_cell_reward)
    plt.title('Exploration Reward Table (cell_rewards)')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.colorbar(label='Cell Reward Value')
    plt.tight_layout()
    plt.show()

    # --- Integrated Reward System Heatmap (for last episode) ---
    # Use the last env state (cell_rewards, fire_centers, etc.) from the final episode
    grid_size = env.grid_size
    grid_dim = env.grid_dim
    area_size = env.area_size
    fire_centers = env.fire_centers
    fire_radius = env.fire_radius
    drone_radius = env.drone_radius
    safety_margin = env.base_safety_margin  # Use base safety margin for analysis
    max_cell_reward = env.max_cell_reward
    cell_rewards = env.cell_rewards
    alpha = 1.2  # Updated to match new parameters
    proximity_scale = PROXIMITY_SCALE
    edge_penalty_scale = 50.0
    edge_penalty_alpha = 2.0
    z = 7.5  # fixed height for analysis
    integrated_reward = np.zeros((grid_dim, grid_dim))
    for gx in range(grid_dim):
        for gy in range(grid_dim):
            cx = gx * grid_size + grid_size/2
            cy = gy * grid_size + grid_size/2
            pos = np.array([cx, cy, z])
            # Proximity to closest fire (for current scenario)
            if hasattr(env, 'fire_centers') and env.fire_centers:
                dists = [np.linalg.norm(pos - fc) for fc in fire_centers]
                dist_fire = min(dists)
            else:
                # For scenario 1, use fire_line
                def point_to_segment_dist(p, a, b):
                    ap = p - a
                    ab = b - a
                    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-8)
                    t = np.clip(t, 0, 1)
                    closest = a + t * ab
                    return np.linalg.norm(p - closest)
                dist_fire = point_to_segment_dist(pos, env.fire_line[0], env.fire_line[1])
            min_safe_dist = drone_radius + fire_radius + safety_margin
            collision_dist = drone_radius + fire_radius
            if dist_fire <= collision_dist:
                fire_reward = -200
            elif dist_fire <= min_safe_dist:
                danger_factor = (min_safe_dist - dist_fire) / (min_safe_dist - collision_dist)
                fire_reward = -50 * (danger_factor ** 2)
            else:
                safe_distance = dist_fire - min_safe_dist
                fire_reward = proximity_scale * np.exp(-alpha * safe_distance)
                # Add optimal range bonus
                optimal_range = 0.3
                if safe_distance <= optimal_range:
                    optimal_bonus = OPTIMAL_BONUS * (1 - safe_distance / optimal_range)
                    fire_reward += optimal_bonus
            # Edge penalty (only when very close)
            dists_edge = [pos[0], area_size - pos[0], pos[1], area_size - pos[1], pos[2] - 5, 10 - pos[2]]
            min_dist_edge = min(dists_edge)
            if min_dist_edge < 1.0:
                edge_penalty = edge_penalty_scale * np.exp(-edge_penalty_alpha * min_dist_edge)
            else:
                edge_penalty = 0
            # Exploration reward (reduced impact)
            exploration_reward = cell_rewards[gx, gy] * 0.5
            # Total
            integrated_reward[gx, gy] = fire_reward - edge_penalty + exploration_reward
    plt.figure(figsize=(6, 5))
    plt.imshow(integrated_reward.T, origin='lower', cmap='coolwarm')
    plt.title('Integrated Reward System Heatmap (Last Episode)')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.colorbar(label='Total Reward Value')
    plt.tight_layout()
    plt.show()

    # --- Save trained agents ---
    def export_decision_map(agent, env, fname=logs_dir / "decision_map.npy", x_range=(2.5,7.5), y_range=(2.5,7.5), res=0.15):
        # Evaluate deterministic actor on grid (no candidate noise)
        xs = np.arange(x_range[0], x_range[1], res)
        ys = np.arange(y_range[0], y_range[1], res)
        grid = np.zeros((len(xs), len(ys), 3), dtype=np.float32)
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                # z fixed to 7.5 for decision map
                obs = np.array([x, y, 7.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                # deterministic actor
                with torch.no_grad():
                    act = agent.actor(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).cpu().numpy()[0]
                grid[ix, iy] = act
        # Save grid and axes for plotting
        np.save(fname, {'xs': xs, 'ys': ys, 'grid': grid})
        print(f"Wrote decision map to {fname}")

    save_input = input(f"Save trained agents to file? (y/n): ").strip().lower()
    if save_input == 'y':
        for i, agent in enumerate(agents):
            fname = f"td3_agent_{i}.pkl"
            if os.path.exists(fname):
                overwrite = input(f"File '{fname}' exists. Overwrite? (y/n): ").strip().lower()
                if overwrite != 'y':
                    print(f"Skipping save for agent {i}.")
                    continue
            agent.save(fname)
            print(f"Saved agent {i} to '{fname}'.")
            # Export decision map for agent 0 only (helpful for plotting)
            if i == 0:
                try:
                    export_decision_map(agent, env)
                except Exception as e:
                    print("Failed to export decision map:", e)

    # Close log files if open
    try:
        training_fp.close()
        steps_fp.close()
    except Exception:
        pass

    # Offer to generate analysis plots now
    plot_now = input("Generate analysis plots now from recorded logs? (y/n): ").strip().lower()
    if plot_now == 'y':
        try:
            # Call the analysis plotting script
            import runpy
            runpy.run_path("analysis/plot_all_figures.py", run_name="__main__")
        except Exception as e:
            print("Failed to run plotting script:", e)

if __name__ == "__main__":
    import os
    if os.environ.get('SMOKE_RUN', '0') == '1':
        # Minimal non-interactive smoke run for quick validation
        def smoke_run(episodes=10):
            env = DroneEnv(scenario=1)
            obs, _ = env.reset()
            agents = [TD3Agent(env.area_size, env.drone_radius, env.fire_line, env.fire_radius, env.safety_margin, env.max_step_size, obs_dim=env.observation_space.shape[1], act_dim=env.action_space.shape[1]) for _ in range(env.n_drones)]
            for ep in range(episodes):
                obs, _ = env.reset()
                done = [False] * env.n_drones
                steps = 0
                while not all(done) and steps < 200:
                    actions = np.zeros((env.n_drones, 3))
                    for i in range(env.n_drones):
                        if not done[i]:
                            actions[i] = agents[i].select_action(obs[i])
                    next_obs, rewards, done, _, info = env.step(actions)
                    for i in range(env.n_drones):
                        agents[i].store(obs[i], actions[i], float(np.clip(rewards[i], -REWARD_CLIP, REWARD_CLIP)), next_obs[i], float(done[i]))
                        agents[i].train()
                    obs = next_obs
                    steps += 1
            print('SMOKE_RUN_COMPLETE')
        smoke_run(episodes=10)
    else:
        main()