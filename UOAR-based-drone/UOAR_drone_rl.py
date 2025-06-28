import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- ENVIRONMENT ---
class DroneEnv(gym.Env):
    def __init__(self, area_size=10, drone_radius=0.5, fire_line=((5,5,0),(5,5,10)), fire_radius=0.5, safety_margin=0.2, max_step_size=0.5):
        super().__init__()
        self.area_size = area_size
        self.drone_radius = drone_radius
        self.fire_line = np.array(fire_line)
        self.fire_radius = fire_radius
        self.safety_margin = safety_margin
        self.max_step_size = max_step_size
        self.n_drones = 2
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_drones,3), dtype=np.float32)
        # Each drone: [x, y, z, fire_below, other_x, other_y, other_z]
        low = np.array([0,0,0,0,0,0,0]*self.n_drones).reshape((self.n_drones,7))
        high = np.array([area_size,area_size,area_size,1,area_size,area_size,area_size]*self.n_drones).reshape((self.n_drones,7))
        self.observation_space = spaces.Box(low=low, high=high, shape=(self.n_drones,7), dtype=np.float32)
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
        obs = []
        k = 1.0  # scaling factor for cone radius (can be tuned)
        fire_a_2d = np.array([self.fire_line[0][0], self.fire_line[0][1]])
        fire_b_2d = np.array([self.fire_line[1][0], self.fire_line[1][1]])
        for i in range(self.n_drones):
            pos = self.drone_pos[i]
            other = self.drone_pos[1-i]
            # Project drone position onto ground
            drone_xy = np.array([pos[0], pos[1]])
            z = pos[2]
            view_radius = k * z / 4  # decreased by 2
            # Check if fire line is within the cone's base (circle on ground)
            dist_to_fire = point_to_segment_dist_2d(drone_xy, fire_a_2d, fire_b_2d)
            fire_in_view = 1.0 if dist_to_fire <= (view_radius + self.fire_radius) else 0.0
            obs.append(np.concatenate([pos, [fire_in_view], other]))
        return np.stack(obs)

    def step(self, actions, seen_grids=None):
        actions = np.clip(actions, -1, 1)
        rewards = [1, 1]
        dones = [False, False]
        # Move both drones
        for i in range(self.n_drones):
            a = actions[i]
            a = a / (np.linalg.norm(a) + 1e-8) * min(np.linalg.norm(a), 1.0)
            a = a * self.max_step_size
            self.drone_pos[i] = np.clip(self.drone_pos[i] + a, [0, 0, 5], [self.area_size, self.area_size, 10])
        # Check collisions
        def point_to_segment_dist(p, a, b):
            ap = p - a
            ab = b - a
            t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-8)
            t = np.clip(t, 0, 1)
            closest = a + t * ab
            return np.linalg.norm(p - closest)
        min_dist = self.drone_radius + self.fire_radius + self.safety_margin
        for i in range(self.n_drones):
            dist = point_to_segment_dist(self.drone_pos[i], self.fire_line[0], self.fire_line[1])
            if dist <= min_dist:
                rewards[i] = -10
                dones[i] = True
        # Drone-drone collision
        if np.linalg.norm(self.drone_pos[0] - self.drone_pos[1]) < 2*self.drone_radius + self.safety_margin:
            rewards = [-10, -10]
            dones = [True, True]
        # Exploration reward (grid-based)
        if seen_grids is not None:
            k = 1.0
            grid_size = 0.5  # tune as needed
            for i in range(self.n_drones):
                pos = self.drone_pos[i]
                drone_xy = np.array([pos[0], pos[1]])
                z = pos[2]
                view_radius = k * z / 4
                # Find grid cells within view_radius
                x_min = max(0, int((drone_xy[0] - view_radius) // grid_size))
                x_max = min(int(self.area_size // grid_size), int((drone_xy[0] + view_radius) // grid_size))
                y_min = max(0, int((drone_xy[1] - view_radius) // grid_size))
                y_max = min(int(self.area_size // grid_size), int((drone_xy[1] + view_radius) // grid_size))
                new_cells = 0
                for gx in range(x_min, x_max+1):
                    for gy in range(y_min, y_max+1):
                        cx = gx * grid_size + grid_size/2
                        cy = gy * grid_size + grid_size/2
                        if np.linalg.norm(drone_xy - np.array([cx, cy])) <= view_radius:
                            if (gx, gy) not in seen_grids[i]:
                                seen_grids[i].add((gx, gy))
                                new_cells += 1
                if new_cells > 0:
                    rewards[i] += 5  # exploration reward set to 5
        self.done = dones
        return self._get_obs(), rewards, dones, False, {}

    def render(self):
        # 3D visualization using matplotlib
        fig = plt.figure(1)
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([0, self.area_size])
        ax.set_ylim([0, self.area_size])
        ax.set_zlim([0, self.area_size])
        # Draw fire line
        p1, p2 = self.fire_line
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='r', linewidth=4, label='Fire Line')
        # Draw drones as spheres
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        for pos in self.drone_pos:
            x = pos[0] + self.drone_radius * np.cos(u) * np.sin(v)
            y = pos[1] + self.drone_radius * np.sin(u) * np.sin(v)
            z = pos[2] + self.drone_radius * np.cos(v)
            ax.plot_surface(x, y, z, color='b', alpha=0.5)
        # Draw cone base (field of view) for each drone
        k = 1.0  # must match _get_obs
        for pos in self.drone_pos:
            drone_xy = np.array([pos[0], pos[1]])
            z = pos[2]
            view_radius = k * z / 4  # decreased by 2
            # Draw a disk on the ground (z=0)
            theta = np.linspace(0, 2 * np.pi, 50)
            circle_x = drone_xy[0] + view_radius * np.cos(theta)
            circle_y = drone_xy[1] + view_radius * np.sin(theta)
            circle_z = np.zeros_like(theta)
            ax.plot(circle_x, circle_y, circle_z, color='orange', alpha=0.7, linewidth=1.5)
            # Filled disk using Poly3DCollection
            verts = [list(zip(circle_x, circle_y, circle_z))]
            poly = Poly3DCollection(verts, color='orange', alpha=0.15)
            ax.add_collection3d(poly)
        # Draw safety AABB as transparent cubes (12 edges each)
        for pos in self.drone_pos:
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
        ax.set_title(f"Drone positions: {self.drone_pos.round(2)}")
        plt.draw()
        plt.pause(0.01)

# --- TD3 AGENT WITH aPE2 ACTION SELECTION ---
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, act_dim), nn.Tanh()  # output in [-1,1]
        )
    def forward(self, obs):
        return self.net(obs)

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)

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
        idx = np.random.randint(0, max_i, size=batch_size)
        return (
            torch.tensor(self.obs[idx]),
            torch.tensor(self.act[idx]),
            torch.tensor(self.rew[idx]),
            torch.tensor(self.next_obs[idx]),
            torch.tensor(self.done[idx])
        )

class TD3Agent:
    def __init__(self, area_size, drone_radius, fire_line, fire_radius, safety_margin, max_step_size, obs_dim=7, act_dim=3, n_critics=2, n_candidates=10, noise_levels=[0.1,0.2,0.3], gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_delay=2, buffer_size=100000, batch_size=64):
        self.area_size = area_size
        self.drone_radius = drone_radius
        self.fire_line = np.array(fire_line)
        self.fire_radius = fire_radius
        self.safety_margin = safety_margin
        self.max_step_size = max_step_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.actor = Actor(self.obs_dim, self.act_dim)
        self.actor_target = Actor(self.obs_dim, self.act_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critics = nn.ModuleList([Critic(self.obs_dim, self.act_dim) for _ in range(n_critics)])
        self.critics_target = nn.ModuleList([Critic(self.obs_dim, self.act_dim) for _ in range(n_critics)])
        for i in range(n_critics):
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_opts = [torch.optim.Adam(c.parameters(), lr=1e-3) for c in self.critics]
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
    def select_action(self, obs_np):
        obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            base_action = self.actor(obs).cpu().numpy()[0]
        candidates = []
        for noise in self.noise_levels:
            for _ in range(self.n_candidates):
                perturbed = base_action + np.random.normal(0, noise, size=self.act_dim)
                perturbed = np.clip(perturbed, -1, 1)
                candidates.append(perturbed)
        # aPE2: For each candidate, simulate step and get immediate reward, then get Q value
        scores = []
        for a in candidates:
            # Simulate next state (only position part)
            sim_pos = np.clip(obs_np[:3] + a * self.max_step_size, 0, self.area_size)
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
            # Immediate reward (simulate collision)
            dist = point_to_segment_dist(sim_pos, self.fire_line[0], self.fire_line[1])
            min_dist = self.drone_radius + self.fire_radius + self.safety_margin
            collision = dist <= min_dist
            immediate_reward = -10 if collision else 1
            # Long-term Q (average over critics)
            obs_t = torch.tensor(sim_obs, dtype=torch.float32).unsqueeze(0)
            act_t = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
            q_vals = [critic(obs_t, act_t).item() for critic in self.critics]
            avg_q = np.mean(q_vals)
            # Weighted score: immediate + gamma * avg_q
            score = immediate_reward + self.gamma * avg_q
            scores.append(score)
        best_idx = int(np.argmax(scores))
        return candidates[best_idx]
    def store(self, o, a, r, no, d):
        self.buffer.add(o, a, r, no, d)
    def train(self):
        if self.buffer.ptr < self.batch_size:
            return
        self.total_it += 1
        o, a, r, no, d = self.buffer.sample(self.batch_size)
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_a = self.actor_target(no)
            next_a = (next_a + noise).clamp(-1, 1)
            # Target Q
            target_qs = [ct(no, next_a) for ct in self.critics_target]
            min_target_q = torch.min(torch.stack(target_qs, dim=0), dim=0)[0]
            target = r + self.gamma * (1 - d) * min_target_q
        # Critic update
        for i, critic in enumerate(self.critics):
            q = critic(o, a)
            loss = nn.MSELoss()(q, target)
            self.critic_opts[i].zero_grad()
            loss.backward()
            self.critic_opts[i].step()
        # Delayed policy update
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critics[0](o, self.actor(o)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            # Polyak averaging
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for i in range(len(self.critics)):
                for param, target_param in zip(self.critics[i].parameters(), self.critics_target[i].parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# --- MAIN LOOP ---
def main():
    num_episodes = 10
    env = DroneEnv()
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
    episode_rewards = [[], []]  # Store total points per episode for each drone
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = [False, False]
        total_reward = [0, 0]
        steps = 0
        crash_type = [None, None]  # Track crash reason for each drone
        # Track seen areas for each drone (list of (x, y, radius))
        seen_areas = [[], []]
        # Track seen grid cells for exploration reward
        seen_grids = [set(), set()]
        while not all(done) and steps < 100:
            actions = np.zeros((env.n_drones, 3))
            for i in range(env.n_drones):
                if not done[i]:
                    actions[i] = agents[i].select_action(obs[i])
            next_obs, rewards, done, _, _ = env.step(actions, seen_grids=seen_grids)
            # Record seen area for each drone
            k = 1.0  # must match _get_obs and render
            for i in range(env.n_drones):
                pos = next_obs[i]
                drone_xy = np.array([pos[0], pos[1]])
                z = pos[2]
                view_radius = k * z / 4
                seen_areas[i].append((drone_xy[0], drone_xy[1], view_radius))
            for i in range(env.n_drones):
                agents[i].store(obs[i], actions[i], rewards[i], next_obs[i], float(done[i]))
                agents[i].train()
                total_reward[i] += rewards[i]
                # Detect crash type
                if done[i] and crash_type[i] is None:
                    if rewards[i] == -10:
                        # Check if crashed into fire or other drone
                        if done[0] and done[1] and rewards[0] == -10 and rewards[1] == -10 and np.linalg.norm(next_obs[0][:3] - next_obs[1][:3]) < 2*env.drone_radius + env.safety_margin + 1e-3:
                            crash_type[0] = crash_type[1] = 'drone-drone collision'
                        else:
                            crash_type[i] = 'fire collision'
                            total_reward[i] = 0  # Immediately set points to zero on fire collisio
            if episode == num_episodes - 1:
                print(f"Step {steps}: Drone positions {next_obs[:, :3].round(2)}, Rewards {rewards}, Done {done}")
                env.render()
            obs = next_obs
            steps += 1
        for i in range(env.n_drones):
            if crash_type[i] is None and done[i]:
                crash_type[i] = 'other (timeout or unknown)'
            episode_rewards[i].append(total_reward[i])
        print(f"Episode {episode+1}/{num_episodes} Total Rewards: {total_reward}")
        for i in range(env.n_drones):
            print(f"  Drone {i+1}: Total Points = {total_reward[i]}, Ended by: {crash_type[i]}")
    # Remove non-blocking show here to avoid closing the stats plot
    # Plot total points per episode for each drone
    plt.ioff()  # Disable interactive mode for the stats plot
    plt.figure()
    plt.plot(episode_rewards[0], label='Drone 1')
    plt.plot(episode_rewards[1], label='Drone 2')
    plt.xlabel('Episode')
    plt.ylabel('Total Points')
    plt.title('Total Points per Episode')
    plt.legend()
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

if __name__ == "__main__":
    main()
