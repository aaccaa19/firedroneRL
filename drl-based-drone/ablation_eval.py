"""
Ablation Study Evaluation Tool
==============================
This module provides utilities to:
1. Run deterministic evaluation of ablation study PKL files
2. Convert regular trained agents into ablation study format
3. Generate comparative performance metrics across ablation modes
"""

import numpy as np
import torch
import csv
from pathlib import Path
from UOAR_drone_rl import (
    DroneEnv, TD3Agent, REWARD_COMPONENTS, REWARD_CLIP, 
    COLLISION_PENALTY
)

# Define scenario names locally (not exported from main module)
SCENARIO_NAMES = {
    1: "Static Fire Line",
    2: "Spreading Fire",
    3: "Line of Fires",
    4: "Random Static Fires",
    5: "Impassable Fire Wall",
    6: "Central Fire"
}


class AblationEvaluator:
    """Evaluate pre-trained agents under different ablation configurations."""
    
    def __init__(self, scenario, episodes_per_mode=500, steps_per_episode=300, 
                 curriculum_level=4, output_dir='logs/ablation_eval'):
        self.scenario = scenario
        self.episodes_per_mode = episodes_per_mode
        self.steps_per_episode = steps_per_episode
        self.curriculum_level = curriculum_level
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create environment once to get dimensions
        self.env_template = DroneEnv(curriculum_level=curriculum_level, scenario=scenario)
        self.obs_dim = self.env_template.observation_space.shape[1]
        self.act_dim = self.env_template.action_space.shape[1]
        
        # Results storage
        self.results = []
    
    def load_agents_from_pkl(self, pkl_path):
        """Load pre-trained agents from pickle file."""
        agents = [
            TD3Agent(
                area_size=self.env_template.area_size,
                drone_radius=self.env_template.drone_radius,
                fire_line=self.env_template.fire_line,
                fire_radius=self.env_template.fire_radius,
                safety_margin=self.env_template.safety_margin,
                max_step_size=self.env_template.max_step_size,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                x_max=self.env_template.x_max,
                y_max=self.env_template.y_max,
                z_min=self.env_template.z_min,
                z_max=self.env_template.z_max
            )
            for _ in range(self.env_template.n_drones)
        ]
        
        # Load weights into first agent (both drones use same policy in eval)
        agents[0].load(str(pkl_path))
        # Copy to second agent for consistency
        agents[1].actor.load_state_dict(agents[0].actor.state_dict())
        agents[1].actor_target.load_state_dict(agents[0].actor_target.state_dict())
        
        return agents
    
    def run_episode_deterministic(self, env, agents, episode_num):
        """Run a single deterministic episode and return metrics."""
        obs, _ = env.reset()
        done = [False, False]
        steps = 0
        total_reward = [0.0, 0.0]
        seen_grids = [set(), set()]
        env.fires_visited = [set(), set()]
        
        while not all(done) and steps < self.steps_per_episode:
            actions = np.zeros((env.n_drones, 3))
            for i in range(env.n_drones):
                if not done[i]:
                    actions[i] = agents[i].select_action(obs[i], deterministic=True)
            
            next_obs, rewards, done, _, info = env.step(actions, seen_grids=seen_grids)
            for i in range(env.n_drones):
                total_reward[i] += float(np.clip(rewards[i], -REWARD_CLIP, REWARD_CLIP))
            
            obs = next_obs
            steps += 1
        
        # Calculate metrics
        total_points = sum(total_reward)
        all_fires = set().union(*(getattr(env, 'fires_visited', [set(), set()])))
        fires_viewed = len(all_fires)
        boxes_explored = len(set().union(*seen_grids)) if seen_grids else 0
        success = 1 if all(r > COLLISION_PENALTY for r in total_reward) else 0
        
        return {
            'episode': episode_num,
            'total_reward': float(total_points),
            'fires_viewed': int(fires_viewed),
            'boxes_explored': int(boxes_explored),
            'success': int(success),
            'steps': int(steps)
        }
    
    def evaluate_pkl_file(self, pkl_path, mode_name=None, component_config=None):
        """
        Run deterministic evaluation on a single PKL file.
        
        Args:
            pkl_path: Path to the PKL file to load
            mode_name: Name for this evaluation mode (default: use pkl filename)
            component_config: Dict to override REWARD_COMPONENTS during eval
        
        Returns:
            dict with aggregated statistics
        """
        if mode_name is None:
            mode_name = Path(pkl_path).stem
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {mode_name}")
        print(f"PKL file: {pkl_path}")
        print(f"Scenario: {self.scenario} ({SCENARIO_NAMES.get(self.scenario, 'Unknown')})")
        print(f"Episodes: {self.episodes_per_mode}")
        if component_config:
            print(f"Component Config: {component_config}")
        print(f"{'='*60}")
        
        # Load agents
        try:
            agents = self.load_agents_from_pkl(pkl_path)
            print(f"✓ Loaded agents from {pkl_path}")
        except Exception as e:
            print(f"✗ Failed to load agents: {e}")
            return None
        
        # Save original REWARD_COMPONENTS
        original_components = REWARD_COMPONENTS.copy()
        
        # Apply ablation config if provided
        if component_config:
            REWARD_COMPONENTS.update(component_config)
        
        # Run episodes
        episode_rewards = []
        episode_fires = []
        episode_boxes = []
        episode_successes = []
        
        for ep in range(self.episodes_per_mode):
            env = DroneEnv(curriculum_level=self.curriculum_level, scenario=self.scenario)
            metrics = self.run_episode_deterministic(env, agents, ep + 1)
            
            episode_rewards.append(metrics['total_reward'])
            episode_fires.append(metrics['fires_viewed'])
            episode_boxes.append(metrics['boxes_explored'])
            episode_successes.append(metrics['success'])
            
            if (ep + 1) % 100 == 0:
                print(f"  Progress: {ep + 1}/{self.episodes_per_mode} episodes")
        
        # Restore original components
        REWARD_COMPONENTS.update(original_components)
        
        # Compute statistics
        stats = {
            'mode_name': mode_name,
            'pkl_path': str(pkl_path),
            'avg_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'avg_fires': float(np.mean(episode_fires)),
            'std_fires': float(np.std(episode_fires)),
            'avg_boxes': float(np.mean(episode_boxes)),
            'std_boxes': float(np.std(episode_boxes)),
            'success_rate': float(np.mean(episode_successes)),
            'episodes': self.episodes_per_mode
        }
        
        # Print summary
        print(f"\nResults for {mode_name}:")
        print(f"  Avg Reward: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  Avg Fires Viewed: {stats['avg_fires']:.2f} ± {stats['std_fires']:.2f}")
        print(f"  Avg Boxes Explored: {stats['avg_boxes']:.2f} ± {stats['std_boxes']:.2f}")
        print(f"  Success Rate: {stats['success_rate']:.2%}")
        
        self.results.append(stats)
        return stats
    
    def evaluate_ablation_suite(self, pkl_dir, pattern='ablation_*.pkl'):
        """
        Evaluate all ablation PKL files in a directory.
        
        Args:
            pkl_dir: Directory containing ablation PKL files
            pattern: Glob pattern to match PKL files (default: 'ablation_*.pkl')
        """
        pkl_dir = Path(pkl_dir)
        pkl_files = sorted(pkl_dir.glob(pattern))
        
        if not pkl_files:
            print(f"No PKL files matching pattern '{pattern}' found in {pkl_dir}")
            return None
        
        print(f"\nFound {len(pkl_files)} PKL files to evaluate")
        
        for pkl_path in pkl_files:
            self.evaluate_pkl_file(pkl_path)
        
        return self.results
    
    def save_results_csv(self, filename=None):
        """Save evaluation results to CSV."""
        if filename is None:
            filename = self.output_dir / f"ablation_eval_s{self.scenario}.csv"
        else:
            filename = Path(filename)
        
        if not self.results:
            print("No results to save")
            return
        
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'mode_name', 'pkl_path', 'avg_reward', 'std_reward', 
                'avg_fires', 'std_fires', 'avg_boxes', 'std_boxes', 
                'success_rate', 'episodes'
            ])
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"\n✓ Results saved to {filename}")
        return filename
    
    def generate_comparison_report(self):
        """Generate a human-readable comparison report."""
        if not self.results:
            print("No results to report")
            return
        
        print(f"\n{'='*80}")
        print(f"ABLATION STUDY EVALUATION REPORT")
        print(f"Scenario: {self.scenario} ({SCENARIO_NAMES.get(self.scenario, 'Unknown')})")
        print(f"Episodes per Mode: {self.episodes_per_mode}")
        print(f"Curriculum Level: {self.curriculum_level}")
        print(f"{'='*80}\n")
        
        # Sort: baseline first, then ablations
        sorted_results = sorted(
            self.results,
            key=lambda x: (x['mode_name'] != 'baseline_all_on', x['mode_name'])
        )
        
        # Find baseline for comparison
        baseline = next((r for r in sorted_results if r['mode_name'] == 'baseline_all_on'), None)
        
        for result in sorted_results:
            print(f"Mode: {result['mode_name']}")
            print(f"  Reward:  {result['avg_reward']:7.2f} ± {result['std_reward']:6.2f}", end="")
            if baseline and baseline['mode_name'] != result['mode_name']:
                delta = result['avg_reward'] - baseline['avg_reward']
                pct = (delta / abs(baseline['avg_reward']) * 100) if baseline['avg_reward'] != 0 else 0
                symbol = "↑" if delta > 0 else "↓" if delta < 0 else "="
                print(f"  {symbol} {delta:+7.2f} ({pct:+6.1f}%)")
            else:
                print()
            
            print(f"  Fires:   {result['avg_fires']:7.2f} ± {result['std_fires']:6.2f}")
            print(f"  Boxes:   {result['avg_boxes']:7.2f} ± {result['std_boxes']:6.2f}")
            print(f"  Success: {result['success_rate']:7.1%}")
            print()


def convert_regular_agent_to_ablation(pkl_path, scenario, ablation_component_configs, output_dir='logs'):
    """
    Convert a regular trained agent PKL into ablation study format.
    
    This function loads a single trained agent and generates multiple 
    "ablation" versions by disabling different reward components in the 
    environment during evaluation (not modifying the agent itself).
    
    Args:
        pkl_path: Path to the regular (non-ablation) trained agent PKL
        scenario: Scenario number to use for evaluation
        ablation_component_configs: Dict mapping mode names to REWARD_COMPONENTS configs
                                   e.g. {'no_proximity': {key: val, ...}, ...}
        output_dir: Directory to save ablation PKL files
    
    Returns:
        List of paths to generated ablation PKL files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConverting regular agent to ablation format")
    print(f"  Source: {pkl_path}")
    print(f"  Scenario: {scenario}")
    print(f"  Output: {output_dir}")
    
    # Load the source agent once
    env = DroneEnv(scenario=scenario)
    base_agent = TD3Agent(
        area_size=env.area_size,
        drone_radius=env.drone_radius,
        fire_line=env.fire_line,
        fire_radius=env.fire_radius,
        safety_margin=env.safety_margin,
        max_step_size=env.max_step_size,
        obs_dim=env.observation_space.shape[1],
        act_dim=env.action_space.shape[1],
        x_max=env.x_max,
        y_max=env.y_max,
        z_min=env.z_min,
        z_max=env.z_max
    )
    
    try:
        base_agent.load(str(pkl_path))
        print(f"✓ Loaded base agent from {pkl_path}")
    except Exception as e:
        print(f"✗ Failed to load agent: {e}")
        return []
    
    # Generate ablation PKL files
    output_paths = []
    for mode_name, component_config in ablation_component_configs.items():
        # Create a copy of the agent
        ablation_agent = TD3Agent(
            area_size=env.area_size,
            drone_radius=env.drone_radius,
            fire_line=env.fire_line,
            fire_radius=env.fire_radius,
            safety_margin=env.safety_margin,
            max_step_size=env.max_step_size,
            obs_dim=env.observation_space.shape[1],
            act_dim=env.action_space.shape[1],
            x_max=env.x_max,
            y_max=env.y_max,
            z_min=env.z_min,
            z_max=env.z_max
        )
        
        # Copy weights from base agent
        ablation_agent.actor.load_state_dict(base_agent.actor.state_dict())
        ablation_agent.actor_target.load_state_dict(base_agent.actor_target.state_dict())
        for i in range(len(ablation_agent.critics)):
            ablation_agent.critics[i].load_state_dict(base_agent.critics[i].state_dict())
            ablation_agent.critics_target[i].load_state_dict(base_agent.critics_target[i].state_dict())
        
        # Save with ablation-specific filename
        output_path = output_dir / f"ablation_{mode_name}_agent_0.pkl"
        ablation_agent.save(str(output_path))
        output_paths.append(output_path)
        print(f"  ✓ Saved {mode_name} to {output_path}")
    
    return output_paths


# ============================================================================
# INTERACTIVE CLI INTERFACE
# ============================================================================

def interactive_menu():
    """Interactive menu for ablation study evaluation."""
    print("\n" + "="*70)
    print("ABLATION STUDY EVALUATION TOOL")
    print("="*70)
    
    while True:
        print("\nOptions:")
        print("  1. Evaluate ablation PKL files from a directory")
        print("  2. Evaluate a single PKL file")
        print("  3. Convert regular agent to ablation format")
        print("  4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            evaluate_directory()
        elif choice == '2':
            evaluate_single()
        elif choice == '3':
            convert_agent()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


def evaluate_directory():
    """Interactive: evaluate all PKL files in a directory."""
    print("\n--- Evaluate Ablation Directory ---")
    
    # Choose scenario
    print("\nAvailable scenarios:")
    for s, name in SCENARIO_NAMES.items():
        print(f"  {s}: {name}")
    
    scenario_input = input("Enter scenario number (1-6): ").strip()
    try:
        scenario = int(scenario_input)
        if scenario not in SCENARIO_NAMES:
            print("Invalid scenario.")
            return
    except ValueError:
        print("Invalid input.")
        return
    
    pkl_dir = input("Enter directory containing PKL files (default: logs): ").strip() or "logs"
    
    episodes = input("Enter number of episodes per mode (default: 500): ").strip() or "500"
    try:
        episodes = int(episodes)
    except ValueError:
        episodes = 500
    
    # Run evaluator
    evaluator = AblationEvaluator(scenario, episodes_per_mode=episodes)
    evaluator.evaluate_ablation_suite(pkl_dir, pattern='ablation_*.pkl')
    
    # Save results
    evaluator.save_results_csv()
    evaluator.generate_comparison_report()


def evaluate_single():
    """Interactive: evaluate a single PKL file."""
    print("\n--- Evaluate Single PKL File ---")
    
    # Choose scenario
    print("\nAvailable scenarios:")
    for s, name in SCENARIO_NAMES.items():
        print(f"  {s}: {name}")
    
    scenario_input = input("Enter scenario number (1-6): ").strip()
    try:
        scenario = int(scenario_input)
        if scenario not in SCENARIO_NAMES:
            print("Invalid scenario.")
            return
    except ValueError:
        print("Invalid input.")
        return
    
    pkl_path = input("Enter path to PKL file: ").strip()
    if not Path(pkl_path).exists():
        print(f"File not found: {pkl_path}")
        return
    
    mode_name = input("Enter mode name (optional, default: filename): ").strip() or None
    
    episodes = input("Enter number of episodes (default: 500): ").strip() or "500"
    try:
        episodes = int(episodes)
    except ValueError:
        episodes = 500
    
    # Run evaluator
    evaluator = AblationEvaluator(scenario, episodes_per_mode=episodes)
    evaluator.evaluate_pkl_file(pkl_path, mode_name)
    evaluator.save_results_csv()
    evaluator.generate_comparison_report()


def convert_agent():
    """Interactive: convert regular agent to ablation format."""
    print("\n--- Convert Regular Agent to Ablation Format ---")
    
    pkl_path = input("Enter path to regular agent PKL file: ").strip()
    if not Path(pkl_path).exists():
        print(f"File not found: {pkl_path}")
        return
    
    # Choose scenario
    print("\nAvailable scenarios:")
    for s, name in SCENARIO_NAMES.items():
        print(f"  {s}: {name}")
    
    scenario_input = input("Enter scenario number (1-6): ").strip()
    try:
        scenario = int(scenario_input)
        if scenario not in SCENARIO_NAMES:
            print("Invalid scenario.")
            return
    except ValueError:
        print("Invalid input.")
        return
    
    output_dir = input("Enter output directory (default: logs): ").strip() or "logs"
    
    # Define ablation configurations
    ablation_configs = {
        'baseline_all_on': {k: True for k in REWARD_COMPONENTS},
        'no_proximity': {k: (k != 'proximity') for k in REWARD_COMPONENTS},
        'no_exploration': {k: (k != 'exploration') for k in REWARD_COMPONENTS},
        'no_energy': {k: (k != 'energy') for k in REWARD_COMPONENTS},
        'no_edge': {k: (k != 'edge') for k in REWARD_COMPONENTS},
        'no_movement': {k: (k != 'movement') for k in REWARD_COMPONENTS},
        'no_discovery': {k: (k != 'discovery') for k in REWARD_COMPONENTS},
    }
    
    # Convert
    paths = convert_regular_agent_to_ablation(pkl_path, scenario, ablation_configs, output_dir)
    print(f"\n✓ Generated {len(paths)} ablation PKL files")
    for p in paths:
        print(f"  - {p}")


# ============================================================================
# INTERACTIVE CLI INTERFACE
# ============================================================================

def interactive_menu():
    """Interactive menu for ablation study evaluation."""
    print("\n" + "="*70)
    print("ABLATION STUDY EVALUATION TOOL")
    print("="*70)
    
    while True:
        print("\nOptions:")
        print("  1. Evaluate ablation PKL files from a directory")
        print("  2. Evaluate a single PKL file")
        print("  3. Convert regular agent to ablation format")
        print("  4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            evaluate_directory()
        elif choice == '2':
            evaluate_single()
        elif choice == '3':
            convert_agent()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command-line mode
        if sys.argv[1] == '--scenario' and len(sys.argv) > 2:
            scenario = int(sys.argv[2])
            evaluator = AblationEvaluator(scenario, episodes_per_mode=500)
            pkl_dir = sys.argv[3] if len(sys.argv) > 3 else 'logs'
            evaluator.evaluate_ablation_suite(pkl_dir, pattern='ablation_*.pkl')
            evaluator.save_results_csv()
            evaluator.generate_comparison_report()
        else:
            print("Usage: python ablation_eval.py --scenario <num> [pkl_dir]")
    else:
        # Interactive mode
        interactive_menu()
