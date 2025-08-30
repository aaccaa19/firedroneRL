import matplotlib.pyplot as plt

# Example results structure for comparison framework
results = {
    "TD3": {
        "rewards": [[100, 120, 130], [110, 115, 125]],
        "fire_distances": [[2.5, 2.2, 2.0], [2.6, 2.3, 2.1]]
    },
    "Rule": {
        "rewards": [[90, 100, 105], [95, 98, 102]],
        "fire_distances": [[3.0, 2.8, 2.7], [3.1, 2.9, 2.8]]
    }
}

method_labels = ["TD3", "Rule"]

plt.ioff()
plt.figure(figsize=(15, 5))

# Subplot 1: Total rewards
plt.subplot(1, 3, 1)
for method_name in method_labels:
    plt.plot(results[method_name]["rewards"][0], label=f'{method_name} Drone 1', alpha=0.7)
    plt.plot(results[method_name]["rewards"][1], label=f'{method_name} Drone 2', alpha=0.7)
plt.xlabel('Episode')
plt.ylabel('Total Points')
plt.title('Total Points per Episode')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Average fire distances
plt.subplot(1, 3, 2)
for method_name in method_labels:
    dist1_filtered = [d for d in results[method_name]["fire_distances"][0] if d != float('inf')]
    dist2_filtered = [d for d in results[method_name]["fire_distances"][1] if d != float('inf')]
    plt.plot(range(len(dist1_filtered)), dist1_filtered, label=f'{method_name} Drone 1', alpha=0.7)
    plt.plot(range(len(dist2_filtered)), dist2_filtered, label=f'{method_name} Drone 2', alpha=0.7)
optimal_dist = 2.0  # Example value, replace with your calculation
plt.axhline(y=optimal_dist, color='green', linestyle=':', label=f'Optimal Distance ({optimal_dist:.2f})')
plt.xlabel('Episode')
plt.ylabel('Average Distance to Fire')
plt.title('Average Fire Distance per Episode')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 3: Combined performance metric
plt.subplot(1, 3, 3)
for method_name in method_labels:
    combined_scores = []
    rewards0 = results[method_name]["rewards"][0]
    rewards1 = results[method_name]["rewards"][1]
    fire_distances0 = results[method_name]["fire_distances"][0]
    for i in range(len(rewards0)):
        reward_norm = (rewards0[i] + rewards1[i]) / 2
        if i < len(fire_distances0) and fire_distances0[i] != float('inf'):
            dist_penalty = max(0, fire_distances0[i] - optimal_dist) * 50
            combined_score = reward_norm - dist_penalty
        else:
            combined_score = reward_norm - 100
        combined_scores.append(combined_score)
    plt.plot(combined_scores, linewidth=2, label=f'{method_name} Combined')
plt.xlabel('Episode')
plt.ylabel('Combined Performance Score')
plt.title('Combined Performance (Reward - Distance Penalty)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
