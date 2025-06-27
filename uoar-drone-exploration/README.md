# Rule Drone Exploration

This project implements a simple rule-based agent and environment for Unmanned Online Area Reconnaissance (UOAR) with drones. The goal is for the drone to explore the maximum area while avoiding fire zones. The agent uses a deterministic rule-based approach (no reinforcement learning).

## Features

- Custom Gymnasium environment (`UOAREnv`) in a single file (`uoar_flat.py`)
- Rule-based agent (`RuleAgent`) for demonstration
- Drones receive rewards for exploring new areas and penalties for entering fire zones
- Visualization using matplotlib

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the main script directly:

```bash
python uoar_flat.py
```

This will run 100 episodes of the rule-based agent in the environment and plot the total points per episode.

## File Structure

- `uoar_flat.py` — Main environment, agent, and script (all-in-one)
- `requirements.txt` — Python dependencies

## Requirements

- gymnasium
- numpy
- matplotlib

## Example Output

The script will display a plot titled **"Rule Drone: Total Points per Episode"** showing the agent's performance over time.

## Notes

- There is no reinforcement learning in this version; the agent follows a set of rules to explore the environment efficiently.
- If you want to use a different agent, you can modify or extend the `RuleAgent` class in `uoar_flat.py`.

---

For any questions or improvements, feel free to open an issue or contribute!