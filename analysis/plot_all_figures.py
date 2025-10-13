"""
plot_all_figures.py

Generate a set of diagnostic plots and simple diagrams for RL training and behavior.

Usage:
    python analysis/plot_all_figures.py

The script will look for logs in `logs/training_metrics.csv` and `logs/steps.csv`.
If those files are missing it will synthesize example data so you can see the plots.

Outputs are saved to `outputs/plots/` as PNG files.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUT_DIR = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_or_synthesize():
    logs_dir = Path("logs")
    training_csv = logs_dir / "training_metrics.csv"
    steps_csv = logs_dir / "steps.csv"

    if training_csv.exists() and steps_csv.exists():
        print(f"Found logs: {training_csv}, {steps_csv}")
        df_train = pd.read_csv(training_csv)
        df_steps = pd.read_csv(steps_csv)
    else:
        print("Log files not found - synthesizing example data.")
        # Synthesize training episodes
        episodes = np.arange(1, 501)
        reward = np.tanh((episodes - 50) / 100.0) * 200 + np.random.randn(len(episodes)) * 10
        length = np.clip(200 - (episodes / 3) + np.random.randn(len(episodes)) * 10, 20, 200).astype(int)
        success = (reward > 50).astype(int)
        collisions = np.random.poisson(lam=np.clip(5 - episodes / 150, 0.2, 5), size=len(episodes))
        goal_dist = np.clip(100 - reward + np.random.randn(len(episodes)) * 5, 0, 200)
        df_train = pd.DataFrame({
            'episode': episodes,
            'total_reward': reward,
            'length': length,
            'success': success,
            'collisions': collisions,
            'goal_dist': goal_dist,
        })

        # Synthesize per-step data for a subset of episodes
        rows = []
        for ep in range(1, 51):
            steps = int(np.clip(150 - ep, 20, 150))
            x = np.cumsum(np.random.randn(steps) * 0.5)
            y = np.cumsum(np.random.randn(steps) * 0.5)
            z = np.clip(np.random.randn(steps) * 0.1 + 10, 0.0, 20.0)
            actions = np.tanh(np.random.randn(steps, 3))
            rewards = np.random.randn(steps) * 0.5 + (np.linspace(-1, 1, steps) * (ep / 50))
            td_err = np.random.randn(steps) * (1.0 - ep / 100.0)
            rule_flag = (np.random.rand(steps) < 0.05).astype(int)  # rare rule overrides
            for i in range(steps):
                rows.append({
                    'episode': ep,
                    'step': i,
                    'x': x[i],
                    'y': y[i],
                    'z': z[i],
                    'action0': actions[i, 0],
                    'action1': actions[i, 1],
                    'action2': actions[i, 2],
                    'reward': rewards[i],
                    'td_error': td_err[i],
                    'rule_override': rule_flag[i],
                })
        df_steps = pd.DataFrame(rows)

    return df_train, df_steps


def save_fig(fig, name):
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=350)
    plt.close(fig)
    print(f"Saved {path}")


def plot_training_curve(df_train):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df_train['episode'], df_train['total_reward'], color='C0', alpha=0.25, label='raw')
    ax.plot(df_train['episode'], df_train['total_reward'].rolling(50, min_periods=1).mean(), color='C1', label='rolling(50)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    #ax.set_title('Training curve: total reward per episode')
    ax.legend()
    save_fig(fig, '01_training_curve.png')


def plot_avg_fire_distance(df_train):
    fig, ax = plt.subplots(figsize=(9, 4))
    if 'avg_goal_dist' in df_train.columns:
        ax.plot(df_train['episode'], df_train['avg_goal_dist'], color='C2', alpha=0.6)
        ax.plot(df_train['episode'], df_train['avg_goal_dist'].rolling(50, min_periods=1).mean(), color='C3', label='rolling(50)')
        ax.set_ylabel('Average Distance to Fire')
        #ax.set_title('Average Fire Distance per Episode')
    else:
        ax.text(0.5, 0.5, 'avg_goal_dist not available in logs', ha='center', va='center')
    ax.set_xlabel('Episode')
    save_fig(fig, '01b_avg_fire_distance.png')


def plot_length_and_success(df_train):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(df_train['episode'], df_train['length'], alpha=0.4)
    #axs[0].set_title('Episode length')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Length (steps)')

    window = 50
    success_rate = df_train['success'].rolling(window, min_periods=1).mean()
    axs[1].plot(df_train['episode'], success_rate)
    #axs[1].set_title(f'Success rate (rolling {window})')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Success rate')
    save_fig(fig, '02_length_and_success.png')


def plot_losses(df_train):
    # If loss columns exist use them; else synthesize
    if 'actor_loss' in df_train.columns and 'critic_loss' in df_train.columns:
        actor = df_train['actor_loss'].replace([np.inf, -np.inf], np.nan)
        critic = df_train['critic_loss'].replace([np.inf, -np.inf], np.nan)
        # If both are entirely NaN synthesize instead
        if actor.dropna().empty and critic.dropna().empty:
            actor = None
            critic = None
    else:
        actor = None
        critic = None
    if actor is None or critic is None:
        n = len(df_train)
        actor = np.abs(np.random.randn(n) * (1.0 - np.linspace(0, 0.9, n)))
        critic = np.abs(np.random.randn(n) * (1.2 - np.linspace(0, 0.9, n)))

    # Create two separate plots: one for actor, one for critic
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(df_train['episode'], actor, label='actor_loss', color='C0')
    ax1.set_yscale('log')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Actor Loss (log scale)')
    #ax1.set_title('Actor loss curve')
    save_fig(fig1, '03_actor_loss.png')

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(df_train['episode'], critic, label='critic_loss', color='C1')
    ax2.set_yscale('log')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Critic Loss (log scale)')
    #ax2.set_title('Critic loss curve')
    save_fig(fig2, '03_critic_loss.png')


def plot_td_error_hist(df_steps):
    fig, ax = plt.subplots(figsize=(7, 4))
    if 'td_error' in df_steps.columns and not df_steps['td_error'].dropna().empty:
        sns.histplot(df_steps['td_error'].dropna(), bins=80, kde=True, ax=ax)
        #ax.set_title('TD-error distribution')
    else:
        ax.text(0.5, 0.5, 'No TD-error data available', ha='center', va='center')
        ax.set_xlabel('td_error')
        ax.set_ylabel('Count')
    save_fig(fig, '04_td_error_hist.png')


def plot_state_visitation(df_steps):
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(df_steps['x'], df_steps['y'], gridsize=80, cmap='inferno')
    fig.colorbar(hb, ax=ax, label='visitation count')
    #ax.set_title('State visitation heatmap (x,y)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    save_fig(fig, '06_state_visitation.png')


def plot_trajectories(df_steps):
    # Plot sample trajectories for first N episodes
    sample_eps = sorted(df_steps['episode'].unique())[:20]
    fig, ax = plt.subplots(figsize=(6, 6))
    for ep in sample_eps:
        g = df_steps[df_steps['episode'] == ep]
        ax.plot(g['x'], g['y'], alpha=0.6)
    #ax.set_title('Trajectories (x,y)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    save_fig(fig, '07_trajectories.png')


def plot_reward_components(df_train):
    # synthesize components if not present
    if {'reward_goal', 'reward_collision', 'reward_energy'}.issubset(df_train.columns):
        comp = df_train[['reward_goal', 'reward_collision', 'reward_energy']]
    else:
        # divide total_reward into components for display
        total = df_train['total_reward']
        goal = np.clip(total * 0.6 + np.random.randn(len(total)) * 5, -200, 400)
        collision = -np.abs(np.random.randn(len(total)) * 5 + np.linspace(5, 0.5, len(total)))
        energy = total - goal - collision
        comp = pd.DataFrame({'goal': goal, 'collision': collision, 'energy': energy})

    # Reward components plot removed per user request
    return


def plot_normalized_total(df_train):
    fig, ax = plt.subplots(figsize=(9, 4))
    if 'normalized_total' in df_train.columns:
        ax.plot(df_train['episode'], df_train['normalized_total'], color='C4', alpha=0.3)
        ax.plot(df_train['episode'], df_train['normalized_total'].rolling(50, min_periods=1).mean(), color='C5', label='rolling(50)')
        #ax.set_title('Normalized Reward (per-episode)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Normalized Total')
    else:
        ax.text(0.5, 0.5, 'normalized_total not available in logs', ha='center', va='center')
    save_fig(fig, '08b_normalized_total.png')


def plot_action_vs_state_slice(df_steps):
    # pick action0 vs distance-to-origin
    df_steps = df_steps.copy()
    df_steps['dist'] = np.sqrt(df_steps['x']**2 + df_steps['y']**2)
    sample = df_steps.sample(min(len(df_steps), 2000))
    # Action vs state slice plot removed per user request
    return


def plot_q_value_heatmap(df_steps):
    # synthesize a Q surface by binning x and y and averaging a synthetic q-value
    df = df_steps.copy()
    df['q'] = - (df['x']**2 + df['y']**2) * 0.1 + np.random.randn(len(df)) * 0.5
    xedges = np.linspace(df['x'].quantile(0.01), df['x'].quantile(0.99), 60)
    yedges = np.linspace(df['y'].quantile(0.01), df['y'].quantile(0.99), 60)
    qgrid = np.zeros((len(xedges)-1, len(yedges)-1))
    counts = np.zeros_like(qgrid)
    for i in range(len(xedges)-1):
        for j in range(len(yedges)-1):
            mask = (df['x'] >= xedges[i]) & (df['x'] < xedges[i+1]) & (df['y'] >= yedges[j]) & (df['y'] < yedges[j+1])
            vals = df.loc[mask, 'q']
            if len(vals):
                qgrid[i, j] = vals.mean()
                counts[i, j] = len(vals)
            else:
                qgrid[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.nan_to_num(qgrid.T), origin='lower', aspect='auto', cmap='coolwarm')
    fig.colorbar(im, ax=ax, label='Q value (approx)')
    #ax.set_title('Approximate Q-value heatmap (binned x,y)')
    save_fig(fig, '10_q_value_heatmap.png')

def plot_decision_region(df_steps):
    # coarse grid showing most-frequent discretized action sign
    df = df_steps.copy()
    df['ax_sign'] = np.sign(df['action0']).astype(int)
    xbins = np.linspace(df['x'].quantile(0.02), df['x'].quantile(0.98), 40)
    ybins = np.linspace(df['y'].quantile(0.02), df['y'].quantile(0.98), 40)
    grid = np.zeros((len(xbins)-1, len(ybins)-1))
    for i in range(len(xbins)-1):
        for j in range(len(ybins)-1):
            mask = (df['x'] >= xbins[i]) & (df['x'] < xbins[i+1]) & (df['y'] >= ybins[j]) & (df['y'] < ybins[j+1])
            vals = df.loc[mask, 'ax_sign']
            if len(vals):
                grid[i, j] = vals.mode().iloc[0]
            else:
                grid[i, j] = 0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid.T, origin='lower', cmap='coolwarm', extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]])
    #ax.set_title('Decision region (sign of action0)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    save_fig(fig, '12_decision_region.png')



def main():
    df_train, df_steps = load_or_synthesize()
    plot_training_curve(df_train)
    plot_length_and_success(df_train)
    plot_losses(df_train)
    plot_td_error_hist(df_steps)
    plot_state_visitation(df_steps)
    plot_trajectories(df_steps)
    plot_reward_components(df_train)
    plot_avg_fire_distance(df_train)
    plot_normalized_total(df_train)
    plot_action_vs_state_slice(df_steps)
    plot_q_value_heatmap(df_steps)
    plot_decision_region(df_steps)

    print('All plots generated in', OUT_DIR)


if __name__ == '__main__':
    main()