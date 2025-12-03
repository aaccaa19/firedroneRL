import importlib.util
import traceback
from pathlib import Path
import torch
import numpy as np

try:
    p = Path(r"c:\Users\caden\firedroneRL\drl-based-drone\UOAR_drone_rl.py")
    spec = importlib.util.spec_from_file_location("uoa", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    env = mod.DroneEnv(curriculum_level=0, scenario=2)
    obs_dim = env.observation_space.shape[1]
    act_dim = env.action_space.shape[1]
    print(f"env obs_dim={obs_dim} act_dim={act_dim} n_drones={env.n_drones} area_size={env.area_size}")

    agent = mod.TD3Agent(env.area_size, env.drone_radius, env.fire_line, env.fire_radius, env.safety_margin, env.max_step_size, obs_dim=obs_dim, act_dim=act_dim)
    agent.actor.eval()
    for c in agent.critics:
        c.eval()

    batch = 32
    o = torch.randn(batch, obs_dim)

    with torch.no_grad():
        pred = agent.actor(o)
        q_vals = torch.stack([c(o, pred) for c in agent.critics], dim=0)
        q_min = torch.min(q_vals, dim=0).values
        q_clamped = torch.clamp(q_min, -50.0, 50.0)

        print('q_min mean,std', float(q_min.mean().item()), float(q_min.std().item()))
        print('q_clamped mean,std', float(q_clamped.mean().item()), float(q_clamped.std().item()))

        actor_loss_raw = -q_clamped.mean().item()
        print('actor_loss_raw (clamped) =', actor_loss_raw)

        # per-critic stats
        for i, c in enumerate(agent.critics):
            q = c(o, pred)
            print(f'critic[{i}] mean,std: {float(q.mean().item()):.6f}, {float(q.std().item()):.6f}')

    # actor parameter norms
    total_param_norm = 0.0
    for p in agent.actor.parameters():
        try:
            total_param_norm += float(torch.norm(p).item())
        except Exception:
            pass
    print('actor_param_norm', total_param_norm)

    print('actor_lr:', agent.actor_opt.defaults.get('lr'))
    print('critic_lrs:', [opt.defaults.get('lr') for opt in agent.critic_opts])
    print('policy_delay:', agent.policy_delay, 'tau:', agent.tau, 'grad_clip:', mod.GRAD_CLIP_NORM)

except Exception:
    traceback.print_exc()
