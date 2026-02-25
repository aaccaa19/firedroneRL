import runpy
mod = runpy.run_path('drl-based-drone/UOAR_drone_rl.py')
DroneEnv = mod['DroneEnv']
env = DroneEnv(curriculum_level=0, scenario=6)
obs, _ = env.reset()
print('obs shape:', obs.shape)
print('obs len per drone:', obs.shape[1])
print('observation_space shape:', env.observation_space.shape)