from gym.envs.registration import register
from mjrl.envs.mujoco_env import MujocoEnv

# ----------------------------------------

register(
    id='SallyReachFixed-v0',
    entry_point='MPL.MPL_envs.reach.reach_v0:sallyReachEnvFixed',
    max_episode_steps=80,
)

register(
    id='SallyReachRandom-v0',
    entry_point='MPL.MPL_envs.reach.reach_v0:sallyReachEnvRandom',
    max_episode_steps=80,
)