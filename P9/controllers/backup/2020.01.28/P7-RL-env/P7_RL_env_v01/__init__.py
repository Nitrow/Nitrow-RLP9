from gym.envs.registration import register

register(
    id='P7_RL-v0',
    entry_point='P7_RL_env_v01.envs:P7RLEnv',
)
