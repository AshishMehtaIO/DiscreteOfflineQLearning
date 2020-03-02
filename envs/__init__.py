from gym.envs.registration import register

register(
    id='DiscreteMountainCar-v0',
    entry_point='envs.modified_mountain_car:DiscreteMountainCarEnv',
    max_episode_steps=1000,
    reward_threshold=-110.0,
)