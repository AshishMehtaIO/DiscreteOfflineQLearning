from gym.envs.registration import register

register(
    id='DiscreteMountainCar-v0',
    entry_point='envs.modified_mountain_car:DiscreteMountainCarEnv',
    max_episode_steps=1000,
    reward_threshold=-110.0,
)

register(
    id='DeterministicFrozenLake-v0',
    entry_point='envs.modified_frozen_lake:DeterministicFrozenLake',
    kwargs={'map_name' : '8x8'},
    max_episode_steps=200,
    reward_threshold=0.99, # optimum = 1
)