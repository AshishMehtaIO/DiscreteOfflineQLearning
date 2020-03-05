from gym.envs.toy_text import frozen_lake
import gym

class DeterministicFrozenLake(frozen_lake.FrozenLakeEnv):

    def __init__(self, desc=None, map_name="8x8", is_slippery=True):
        super(DeterministicFrozenLake, self).__init__(desc, map_name, is_slippery)

    def reset_to(self, state):
        super(DeterministicFrozenLake, self).reset()
        self.s = state
        return [self.s]

    def reset(self):
        return [super(DeterministicFrozenLake, self).reset()]

    def step(self, a):
        (s_, r, d, i) = super(DeterministicFrozenLake, self).step(a)
        return ([s_], r, d, i)


if __name__=='__main__':
    env = gym.make('FrozenLake-v0', is_slippery=False)
    env.reset()
    print("done")