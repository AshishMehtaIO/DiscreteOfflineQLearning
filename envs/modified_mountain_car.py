import gym
from gym.envs.classic_control import mountain_car
import numpy as np
from gym import spaces


class DiscreteMountainCarEnv(mountain_car.MountainCarEnv):

    def __init__(self, goal_velocity=0, pos_disc_bins=20, vel_dic_bins=20):
        super(DiscreteMountainCarEnv, self).__init__(goal_velocity)
        self.pos_disc_bins = pos_disc_bins
        self.vel_disc_bins = vel_dic_bins
        self.pos_disc_space = np.linspace(self.min_position, self.max_position, self.pos_disc_bins)
        self.vel_disc_space = np.linspace(-self.max_speed, self.max_speed, self.vel_disc_bins)
        self.observation_space = spaces.Discrete(self.pos_disc_bins*self.vel_disc_bins)

    def get_disc_state(self, obs):
        pos, vel = obs
        pos_bin = np.digitize(pos, self.pos_disc_space)
        vel_bin = np.digitize(vel, self.vel_disc_space)
        return np.array([((pos_bin - 1)*self.pos_disc_bins) + (vel_bin - 1)])

    def step(self, action):
        ob, reward, done, _ = super(DiscreteMountainCarEnv, self).step(action)
        disc_ob = self.get_disc_state(ob)
        return disc_ob, reward, done, {}

    def reset(self):
        ob = super(DiscreteMountainCarEnv, self).reset()
        disc_ob = self.get_disc_state(ob)
        return disc_ob


if __name__=='__main__':
    import envs
    env = gym.make('DiscreteMountainCar-v0')
    for i in range(100):
        print(env.reset())