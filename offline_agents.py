from base_agent import Agent
import gym
import os
from gym import wrappers
import matplotlib.pyplot as plt
from pyvirtualdisplay import Display
from tqdm import tqdm
import envs
import numpy as np


class OffLineQ(Agent):
    def __init__(self,
                 env=gym.make('DiscreteMountainCar-v0'),
                 seed=100,
                 gamma=0.99,
                 lr=0.1,
                 eps=1.0):
        super(OffLineQ, self).__init__(env=env, seed=seed, gamma=gamma, lr=lr, eps=eps)
        self._reward_list = []
        self._policy = None
        self._trajectories = np.array([])

    def load_trajectories(self, traj_file):
        self._trajectories = np.concatenate((self._trajectories, np.load(traj_file, allow_pickle=True)))

    def train(self):
        assert self._trajectories.shape[0] is not 0
        for episode in self._trajectories:
            for transition in episode:
                (s, a, s_, r, d) = transition
                self.qlearning_update(s, a, s_, r, d)

    def evaluate_agent(self, render=False):
        if render:
            if not os.path.exists('./save/offline/videos'):
                os.makedirs('./save/offline/videos')

            virtual_display = Display(visible=0, size=(1400, 900))
            virtual_display.start()

        self._policy = self.get_greedy_policy()

        self._reward_list = []
        for i in range(500):
            if render:
                env = wrappers.Monitor(self._env,
                                   "./save/offline/videos/{}_{}_DiscreteMountainCar-v0".format(self._seed, i),
                                   force=True)
            else:
                env = self._env

            s = env.reset()
            if render:
                env.render()
            done = False
            total_reward = 0
            while not done:
                action = self._policy[s]
                s_, r, done, _ = env.step(action[0])
                total_reward += r
                if render:
                    env.render()
                s = s_

            self._reward_list.append(total_reward)
            env.close()

    def plot_ep_rewards(self):
        plt.plot(self._reward_list)
        plt.xlabel('Episodes')
        plt.ylabel('Total rewards')
        plt.show()