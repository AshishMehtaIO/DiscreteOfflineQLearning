from base_agents import Agent, QLearning, QLambda
import gym
import os
from gym import wrappers
import matplotlib.pyplot as plt
from pyvirtualdisplay import Display
from tqdm import tqdm
import envs
import numpy as np


class OffLineBase(Agent):
    def __init__(self,
                 env=gym.make('DiscreteMountainCar-v0'),
                 seed=100,
                 gamma=0.99,
                 lr=0.1,
                 eps=1.0):
        super(OffLineBase, self).__init__(env=env, seed=seed, gamma=gamma, lr=lr, eps=eps)
        self._reward_list = []
        self._policy = None
        self._trajectories = []

    def load_trajectories(self, traj_file):
        # self._trajectories = np.concatenate((self._trajectories, np.load(traj_file, allow_pickle=True)))
        self._trajectories.extend(np.load(traj_file, allow_pickle=True))

    def evaluate_agent(self, render=False):
        if render:
            virtual_display = Display(visible=0, size=(1400, 900))
            virtual_display.start()

        self._policy = self.get_greedy_policy()

        self._reward_list = []
        for i in range(500):
            env = self._env

            if render:
                if i <= 5:
                    env = wrappers.Monitor(self._env,
                                           "./save/offline/videos/{}_{}DiscreteMountainCar-v0".format(self._seed, i),
                                           force=True)
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


class OffLineQ(OffLineBase, QLearning):
    def __init__(self, env=gym.make('DiscreteMountainCar-v0'),
                 seed=100,
                 gamma=0.99,
                 lr=0.1,
                 eps=1.0):
        super(OffLineQ, self).__init__(env=env, seed=seed, gamma=gamma, lr=lr, eps=eps)

    def train(self):
        assert len(self._trajectories) is not 0
        for episode in self._trajectories:
            for transition in episode:
                (s, a, s_, r, d) = transition
                self.qlearning_update(s, a, s_, r, d)


class OffLineQLambda(OffLineBase, QLambda):
    def __init__(self, env=gym.make('DiscreteMountainCar-v0'),
                 seed=100,
                 gamma=0.99,
                 lr=0.1,
                 eps=1.0,
                 lmbd=0.8):
        super(OffLineQLambda, self).__init__(env=env, seed=seed, gamma=gamma, lr=lr, eps=eps, lmbd=lmbd)

    def train(self):
        assert len(self._trajectories) is not 0
        for episode in self._trajectories:
            for idx, transition in enumerate(episode):
                if idx:
                    (s, a, s_, r, d) = episode[idx-1]
                    (_, a_, _, _, _) = transition
                    self.qlamba_update(s, a, s_, a_, r, d)