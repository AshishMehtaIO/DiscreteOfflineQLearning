import numpy as np
import gym
import envs
import matplotlib.pyplot as plt
from tqdm import tqdm
from base_agents import QLearning
from pyvirtualdisplay import Display
from gym import wrappers
import os


class Demonstrator(QLearning):
    def __init__(self,
                 env=gym.make('DiscreteMountainCar-v0'),
                 seed=1,
                 gamma=0.99,
                 lr=0.01,
                 eps=1.0,
                 num_episodes=50000,
                 policy_save_frequency=10):
        super(Demonstrator, self).__init__(env=env, seed=seed, gamma=gamma, lr=lr, eps=eps)
        self._env.seed(self._seed)
        self._num_episodes = num_episodes
        self._reward_list = []
        self._policy = None
        self._trajectories = None
        self._policy_save_frequency = policy_save_frequency

    def train(self):
        if not os.path.exists('./save/demos/policy'):
            os.makedirs('./save/demos/policy')

        for episode in tqdm(range(self._num_episodes+1)):
            total_reward = 0
            s = self._env.reset()
            done = False

            # if episode % self._policy_save_frequency == 0:
            #     self.save_greedy_policy('./save/demos/policy/{}_{}_policy'.format(self._seed, episode),
            #                             './save/demos/policy/{}_{}_q'.format(self._seed, episode))
            # TODO uncomment

            while not done:
                action = self.select_epgreedy_policy(s)
                s_, r, done, _ = self._env.step(action)
                total_reward += r
                self.qlearning_update(s, action, s_, r, done)
                s = s_
            self._eps = self._eps - 2 / self._num_episodes if self._eps > 0.01 else 0.01

            if episode % 1000 == 0:
                print(episode, total_reward, self._eps)
            self._reward_list.append(total_reward)

    def plot_ep_rewards(self):
        plt.plot(self._reward_list)
        plt.xlabel('Episodes')
        plt.ylabel('Total rewards')
        plt.show()

    def save_greedy_policy(self, policy_file, q_file):
        self._policy = self.get_greedy_policy()
        np.save(policy_file, self._policy)
        np.save(q_file, self._Q)

    def viz_policy(self, trajectories=False):

        if not os.path.exists('./save/demos/videos'):
            os.makedirs('./save/demos/videos')

        if not os.path.exists('./save/trajectories/videos'):
            os.makedirs('./save/trajectories/videos')

        virtual_display = Display(visible=0, size=(1400, 900))
        virtual_display.start()
        self._policy = self.get_greedy_policy()

        for i in range(5):
            if not trajectories:
                env = wrappers.Monitor(self._env,
                                   "./save/demos/videos/{}_{}DiscreteMountainCar-v0".format(self._seed, i),
                                   force=True)
            else:
                env = wrappers.Monitor(self._env,
                                       "./save/trajectories/videos/{}_{}DiscreteMountainCar-v0".format(self._seed, i),
                                       force=True)

            s = env.reset()
            env.render()
            done = False
            while not done:
                action = self._policy[s]
                s_, r, done, _ = env.step(action[0])
                env.render()
                s = s_
            env.close()

    def load_saved_policy(self, policy_file, q_file):
        self._policy = np.load(policy_file)
        self._Q = np.load(q_file)

    def rollout_trajectories(self, n):
        assert self._policy is not None
        self._env.seed(9999)
        buffer = []
        for i in tqdm(range(n)):
            trajectory = []
            s = self._env.reset()
            done = False
            while not done:
                a = self._policy[s][0]
                s_, r, done, _ = self._env.step(a)
                trajectory.append((s, a, s_, r, done))
                s = s_
            buffer.append(trajectory)

        self._trajectories = np.array(buffer)

    def save_trajectories(self, traj_file):
        assert self._trajectories is not None
        np.save(traj_file, self._trajectories)

    def get_mc_qvalue(self, state, action):
        self._policy = self.get_greedy_policy()
        returns = []
        for i in range(50000):
            ep_rewards = []
            # s = self._env.reset_to(state)
            s = self._env.reset()
            s, r, d, _ = self._env.step(action)
            ep_rewards.append(r)
            while not d:
                a = self._policy[s]
                s_, r, d, _ = self._env.step(a[0])
                ep_rewards.append(r)
                s = s_
            discounts = np.logspace(0, len(ep_rewards), num=len(ep_rewards), base=self._gamma, endpoint=False)
            ep_return = np.sum(np.multiply(np.array(ep_rewards), discounts))
            returns.append(ep_return)

        return np.average(returns)


if __name__ == '__main__':
    d1 = Demonstrator(num_episodes=5000, seed=2)
    # d1.train()
    # d1.plot_ep_rewards()
    # d1.viz_policy()

    d1.load_saved_policy('./save/demos/policy/2_100_policy.npy', './save/demos/policy/2_100_q.npy')
    traj = d1.rollout_trajectories(30000)
