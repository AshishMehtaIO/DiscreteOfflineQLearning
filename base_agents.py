import numpy as np


class Agent:
    def __init__(self, env, seed, gamma, lr, eps):
        self._seed = seed
        self._env = env
        np.random.seed(self._seed)
        self._statedim = self._env.observation_space.n
        self._actiondim = self._env.action_space.n
        self._statespace = np.arange(self._statedim)
        self._actionspace = np.arange(self._actiondim)
        self._gamma = gamma
        self._lr = lr
        self._eps = eps
        self.init_qtable()

    def init_qtable(self):
        np.random.seed(self._seed)
        self._Q = np.random.random((self._statedim, self._actiondim))
        # TODO terminal state = 0

    def get_max_action(self, s):
        return np.argmax(self._Q[s, :])

    def select_epgreedy_policy(self, s):
        if np.random.random() < self._eps:
            action = np.random.choice(self._actionspace)
        else:
            action = np.argmax(self._Q[s, :])
        return action

    def get_greedy_policy(self):
        return np.argmax(self._Q, axis=1)


class QLearning(Agent):
    def __init__(self, env, seed, gamma, lr, eps):
        super(QLearning, self).__init__(env, seed, gamma, lr, eps)

    def qlearning_update(self, s, a, s_, r, d):
        a_ = self.get_max_action(s_)
        self._Q[s, a] = self._Q[s, a] + self._lr * (r + (1 - d) * self._gamma * self._Q[s_, a_] - self._Q[s, a])


class QLambda(Agent):
    def __init__(self, env, seed, gamma, lr, eps, lmbd):
        super(QLambda, self).__init__(env, seed, gamma, lr, eps)
        self._e = np.zeros_like(self._Q)
        self._lmbd = lmbd

    def qlamba_update(self, s, a, s_, a_, r, d):
        a_max = self.get_max_action(s_)
        delta = r + (1 - d) * self._gamma * self._Q[s_, a_max] - self._Q[s, a]
        self._e[s, a] = 1

        for state in self._statespace:
            for action in self._actionspace:
                self._Q[state, action] = self._Q[state, action] + self._lr * self._lmbd * self._e[state, action] * delta
                if a_max == a_:
                    self._e[state, action] = self._gamma * self._lmbd * self._e[state, action]
                else:
                    self._e[state, action] = 0
