import numpy as np
import gym
import envs
from offline_agents import OffLineQ
from demonstrators import Demonstrator


if __name__=='__main__':
    d1 = Demonstrator(num_episodes=5000, seed=2)
    a1 = OffLineQ()

    seed = 0
    iteration = 10000

    # print('Training offline agent')
    # d1.train()
    # d1.plot_ep_rewards()
    # print('Generating videos')
    # d1.viz_policy()

    # print('Loading Demonstrator Policy')
    # d1.load_saved_policy('./save/demos/policy/{}_{}_policy.npy'.format(seed, iteration),
    #                      './save/demos/policy/{}_{}_q.npy'.format(seed, iteration))
    # print('Rolling out trajectories')
    # d1.rollout_trajectories(10000)
    # d1.save_trajectories('./save/trajectories/{}_{}_traj.npy'.format(seed, iteration))

    print('Training offline agent')
    a1.load_trajectories('./save/trajectories/{}_{}_traj.npy'.format(seed, iteration))
    a1._Q = a1._Q * -10000
    a1.train()
    print('Evaluating offline agent')
    a1.evaluate_agent()
    a1.plot_ep_rewards()


    # demonstrators = []
    # for seed in range(16):
    #     demonstrators.append(Demonstrator(num_episodes=10000, seed=seed))
    #
    # for d in demonstrators:
    #     d.train()
    #     d.plot_ep_rewards()
    #     print('Generating videos')
    #     d.viz_policy()
    #
    # # print('Loading Demonstrator Policy')
    # # d1.load_saved_policy('./save/demos/policy/2_5000_policy.npy', './save/demos/policy/2_5000_q.npy')
    # # print('Rolling out trajectories')
    # traj = []
    # for d in demonstrators:
    #     traj.extend(d.rollout_trajectories(1000))

