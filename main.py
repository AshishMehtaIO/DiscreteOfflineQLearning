import numpy as np
import gym
import envs
from offline_agents import OffLineQ
from train_demonstrator import Demonstrator


if __name__=='__main__':
    d1 = Demonstrator(num_episodes=5000, seed=2)
    a1 = OffLineQ()

    # print('Training offline agent')
    # d1.train()
    # d1.plot_ep_rewards()
    # print('Generating videos')
    # d1.viz_policy()

    # print('Loading Demonstrator Policy')
    # d1.load_saved_policy('./save/demos/policy/0_10000_policy.npy', './save/demos/policy/0_10000_q.npy')
    # print('Rolling out trajectories')
    # d1.rollout_trajectories(10000)
    # d1.save_trajectories('./save/trajectories/0_10000_traj.npy')

    print('Training offline agent')
    a1.load_trajectories('./save/trajectories/0_10000_traj.npy')
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

