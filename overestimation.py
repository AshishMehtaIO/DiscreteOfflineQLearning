from main import *
from demonstrators import Demonstrator
import envs

if __name__=='__main__':
    d = [Demonstrator(gym.make('DeterministicFrozenLake-v0'), policy_save_frequency=1000000, num_episodes=100000, gamma=0.9)]
    # d = [Demonstrator( policy_save_frequency=1000000, num_episodes=10)]
    train_demonstrators(d)
    d[0].plot_ep_rewards()
    print("MC estimate", d[0].get_mc_qvalue(0,0))
    print("Q Learning", d[0]._Q[0,0])
    print('\n')
    print("MC estimate", d[0].get_mc_qvalue(0, 1))
    print("Q Learning", d[0]._Q[0, 1])
    print('\n')
    print("MC estimate", d[0].get_mc_qvalue(0, 2))
    print("Q Learning", d[0]._Q[0, 2])
    print('\n')
    print("MC estimate", d[0].get_mc_qvalue(0, 3))
    print("Q Learning", d[0]._Q[0, 3])