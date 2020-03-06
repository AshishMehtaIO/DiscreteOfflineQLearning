import numpy as np
import gym
import envs
from offline_agents import OffLineQ, OffLineQLambda
from demonstrators import Demonstrator
import os


def make_demonstrators(seeds):
    demonstrators = []
    for seed in seeds:
        seed = int(seed)
        demonstrators.append(Demonstrator(num_episodes=10000, seed=seed))

    return demonstrators


def train_demonstrators(demos):
    print('Training demonstrators')
    for d in demos:
        d.train()
        d.plot_ep_rewards()


def generate_videos(demos):
    print('Generating videos')
    for d in demos:
        d.viz_policy()


def load_demonstrator_policy(demos, seeds, iteration):
    print('Loading Demonstrator Policy')
    for d, seed in zip(demos, seeds):
        seed = int(seed)
        d.load_saved_policy('./save/demos/policy/{}_{}_policy.npy'.format(seed, iteration),
                            './save/demos/policy/{}_{}_q.npy'.format(seed, iteration))


def rollout_demo_trajectories(demos, seeds, iteration, n):
    num_tr = int(n / len(demos))
    print('Rolling out trajectories')
    for d, seed in zip(demos, seeds):
        seed = int(seed)
        d.rollout_trajectories(num_tr)
        d.save_trajectories('./save/trajectories/{}_{}_traj.npy'.format(seed, iteration))

def viz_demo_policy(demos):
    print('Rendering demo videos')
    for d in demos:
        d.viz_policy(trajectories=True)


if __name__ == '__main__':
    num_demos = 32
    iteration = 300

    seeds = np.arange(num_demos, dtype=int)

    if not os.path.exists('./save/offline/videos'):
        os.makedirs('./save/offline/videos')
    if not os.path.exists('./save/demos/policy'):
        os.makedirs('./save/demos/policy')
    if not os.path.exists('./save/demos/videos'):
        os.makedirs('./save/demos/videos')
    if not os.path.exists('./save/trajectories/videos'):
        os.makedirs('./save/trajectories/videos')

    demonstrators = make_demonstrators(seeds)
    # train_demonstrators(demonstrators)
    # generate_videos(demonstrators)
    load_demonstrator_policy(demonstrators, seeds, iteration)
    rollout_demo_trajectories(demonstrators, seeds, iteration, 100000)
    # viz_demo_policy(demonstrators)

    agent = OffLineQ()

    print('Training offline agent')
    for seed in seeds:
        seed = int(seed)
        agent.load_trajectories('./save/trajectories/{}_{}_traj.npy'.format(seed, iteration))
    agent._Q = agent._Q * -10000
    agent.train()

    print('Evaluating offline agent')
    agent.evaluate_agent(render=False)
    agent.plot_ep_rewards()