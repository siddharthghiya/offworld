import torch
import gym
import offworld_gym
from offworld_gym.envs.common.channels import Channels
import numpy as np
import types
from ppo.multiprocessing_env import DummyVecEnv, VecNormalize
import argparse
import pdb

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--real', action='store_true', help='training on real environment')
parser.add_argument('--load-dir', default=None, help='path to the weights that you want to pre load')
parser.add_argument('--channel-type', default='DEPTH_ONLY', help='type of observation')
parser.add_argument('--num-eval', type=int, default=30, help='number of evaluation episodes')

args = parser.parse_args()

REAL = args.real
CHANNEL_TYPE = args.channel_type
LOAD_DIR = args.load_dir
NUM_EVAL_EPISODES = args.num_eval

def make_env():
    if REAL:
        try:
            return gym.make('OffWorldMonolithDiscreteReal-v0', channel_type=Channels.DEPTH_ONLY, resume_experiment=False,
                            learning_type=LearningType.END_TO_END, algorithm_mode=AlgorithmMode.TRAIN, experiment_name='real_ppo')
        except:
            return gym.make('OffWorldMonolithDiscreteReal-v0', channel_type=Channels.DEPTH_ONLY, resume_experiment=True,
                            learning_type=LearningType.END_TO_END, algorithm_mode=AlgorithmMode.TRAIN, experiment_name='real_ppo')

    else:
        if CHANNEL_TYPE == 'RGB_ONLY':
            return gym.make('OffWorldDockerMonolithDiscreteSim-v0', channel_type=Channels.RGB_ONLY)
        else:
            return gym.make('OffWorldDockerMonolithDiscreteSim-v0', channel_type=Channels.DEPTH_ONLY)

def update_current_obs(obs):
    obs = obs.reshape(1, obs.shape[-3], obs.shape[-2], obs.shape[-1])
    obs = np.transpose(obs, (0, 3, 1, 2))
    obs = torch.from_numpy(obs).float()
    current_obs[:, :] = obs

policy = torch.load(LOAD_DIR)
env = make_env

env = DummyVecEnv([env])
obs_shape = env.observation_space.shape
obs_shape = env.observation_space.shape[1:]
obs_shape = (obs_shape[-1], obs_shape[0], obs_shape[1])
current_obs = torch.zeros(1, *obs_shape)

total_steps = 0
success = 0

for i in range(NUM_EVAL_EPISODES):
    obs = env.reset()
    update_current_obs(obs)
    done = False
    episode_reward = 0.0
    steps = 0
    while not done:
        with torch.no_grad():
            _, action, _ = policy.act(current_obs, deterministic=True)
        action = action.squeeze(1).cpu().numpy()
        obs, reward, done, _ = env.step(action)
        
        episode_reward += reward

        update_current_obs(obs)
        env.render()
        steps += 1
    
    total_steps += steps
    success += episode_reward[0]
    print('episode' , i, 'total reward ', episode_reward, 'steps ', steps)

print('average steps per episode: ', total_steps/NUM_EVAL_EPISODES)
print('percentage of successful episodes: ', success/NUM_EVAL_EPISODES*100)