import torch
import gym
import offworld_gym
from offworld_gym.envs.common.channels import Channels
import numpy as np
import types
from ppo.multiprocessing_env import DummyVecEnv, VecNormalize
import pdb

ENV_NAME = 'OffWorldDockerMonolithDiscreteSim-v0'
load_path = 'weights/model_0.pt'
policy = torch.load(load_path)

def make_env():
    return gym.make(ENV_NAME)
env = make_env

env = DummyVecEnv([env])
obs_shape = env.observation_space.shape
obs_shape = env.observation_space.shape[1:]
obs_shape = (obs_shape[-1], obs_shape[0], obs_shape[1])
current_obs = torch.zeros(1, *obs_shape)
def update_current_obs(obs):
    obs = obs.reshape(1, obs.shape[-3], obs.shape[-2], obs.shape[-1])
    obs = np.transpose(obs, (0, 3, 1, 2))
    obs = torch.from_numpy(obs).float()
    current_obs[:, :] = obs


for i in range(10):
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

    print('episode' , i, 'total reward ', episode_reward, 'steps ', steps)

