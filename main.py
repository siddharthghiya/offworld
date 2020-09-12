from offworld_gym import version

__version__     = version.__version__

import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys, pdb, time, glob
import numpy as np
from datetime import datetime
import pickle
import pdb

import gym
import offworld_gym
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.common.enums import AlgorithmMode, LearningType

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from ppo.memory import RolloutStorage
from ppo.model import Policy
from ppo.multiprocessing_env import SubprocVecEnv, VecNormalize
from tensorboardX import SummaryWriter
import shutil
import copy
import argparse

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--num-envs', type=int, default=8)
parser.add_argument('--real', action='store_true', help='training on real environment')
parser.add_argument('--load-dir', default=None, help='path to the weights that you want to pre load')
parser.add_argument('--env-name', default='OffWorldDockerMonolithDiscreteSim-v0', help='name of the environment')
parser.add_argument('--exp-name', default='tmp', help='directory to save models (default: tmp)')
parser.add_argument('--channel-type', default='DEPTH_ONLY', help='type of observation')
args = parser.parse_args()

# Standard definition of epsilon
EPS = 1e-5
# Hyperparams of the PPO equation
CLIP_PARAM = 0.2
# Number of update epochs
N_EPOCH = 4
# Number of mini batches to use in updating
N_MINI_BATCH = 32
# Coefficients for the loss term. (all relative to the action loss which is 1.0)
VALUE_LOSS_COEFF = 0.5
ENTROPY_COEFF = 0.01
# Learning rate of the optimizer
LR = 7e-4
# Clip gradient norm
MAX_GRAD_NORM = 0.5
# Number of steps to generate actions
N_STEPS = 100
# Total number of frames to train on
N_FRAMES = 1e6
# Should we use GPU?
CUDA = True
# Discounted reward factor
GAMMA = 0.99
# Number of environments to run in paralell this is like the batch size
N_ENVS = args.num_envs
# Environment we are going to use. Make sure it is a continuous action space task.
SAVE_INTERVAL = 10
LOG_INTERVAL = 1
MODEL_DIR = 'weights/' + args.exp_name
LOG_DIR = 'runs/'+ args.exp_name
LOAD_DIR = args.load_dir
CHANNEL_TYPE = args.channel_type
REAL = args.real

def make_env():
    if REAL:
        if LOAD_DIR:
            return gym.make('OffWorldMonolithDiscreteReal-v0', channel_type=Channels.DEPTH_ONLY, resume_experiment=True,
                            learning_type=LearningType.END_TO_END, algorithm_mode=AlgorithmMode.TRAIN)
        else:
            return gym.make('OffWorldMonolithDiscreteReal-v0', channel_type=Channels.DEPTH_ONLY, resume_experiment=False,
                            learning_type=LearningType.END_TO_END, algorithm_mode=AlgorithmMode.TRAIN)

    else:
        if CHANNEL_TYPE == 'RGB_ONLY':
            return gym.make('OffWorldDockerMonolithDiscreteSim-v0', channel_type=Channels.RGB_ONLY)
        else:
            return gym.make('OffWorldDockerMonolithDiscreteSim-v0', channel_type=Channels.DEPTH_ONLY)

def update_current_obs(obs):
    # we want to use the same tensor every time so just copy it over.
    obs = torch.from_numpy(obs).float()
    current_obs[:,:] = obs

def update_params(rollouts, policy, optimizer):
    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    # Normalize advantages. (0 mean. 1 std)
    advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

    for epoch_i in range(N_EPOCH):
        samples = rollouts.sample(advantages, N_MINI_BATCH)

        value_losses = []
        action_losses = []
        entropy_losses = []
        losses = []
        for obs, actions, returns, masks, old_action_log_probs, adv_targ in samples:
            values, action_log_probs, dist_entropy = policy.evaluate_actions(obs, actions)

            # This is where we apply the PPO equation.
            ratio = torch.exp(action_log_probs - old_action_log_probs)

            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM) * adv_targ

            action_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(returns, values)
            optimizer.zero_grad()

            loss = (value_loss * VALUE_LOSS_COEFF + action_loss - dist_entropy *
                    ENTROPY_COEFF)
            loss.backward()

            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            value_losses.append(value_loss.item())
            action_losses.append(action_loss.item())
            entropy_losses.append(dist_entropy.item())
            losses.append(loss.item())

    return np.mean(value_losses), np.mean(action_losses), np.mean(entropy_losses), np.mean(losses)

# Create our model output path if it does not exist.
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
else:
    shutil.rmtree(MODEL_DIR)
    os.makedirs(MODEL_DIR)

#create a logging directory
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
else:
    shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR)

writer = SummaryWriter(logdir=LOG_DIR)

# Parallelize environments
envs = [make_env for i in range(N_ENVS)]
envs = SubprocVecEnv(envs)
envs = VecNormalize(envs, gamma=GAMMA, ob=False, ret=False)
obs_shape = envs.observation_space.shape[1:]
obs_shape = (obs_shape[-1], obs_shape[0], obs_shape[1])

#define the policy
if LOAD_DIR:
    policy = torch.load(LOAD_DIR)
else:
    policy = Policy(obs_shape, envs.action_space.n)

optimizer = optim.Adam(policy.parameters(), lr=LR, eps=EPS)

current_obs = torch.zeros(N_ENVS, *obs_shape)
obs = envs.reset()
obs = obs.reshape(N_ENVS, obs.shape[-3], obs.shape[-2], obs.shape[-1])
obs = np.transpose(obs, (0, 3, 1, 2))
update_current_obs(obs)

# Intialize our rollouts
rollouts = RolloutStorage(N_STEPS, N_ENVS, obs_shape, envs.action_space.n, current_obs)

# Put on the GPU
if CUDA:
    policy.cuda()
    rollouts.cuda()
    current_obs.cuda()

episode_rewards = torch.zeros([N_ENVS, 1])
final_rewards = torch.zeros([N_ENVS, 1])

n_updates = int(N_FRAMES // N_STEPS // N_ENVS)
for update_i in range(n_updates):
    # Generate samples
    for step in range(N_STEPS):
        # Generate and take an action
        with torch.no_grad():
            value, action, action_log_prob = policy.act(rollouts.observations[step])

        take_actions = action.squeeze(1).cpu().numpy()

        if len(take_actions.shape) == 1:
            take_actions = np.expand_dims(take_actions, axis=-1)

        obs, reward, done, info = envs.step(take_actions)
        obs = obs.reshape(N_ENVS, obs.shape[-3], obs.shape[-2], obs.shape[-1])
        obs = np.transpose(obs, (0, 3, 1, 2))

        # convert to pytorch tensor
        reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
        reward_masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done])
        masks = torch.FloatTensor([np.zeros([*obs_shape]) if d else np.ones([*obs_shape]) for d in done])

        # update reward info for logging
        episode_rewards += reward
        final_rewards *= reward_masks
        final_rewards += (1 - reward_masks) * episode_rewards
        episode_rewards *= reward_masks

        # Update our current observation tensor
        current_obs *= masks
        update_current_obs(obs)

        rollouts.insert(current_obs, action, action_log_prob, value, reward, reward_masks)

    with torch.no_grad():
        next_value = policy.get_value(rollouts.observations[-1]).detach()

    rollouts.compute_returns(next_value, GAMMA)

    value_loss, action_loss, entropy_loss, overall_loss = update_params(rollouts, policy,
            optimizer)

    rollouts.after_update()

    # Log to tensorboard
    writer.add_scalar('data/action_loss', action_loss, update_i)
    writer.add_scalar('data/value_loss', value_loss, update_i)
    writer.add_scalar('data/entropy_loss', entropy_loss, update_i)
    writer.add_scalar('data/overall_loss', overall_loss, update_i)
    writer.add_scalar('data/avg_reward', final_rewards.mean(), update_i)
    
    if update_i % LOG_INTERVAL == 0:
        print('Update number ', update_i,'Reward: %.3f' % (final_rewards.mean()))

    if update_i % SAVE_INTERVAL == 0:
        save_model = policy
        if CUDA:
            save_model = copy.deepcopy(policy).cpu()

        torch.save(save_model, os.path.join(MODEL_DIR, 'model_%i.pt' % update_i))


writer.close()
