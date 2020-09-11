import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m,
              nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0),
              gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)

# Init layer to have the proper weight initializations.
def init_layer(m):
    weight = m.weight.data
    weight.normal_(0, 1)
    weight *= 1.0 / torch.sqrt(weight.pow(2).sum(1, keepdim=True))
    nn.init.constant_(m.bias.data, 0)
    return m

#network
class BasicBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=5, stride=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size=5, stride=stride, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(2, stride=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.lrelu(x)
        x = self.maxpool(x)

        return x

class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        #intial layers
        self.BB1 = BasicBlock(input_size=input_size, output_size=64, kernel_size=5)
        self.BB2 = BasicBlock(input_size=64, output_size=64, kernel_size=5)
        self.BB3 = BasicBlock(input_size=64, output_size=64, kernel_size=5)

        # We do not want to select action yet as that will be probablistic.
        self.actor_hidden = nn.Sequential(
                init_layer(nn.Linear(64, 64)),
                nn.Tanh(),
                init_layer(nn.Linear(64, 64)),
                nn.Tanh(),
            )

        self.critic = nn.Sequential(
                init_layer(nn.Linear(64, 64)),
                nn.Tanh(),
                init_layer(nn.Linear(64, 64)),
                nn.Tanh(),
                init_layer(nn.Linear(64, 1)),
            )

        self.train()

    def forward(self, inputs):
        inputs = self.BB1(inputs)
        inputs = self.BB2(inputs)
        inputs = self.BB3(inputs)
        inputs = F.avg_pool2d(inputs, [inputs.size(2), inputs.size(3)], stride=1)
        inputs = torch.flatten(inputs, start_dim=1)
        return self.actor_hidden(inputs), self.critic(inputs)

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space):
        super().__init__()
        self.actor_critic = Model(input_size=obs_shape[0])
        num_outputs = action_space

        # How we will define our normal distribution to sample action from
        # self.action_mean = init_layer(nn.Linear(64, num_outputs))
        # self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        #Categorical distribution for discrete action spcae.
        self.dist = Categorical(64, num_outputs)

    # def __get_dist(self, actor_features):
    #     action_mean = self.action_mean(actor_features)
    #     action_log_std = self.action_log_std.expand_as(action_mean)

    #     return torch.distributions.Normal(action_mean, action_log_std.exp())


    def act(self, inputs, deterministic=False):
        actor_features, value = self.actor_critic(inputs)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action).view(-1, 1)
        # dist_entropy = dist.entropy().sum(-1).mean()

        return value, action, action_log_probs

    def get_value(self, inputs):
        _, value = self.actor_critic(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        actor_features, value = self.actor_critic(inputs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy