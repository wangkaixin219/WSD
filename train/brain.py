import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import MultivariateNormal
from utils import *
from queue import PriorityQueue

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    device = torch.device('cpu')
    print("Device set to : cpu")


class Memory(object):
    def __init__(self):
        self.actions = []
        self.states = []
        self.rewards = []
        self.logprobs = []
        self.done = []

    def size(self):
        return len(self.states)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.logprobs[:]
        del self.done[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_std_init=1e-2):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.ReLU()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def take_prob_action(self, state):
        action_mean = self.actor(state) + 1
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state) + 1
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)

        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

    def forward(self):
        raise NotImplementedError


class Brain(object):

    def __init__(self, state_dim, action_dim, hidden_dim=20, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, k_epochs=80,
                 eps_clip=0.2, action_std_init=1e-2):
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim, action_std_init).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.MseLoss = nn.MSELoss()

        self.action_std = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.buffer = Memory()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate=0.99, min_action_std=1e-4):
        self.action_std *= action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        self.action_std = min_action_std if self.action_std <= min_action_std else self.action_std
        self.set_action_std(self.action_std)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.take_prob_action(state)

        return state, action, action_logprob

    def store_sample(self, state, action, action_logprob, reward, done):
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.rewards.append(reward)
        self.buffer.done.append(done)

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.done)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # print(rewards.mean(), rewards.std())
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        for _ in range(self.k_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # print(ratios)

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = (-torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy).mean()
            # print(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        self.decay_action_std()
        # for g in self.optimizer.param_groups:
        #     g['lr'] *= 0.5

    def save_model(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)

    def load_model(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

