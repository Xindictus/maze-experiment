import copy

import torch
import torch.nn.functional as F

from rl_models.td3.buffer import ReplayBuffer
from rl_models.td3.networks import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3_BC(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        alpha=2.5,
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=3e-4
        )

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=3e-4
        )

        self.max_action = max_action
        self.discount = discount
        self.tau = tau

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0

        # TODO: What is missing here?
        # self.memory = ReplayBuffer(args, state_dim, action_dim, device)
        self.memory = ReplayBuffer(state_dim, action_dim, device)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def learn(self, block_nb):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = self.memory.sample(
            block_nb
        )

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            lmbda = self.alpha / Q.abs().mean().detach()

            actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

        return actor_loss.item(), critic_loss.item(), 0, 0

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(
            self.critic_optimizer.state_dict(), filename + "_critic_optimizer"
        )

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(
            self.actor_optimizer.state_dict(), filename + "_actor_optimizer"
        )

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer")
        )
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer")
        )
        self.actor_target = copy.deepcopy(self.actor)
