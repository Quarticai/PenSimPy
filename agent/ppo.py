"""PPO Agent"""
import numpy as np
import torch
from torch import nn, optim
from torch.distributions.normal import Normal
from agent.ppo_utils import to_np_ary, EpisodeBuffer


class Actor(nn.Module):
    """Gaussian Actor"""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = nn.Sequential(nn.Linear(obs_dim, 64),
                                    nn.Tanh(),
                                    nn.Linear(64, act_dim),
                                    nn.Tanh())

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.v_net = nn.Sequential(nn.Linear(obs_dim, 64),
                                   nn.Tanh(),
                                   nn.Linear(64, 1))

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.pi = Actor(obs_dim, act_dim)
        self.v = Critic(obs_dim)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return to_np_ary(a), to_np_ary(v), to_np_ary(logp_a)

    def act(self, obs):
        return self.step(obs)[0]


class PPO:
    def __init__(self, obs_dim, act_dim, buffer_size, actor_critic=ActorCritic,
                 gamma=0.99, clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3,
                 train_pi_iters=2, train_v_iters=2, lam=0.97, target_kl=0.01):
        self.ac = actor_critic(obs_dim, act_dim)
        self.buffer = EpisodeBuffer(obs_dim, act_dim, buffer_size, gamma, lam)
        self.pi_optimizer = optim.Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = optim.Adam(self.ac.v.parameters(), lr=vf_lr)
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters

    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # track info like KL
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def update(self):
        data = self.buffer.get()
        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                print(f'Early stopping at step {i} due to reaching max kl.')
                break
            loss_pi.backward()
            self.pi_optimizer.step()

        # Learn value function
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            self.vf_optimizer.step()
