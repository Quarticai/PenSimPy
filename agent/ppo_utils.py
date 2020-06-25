import numpy as np
import torch
import scipy.signal


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_np_ary(t, device='cpu'):
    if 'cpu' in device:
        return t.numpy()
    #elif 'cuda' in device:
    #    return t.cpu().numpy()
    else:
        raise ValueError(f"unrecognized device type {device}")


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class EpisodeBuffer:
    """
    A buffer to store one episode worth of data.
    """
    def __init__(self, obs_dim, act_dim, size, gamma, lam):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.max_size = 0, size

        self.adv_mean, self.adv_std = 0, 0
        self.obs_mean, self.obs_std = np.zeros(size), np.zeros(size)

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        rews = np.append(self.rew_buf, last_val)
        vals = np.append(self.val_buf, last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf = discount_cumsum(deltas, self.gamma * self.lam)

        self.ret_buf = discount_cumsum(rews, self.gamma)[:-1]

        self.ptr = 0

    def get(self):
        """TODO: normalize observation space as well?"""
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf) # TODO: moving avg of means, stds?
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v.copy(), dtype=torch.float32) for k, v in data.items()}
