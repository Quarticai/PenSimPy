import time
from pensimpy.pensim_classes.Recipe import Recipe
from pensimpy.env_setup.peni_env_setup import PenSimEnv
import numpy as np
import torch
from agent import PPO
import os.path
import random

obs_dim = 15
act_dim = 7
episodes_digital_twin = 3000
episodes_production = 100
res = []

def learn_one_episode(seed, agent):
    env = PenSimEnv(random_seed_ref=seed)  # 3361 with recipe
    done = False
    batch_data = env.reset()
    o = np.zeros(obs_dim)  # initial obs
    recipe = Recipe()
    time_stamp, batch_yield = 0, 0
    while not done:
        # time is from 1 to 1150
        time_stamp += 1

        a, v, logp = agent.ac.step(torch.as_tensor(o, dtype=torch.float32))

        """add adjustment to each action"""
        Fs_a, Foil_a, Fg_a, Fpres_a, Fdischarge_a, Fw_a, Fpaa_a = list(np.clip(a / 10, -0.1, 0.1))

        # Get action from recipe agent based on time
        Fs, Foil, Fg, Fpres, Fdischarge, Fw, Fpaa = recipe.run(time_stamp)

        """update recipe actions with agent actions"""
        Fs *= (1 + Fs_a)
        Foil *= (1 + Foil_a)
        Fg *= (1 + Fg_a)
        Fpres *= (1 + Fpres_a)
        Fdischarge *= (1 + Fdischarge_a)
        Fw *= (1 + Fw_a)
        Fpaa *= (1 + Fpaa_a)

        observation, batch_data, reward, done = env.step(time_stamp,
                                                         batch_data,
                                                         Fs, Foil, Fg, Fpres, max(Fdischarge, 0), max(Fw, 0),
                                                         max(Fpaa, 0))

        next_o = np.array([x[1] for x in observation])

        """in this setting we just collect reward in the end of the batch"""
        agent.buffer.store(o, a, reward, v, logp)
        o = next_o

        batch_yield += reward
    agent.buffer.finish_path()
    agent.update()
    return batch_yield, agent


def main():

    for i in range(4):
        """time step where agent is allowed to actions"""
        if os.path.isfile('agent.pt'):
            agent = torch.load('agent.pt')
        else:
            agent = PPO(obs_dim, act_dim, buffer_size=1150)

        """run one episode"""
        avg_yields = []
        t = time.time()

        for e in range(episodes_digital_twin):
            batch_yield, agent = learn_one_episode(random.randint(1, 1000), agent)
            print(f"digital twin episode: {e}, elapsed time: {int(time.time() - t)} s, digital twin batch_yield: {batch_yield}")
            avg_yields.append(batch_yield)
        res['digital_twin_'+str(i)] = np.mean(avg_yields)

        torch.save(agent, 'agent.pt')

        avg_yields = []
        t = time.time()
        for e in range(episodes_production):
            batch_yield, agent = learn_one_episode(274, agent)
            print(f"production episode: {e}, elapsed time: {int(time.time() - t)} s, production batch_yield: {batch_yield}")
            avg_yields.append(batch_yield)
        res['production_' + str(i)] = np.mean(avg_yields)
    print(res)

if __name__ == '__main__':
    main()
