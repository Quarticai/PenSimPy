import time
from pensimpy.pensim_classes.Recipe import Recipe
from pensimpy.env_setup.peni_env_setup import PenSimEnv
import numpy as np
import torch
from agent import PPO

obs_dim = 15
act_dim = 7
episodes = 100


"""time step where agent is allowed to actions"""
def get_actiontime():
    # recipes
    Fs = [15, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 800, 1750]
    Fs_sp = [8, 15, 30, 75, 150, 30, 37, 43, 47, 51, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80]

    Foil = [20, 80, 280, 300, 320, 340, 360, 380, 400, 1750]
    Foil_sp = [22, 30, 35, 34, 33, 32, 31, 30, 29, 23]

    Fg = [40, 100, 200, 450, 1000, 1250, 1750]
    Fg_sp = [30, 42, 55, 60, 75, 65, 60]

    pres = [62, 125, 150, 200, 500, 750, 1000, 1750]
    pres_sp = [0.6, 0.7, 0.8, 0.9, 1.1, 1, 0.9, 0.9]

    discharge = [500, 510, 650, 660, 750, 760, 850, 860, 950, 960, 1050, 1060, 1150, 1160, 1250, 1260, 1350, 1360, 1750]
    discharge_sp = [0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 0]

    water = [250, 375, 750, 800, 850, 1000, 1250, 1350, 1750]
    water_sp = [0, 500, 100, 0, 400, 150, 250, 0, 100]

    PAA = [25, 200, 1000, 1500, 1750]
    PAA_sp = [5, 0, 10, 4, 0]

    def get_nonzero_sp_index(t, s):
        return [t[i] for i, x in enumerate(s) if x > 0]

    allowed_time = Fs + Foil + Fg + pres + discharge + water + PAA

    return sorted(list(set([x for x in allowed_time if x < 1150])))

#act_times = get_actiontime()

agent = PPO(obs_dim, act_dim, buffer_size=1150)

"""run one episode"""
avg_yields = []
t = time.time()

for e in range(episodes):
    env = PenSimEnv(random_seed_ref=666)
    done = False
    batch_data = env.reset()
    recipe = Recipe()

    time_stamp, batch_yield = 0, 0
    update_recipe = False
    while not done:
        # time is from 1 to 1150
        time_stamp += 1

        # Get action from recipe agent based on time
        Fs, Foil, Fg, Fpres, Fdischarge, Fw, Fpaa = recipe.run(time_stamp)

        if update_recipe:
            Fs *= (1+Fs_a)
            Foil *= (1+Foil_a)
            Fg *= (1+Fg_a)
            Fpres *= (1+Fpres_a)
            Fdischarge *= (1+Fdischarge_a)
            Fw *= (1+Fw_a)
            Fpaa *= (1+Fpaa_a)
            #update_recipe = False

        observation, batch_data, reward, done = env.step(time_stamp,
                                                         batch_data,
                                                         Fs, Foil, Fg, Fpres, max(Fdischarge, 0), max(Fw, 0), max(Fpaa, 0))

        #if time_stamp in act_times:
        o = np.array([x[1] for x in observation]) # observation array for action
        """agent is allowed to take action here"""
        a, v, logp = agent.ac.step(torch.as_tensor(o, dtype=torch.float32))
        #print(a)
        """add adjustment to each action"""
        Fs_a, Foil_a, Fg_a, Fpres_a, Fdischarge_a, Fw_a, Fpaa_a = list(np.clip(a/10, -0.1, 0.1))
        update_recipe = True

        """in this setting we just collect reward in the end of the batch"""
        #if time_stamp != act_times[-1]:
        if not done:
            r = 0
            agent.buffer.store(o, a, r, v, logp)

        batch_yield += reward
    agent.buffer.store(o, a, batch_yield, v, logp)
    agent.buffer.finish_path()
    agent.update()

    print(f"episode: {e}, elapsed time: {int(time.time() - t)} s, batch_yield: {batch_yield}")
    avg_yields.append(batch_yield)

print(np.mean(avg_yields))
