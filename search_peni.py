from pensimpy.helper.get_recipe_trend import get_recipe_trend
from pensimpy.env_setup.peni_env_setup import PenSimEnv

import numpy as np
from random import random, seed


class RecipeBuilder:
    """builds the recipe that can run with the env, given setpoints"""

    def __init__(self, Fs_sp, pres_sp):
        # recipes
        Fs = [15, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 800, 1750]
        assert len(Fs_sp) == len(Fs)

        # [{"step": t, "value": fs } for t, fs in zip(Fs, Fs_sp)]
        self.Fs_trend = get_recipe_trend(Fs, Fs_sp)

        Foil = [20, 80, 280, 300, 320, 340, 360, 380, 400, 1750]
        Foil_sp = [22, 30, 35, 34, 33, 32, 31, 30, 29, 23]
        self.Foil_trend = get_recipe_trend(Foil, Foil_sp)

        Fg = [40, 100, 200, 450, 1000, 1250, 1750]
        Fg_sp = [30, 42, 55, 60, 75, 65, 60]
        self.Fg_trend = get_recipe_trend(Fg, Fg_sp)

        pres = [62, 125, 150, 200, 500, 750, 1000, 1750]
        assert len(pres) == len(pres)
        self.pres_trend = get_recipe_trend(pres, pres_sp)

        discharge = [500, 510, 650, 660, 750, 760, 850, 860, 950, 960, 1050, 1060, 1150, 1160, 1250, 1260, 1350, 1360,
                     1750]
        discharge_sp = [0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 0]
        self.discharge_trend = get_recipe_trend(discharge, discharge_sp)

        water = [250, 375, 750, 800, 850, 1000, 1250, 1350, 1750]
        water_sp = [0, 500, 100, 0, 400, 150, 250, 0, 100]
        self.water_trend = get_recipe_trend(water, water_sp)

        PAA = [25, 200, 1000, 1500, 1750]
        PAA_sp = [5, 0, 10, 4, 0]
        self.PAA_trend = get_recipe_trend(PAA, PAA_sp)

    def run(self, t):
        t -= 1
        return self.Fs_trend[t], self.Foil_trend[t], self.Fg_trend[t], self.pres_trend[t], self.discharge_trend[t], \
               self.water_trend[t], self.PAA_trend[t]


recipe_Fs_sp = [8, 15, 30, 75, 150, 30, 37, 43, 47, 51, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80]
recipe_pres_sp = [0.6, 0.7, 0.8, 0.9, 1.1, 1, 0.9, 0.9]
# ################## Recipy Policy
# print("Recipy Policy")
# num_batches = 1
# batch_yields = []  # record yield per batch
# for b in range(num_batches):
#     env = PenSimEnv(random_seed_ref=123)
#     done = False
#     batch_data = env.reset()
#     observation = []
#     recipe = RecipeBuilder(Fs_sp=recipe_Fs_sp,
#                            pres_sp=recipe_pres_sp)
#     time_stamp, batch_yield, yield_pre = 0, 0, 0
#     while not done:
#         time_stamp += 1
#         Fs, Foil, Fg, Fpres, Fdischarge, Fw, Fpaa = recipe.run(time_stamp)
#         observation, batch_data, reward, done = env.step(time_stamp,
#                                                          batch_data,
#                                                          Fs, Foil, Fg, Fpres, Fdischarge, Fw, Fpaa)
#         batch_yield += reward
#     print(f"=== batch_yield: {batch_yield}")
#     batch_yields.append(batch_yield)
# print(np.mean(batch_yields))
#
# ################## Random Policy
# print("Random Policy")
# seed(123)
#
#
# def randomize_sp(sp, random_scale=0.1, cast=float):
#     return [cast(x * (1 + (-1 if random() < 0.5 else 1) * random() * random_scale)) for x in sp]
#
#
# num_batches = 1
# batch_yields = []  # record yield per batch
# recipe_Fs_sp = [8, 15, 30, 75, 150, 30, 37, 43, 47, 51, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80]
# recipe_pres_sp = [0.6, 0.7, 0.8, 0.9, 1.1, 1, 0.9, 0.9]
# for b in range(num_batches):
#     env = PenSimEnv(random_seed_ref=123)
#     done = False
#     batch_data = env.reset()
#     observation = []
#     recipe = RecipeBuilder(Fs_sp=randomize_sp(recipe_Fs_sp, cast=int),
#                            pres_sp=randomize_sp(recipe_pres_sp, cast=float))
#     time_stamp, batch_yield, yield_pre = 0, 0, 0
#     while not done:
#         time_stamp += 1
#         Fs, Foil, Fg, Fpres, Fdischarge, Fw, Fpaa = recipe.run(time_stamp)
#         observation, batch_data, reward, done = env.step(time_stamp,
#                                                          batch_data,
#                                                          Fs, Foil, Fg, Fpres, Fdischarge, Fw, Fpaa)
#         batch_yield += reward
#     print(f"=== batch_yield: {batch_yield}")
#     batch_yields.append(batch_yield)

################## Bayesian Optimization Policy
from skopt import gp_minimize
from skopt.space import Real, Integer

"""defines the search space"""
space = []
manup_scale = 0.1
for Fs in recipe_Fs_sp:
    space.append(Integer(int(Fs - Fs * manup_scale), int(Fs + Fs * manup_scale)))

for pres in recipe_pres_sp:
    space.append(Real(pres - pres * manup_scale, pres + pres * manup_scale))


def get_batch_yield(sp_points):
    """return negative batch yield given all the set points"""
    Fs_sp = sp_points[:len(recipe_Fs_sp)]
    pres_sp = sp_points[len(recipe_Fs_sp):]
    env = PenSimEnv(random_seed_ref=123)
    done = False
    batch_data = env.reset()
    observation = []
    recipe = RecipeBuilder(Fs_sp, pres_sp)
    time_stamp, batch_yield, yield_pre = 0, 0, 0
    while not done:
        time_stamp += 1
        Fs, Foil, Fg, Fpres, Fdischarge, Fw, Fpaa = recipe.run(time_stamp)
        observation, batch_data, reward, done = env.step(time_stamp,
                                                         batch_data,
                                                         Fs, Foil, Fg, Fpres, Fdischarge, Fw, Fpaa)

        batch_yield += reward
    return -batch_yield


res_gp = gp_minimize(get_batch_yield, space, n_calls=10, n_random_starts=10, random_state=123, n_jobs=-1)
print(res_gp.func_vals)
print(np.mean(res_gp.func_vals))
