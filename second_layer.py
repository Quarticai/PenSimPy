from pensimpy.helper.get_recipe_trend import get_recipe_trend
from pensimpy.env_setup.peni_env_setup import PenSimEnv
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer


class RecipeBuilder:
    """builds the recipe that can run with the env, given setpoints"""

    def __init__(self, random_int):
        self.Fs_trend = None
        self.Foil_trend = None
        self.Fg_trend = None
        self.pres_trend = None
        self.discharge_trend = None
        self.water_trend = None
        self.random_int = random_int

    def init_recipe(self, Fs_sp, Foil_sp, Fg_sp, pres_sp, discharge_sp, water_sp):
        Fs = [15, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 800, 1750]
        self.Fs_trend = get_recipe_trend(Fs, Fs_sp)
        Foil = [20, 80, 280, 300, 320, 340, 360, 380, 400, 1750]
        self.Foil_trend = get_recipe_trend(Foil, Foil_sp)
        Fg = [40, 100, 200, 450, 1000, 1250, 1750]
        self.Fg_trend = get_recipe_trend(Fg, Fg_sp)
        pres = [62, 125, 150, 200, 500, 750, 1000, 1750]
        self.pres_trend = get_recipe_trend(pres, pres_sp)
        discharge = [500, 510, 650, 660, 750, 760, 850, 860, 950, 960, 1050,
                     1060, 1150, 1160, 1250, 1260, 1350, 1360, 1750]
        self.discharge_trend = get_recipe_trend(discharge, discharge_sp)
        water = [250, 375, 750, 800, 850, 1000, 1250, 1350, 1750]
        self.water_trend = get_recipe_trend(water, water_sp)

        # exception
        PAA = [25, 200, 1000, 1500, 1750]
        PAA_sp = [5, 0, 10, 4, 0]
        self.PAA_trend = get_recipe_trend(PAA, PAA_sp)

    def recipe_at_t(self, t):
        t -= 1
        return self.Fs_trend[t], self.Foil_trend[t], self.Fg_trend[t], self.pres_trend[t], self.discharge_trend[t], \
               self.water_trend[t], self.PAA_trend[t]

    def split(self, x):
        Fs_len, Foil_len, Fg_len, pres_len, discharge_len, water_len = 21, 10, 7, 8, 20, 9
        recipe_Fs_sp = x[:Fs_len]
        recipe_Foil_sp = x[Fs_len:
                           Fs_len + Foil_len]
        recipe_Fg_sp = x[Fs_len + Foil_len:
                         Fs_len + Foil_len + Fg_len]
        recipe_pres_sp = x[Fs_len + Foil_len + Fg_len:
                           Fs_len + Foil_len + Fg_len + pres_len]
        recipe_discharge_sp = x[Fs_len + Foil_len + Fg_len + pres_len:
                                Fs_len + Foil_len + Fg_len + pres_len + discharge_len]
        recipe_water_sp = x[Fs_len + Foil_len + Fg_len + pres_len + discharge_len:
                            Fs_len + Foil_len + Fg_len + pres_len + discharge_len + water_len]

        return recipe_Fs_sp, recipe_Foil_sp, recipe_Fg_sp, recipe_pres_sp, recipe_discharge_sp, recipe_water_sp

    def get_batch_yield(self, sp_points):
        Fs_sp, Foil_sp, Fg_sp, pres_sp, discharge_sp, water_sp = self.split(sp_points)

        env = PenSimEnv(random_seed_ref=self.random_int)
        #env = PenSimEnv(random_seed_ref=np.random.randint(1000))
        done = False
        observation, batch_data = env.reset()
        self.init_recipe(Fs_sp, Foil_sp, Fg_sp, pres_sp, discharge_sp, water_sp)
        time_stamp, batch_yield, yield_pre = 0, 0, 0
        while not done:
            time_stamp += 1
            Fs, Foil, Fg, Fpres, Fdischarge, Fw, Fpaa = self.recipe_at_t(time_stamp)
            observation, batch_data, reward, done = env.step(time_stamp,
                                                             batch_data,
                                                             Fs, Foil, Fg, Fpres, Fdischarge, Fw, Fpaa)
            batch_yield += reward
        return -batch_yield

    def get_bound(self, x, x_opt, manup_scale):
        lower_bound = x - x * manup_scale
        upper_bound = x + x * manup_scale

        lower_bound_new = x_opt - x_opt * manup_scale
        upper_bound_new = x_opt + x_opt * manup_scale

        lower_bound_new = lower_bound_new if lower_bound_new > lower_bound else lower_bound
        upper_bound_new = upper_bound_new if upper_bound_new < upper_bound else upper_bound

        return lower_bound, upper_bound


    def benchmark(self, total_calls, n_calls, n_random_starts, manup_scale):
        # default
        x = [8, 15, 30, 75, 150, 30, 37, 43, 47, 51, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80,
             22, 30, 35, 34, 33, 32, 31, 30, 29, 23,
             30, 42, 55, 60, 75, 65, 60,
             0.6, 0.7, 0.8, 0.9, 1.1, 1, 0.9, 0.9,
             0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 0,
             0, 500, 100, 0, 400, 150, 250, 0, 100]

        # 1000
        # x0 = [7, 15, 27, 78, 163, 31, 33, 39, 50, 52, 54, 58, 62, 64, 71, 82, 89, 94, 116, 98, 87, 24, 31, 37, 31, 34, 31, 29, 30, 30, 25, 33, 40, 56, 66, 75, 68, 54, 0.5554127758991113, 0.7676077357976071, 0.768911753698651, 0.9365571601116198, 1.0861150998563076, 0.9931536974459727, 0.8776714419262855, 0.9493535950911615, 1, 4235, 1, 4309, 1, 4090, 0, 3875, 0, 3899, 1, 4280, 1, 4018, 0, 3795, 0, 4301, 1, 1, 1, 450, 90, 0, 366, 136, 227, 1, 94]

        # 100
        x0 = [8, 16, 28, 81, 137, 28, 38, 43, 48, 48, 56, 66, 60, 78, 77, 87, 80, 84, 104, 90, 82, 23, 29, 35, 30, 31, 28, 33, 31, 26, 25, 31, 37, 53, 61, 75, 64, 59, 0.6468329938975121, 0.7188338813762886, 0.8800000000000001, 0.8655090552916249, 0.9900000000000001, 1.0806341623771156, 0.99, 0.9193644054260693, 1, 3715, 1, 3926, 0, 4223, 0, 3600, 1, 4077, 1, 4003, 1, 3646, 1, 3790, 1, 4236, 1, 0, 1, 528, 91, 0, 418, 154, 270, 1, 105]


        num_iter = 0

        space = []
        yields_summary = []
        recipe_summary = []

        while total_calls > 0:
            print(f"=== === running iter @ {num_iter} with seed {self.random_int}")

            if num_iter == 0:
                recipe_Fs_sp, recipe_Foil_sp, recipe_Fg_sp, \
                recipe_pres_sp, recipe_discharge_sp, recipe_water_sp = self.split(x)

                recipe_Fs_sp_opt, recipe_Foil_sp_opt, recipe_Fg_sp_opt, \
                recipe_pres_sp_opt, recipe_discharge_sp_opt, recipe_water_sp_opt = self.split(x0)

                for Fs, Fs_opt in zip(recipe_Fs_sp, recipe_Fs_sp_opt):
                    lower_bound, upper_bound = self.get_bound(Fs, Fs_opt, manup_scale)
                    space.append(Integer(int(lower_bound), int(upper_bound)))

                for Foil, Foil_opt in zip(recipe_Foil_sp, recipe_Foil_sp_opt):
                    lower_bound, upper_bound = self.get_bound(Foil, Foil_opt, manup_scale)
                    space.append(Integer(int(lower_bound), int(upper_bound)))

                for Fg, Fg_opt in zip(recipe_Fg_sp, recipe_Fg_sp_opt):
                    lower_bound, upper_bound = self.get_bound(Fg, Fg_opt, manup_scale)
                    space.append(Integer(int(lower_bound), int(upper_bound)))

                for pres, pres_opt in zip(recipe_pres_sp, recipe_pres_sp_opt):
                    lower_bound, upper_bound = self.get_bound(pres, pres_opt, manup_scale)
                    space.append(Real(lower_bound, upper_bound))

                for discharge, discharge_opt in zip(recipe_discharge_sp, recipe_discharge_sp_opt):
                    if discharge != 0:
                        lower_bound, upper_bound = self.get_bound(discharge, discharge_opt, manup_scale)
                        space.append(Integer(int(lower_bound), int(upper_bound)))
                    else:
                        space.append(Integer(0, 1))

                for water, water_opt in zip(recipe_water_sp, recipe_water_sp_opt):
                    if water != 0:
                        lower_bound, upper_bound = self.get_bound(water, water_opt, manup_scale)
                        space.append(Integer(int(lower_bound), int(upper_bound)))
                    else:
                        space.append(Integer(0, 1))


            res_gp = gp_minimize(self.get_batch_yield,
                                 space,
                                 x0=x0,
                                 n_calls=n_calls,
                                 n_random_starts=n_random_starts,
                                 random_state=np.random.randint(1000),
                                 n_jobs=-1)

            num_iter += 1
            total_calls -= n_calls
            # x0 = res_gp.x
            yields_summary.extend(res_gp.func_vals.tolist())
            recipe_summary.extend(res_gp.x_iters)

        return yields_summary, recipe_summary


for seed in range(1, 101):
    recipe_builder = RecipeBuilder(random_int=seed)
    yields, recipes = recipe_builder.benchmark(total_calls=100, n_calls=100, n_random_starts=4, manup_scale=0.1)
    print(len(yields))
    print(len(recipes))
    print(yields)

    # import matplotlib.pyplot as plt
    #
    # new_yields = [-ele for ele in yields]
    #
    # import numpy as np
    #
    # print(np.mean(new_yields))
    # print(np.std(new_yields))
    # print(np.min(new_yields))
    # print(np.max(new_yields))
    #
    # plt.plot(new_yields)
    # plt.show()

    import pickle

    with open(f'100_2nd_layer_{seed}_office_yield', 'wb') as fp:
        pickle.dump(yields, fp)

    with open(f'100_2nd_layer_{seed}_office_recipe', 'wb') as fp:
        pickle.dump(recipes, fp)

# import pickle
#
# with open('1000_100_1st_layer_office_yield', 'rb') as fp:
#     yields = pickle.load(fp)
#
# with open('1000_100_1st_layer_office_recipe', 'rb') as fp:
#     recipe = pickle.load(fp)
#
# new_yields = [-ele for ele in yields]
# print(max(new_yields))
# print(new_yields.index((max(new_yields))))
# print(recipe[new_yields.index((max(new_yields)))])
#
# top_1_recipe = [7, 15, 27, 78, 163, 31, 33, 39, 50, 52, 54, 58, 62, 64, 71, 82, 89, 94, 116, 98, 87, 24, 31, 37, 31, 34, 31, 29, 30, 30, 25, 33, 40, 56, 66, 75, 68, 54, 0.5554127758991113, 0.7676077357976071, 0.768911753698651, 0.9365571601116198, 1.0861150998563076, 0.9931536974459727, 0.8776714419262855, 0.9493535950911615, 1, 4235, 1, 4309, 1, 4090, 0, 3875, 0, 3899, 1, 4280, 1, 4018, 0, 3795, 0, 4301, 1, 1, 1, 450, 90, 0, 366, 136, 227, 1, 94]
#
# import matplotlib.pyplot as plt
# plt.plot(new_yields)
# plt.show()
# # import numpy as np
# # print(np.mean(new_yields))
# # print(np.std(new_yields))
# # print(np.min(new_yields))
# # print(np.max(new_yields))
#
# # recipe_builder.botorch()
# # recipe_builder.gpyopt()
