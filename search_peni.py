from pensimpy.helper.get_recipe_trend import get_recipe_trend
from pensimpy.env_setup.peni_env_setup import PenSimEnv
import numpy as np
from random import random, seed
from skopt import gp_minimize
from skopt.space import Real, Integer
import math


# from gpflowopt.domain import ContinuousParameter
# from gpflowopt.design import LatinHyperCube, FactorialDesign, RandomDesign
# from gpflowopt.bo import BayesianOptimizer
# from gpflowopt.acquisition import ExpectedImprovement
# import gpflow
# from gpflowopt.optim import SciPyOptimizer


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

    def get_batch_yield_gpflow(self, X):
        x = [8, 15, 30, 75, 150, 30, 37, 43, 47, 51, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80,
             22, 30, 35, 34, 33, 32, 31, 30, 29, 23,
             30, 42, 55, 60, 75, 65, 60,
             0.6, 0.7, 0.8, 0.9, 1.1, 1, 0.9, 0.9,
             0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 0,
             0, 500, 100, 0, 400, 150, 250, 0, 100]
        yields = []
        for item in X:
            # water_sp = X[i].tolist()
            # Fs_sp, Foil_sp, Fg_sp, pres_sp, discharge_sp, _ = self.split(x)

            Fs_sp, Foil_sp, Fg_sp, pres_sp, discharge_sp, water_sp = self.split(
                item.tolist()
            )


            env = PenSimEnv(random_seed_ref=self.random_int)
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
            yields.append(-batch_yield)

        yields = np.array(yields)
        return yields[:, None]

    def get_bound(self, x, x_opt, manup_scale):
        lower_bound = x - x * manup_scale
        upper_bound = x + x * manup_scale

        lower_bound_new = x_opt - x_opt * manup_scale * 0.2
        upper_bound_new = x_opt + x_opt * manup_scale * 0.2

        lower_bound_new = lower_bound_new if lower_bound_new > lower_bound else lower_bound
        upper_bound_new = upper_bound_new if upper_bound_new < upper_bound else upper_bound

        return lower_bound_new, upper_bound_new

    def botorch(self):
        import torch
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_model
        from gpytorch.mlls import ExactMarginalLogLikelihood

        x = [8, 15, 30, 75, 150, 30, 37, 43, 47, 51, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80,
             22, 30, 35, 34, 33, 32, 31, 30, 29, 23,
             30, 42, 55, 60, 75, 65, 60,
             0.6, 0.7, 0.8, 0.9, 1.1, 1, 0.9, 0.9,
             0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 0,
             0, 500, 100, 0, 400, 150, 250, 0, 100]

        X = []
        for _ in range(10):
            tmp = [ele * (1 + np.random.randint(-10, 10) / 100) for ele in x]
            X.append(tmp)

        train_X = torch.FloatTensor(X)
        print(f"=== train_X: {train_X.shape}")

        Y = self.get_batch_yield_gpflow(train_X)
        train_Y = torch.FloatTensor(Y)

        print(f"=== train_Y: {train_Y.shape}")
        gp = SingleTaskGP(train_X, train_Y)
        # mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        # fit_gpytorch_model(mll)

        from botorch.acquisition import UpperConfidenceBound

        UCB = UpperConfidenceBound(gp, beta=0.1)

        from botorch.optim import optimize_acqf

        lower = [ele * 0.9 if ele != 0 else ele for ele in x]
        upper = [ele * 1.1 if ele != 0 else 1 for ele in x]

        bounds = torch.stack([torch.FloatTensor(lower), torch.FloatTensor(upper)])
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
        )

        print(f"=== candidate: {candidate}")
        print(f"=== acq_value: {acq_value}")
        print(f"=== actual yield: {self.get_batch_yield_gpflow(candidate)}")



    def gpflow_opt(self):
        # default
        x = [8, 15, 30, 75, 150, 30, 37, 43, 47, 51, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80,
             22, 30, 35, 34, 33, 32, 31, 30, 29, 23,
             30, 42, 55, 60, 75, 65, 60,
             0.6, 0.7, 0.8, 0.9, 1.1, 1, 0.9, 0.9,
             0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 0,
             0, 500, 100, 0, 400, 150, 250, 0, 100]
        lower = [ele * 0.9 if ele != 0 else ele for ele in x]
        upper = [ele * 1.1 if ele != 0 else 1 for ele in x]
        domain = np.sum([ContinuousParameter('x{0}'.format(i), l, u) for i, l, u in zip(range(len(x)), lower, upper)])
        design = RandomDesign(10, domain)
        X = design.generate()
        print(f"==== X: {X}")
        print(f"==== X: {len(X)}")

        domain1 = np.sum([ContinuousParameter('y{0}'.format(i), l, u) for i, l, u in zip(range(len(x)), x, x)])
        x0 = RandomDesign(1, domain1)

        Y = self.get_batch_yield_gpflow(X)

        # initializing a standard BO model, Gaussian Process Regression with
        # Matern52 ARD Kernel
        model = gpflow.gpr.GPR(X, Y, gpflow.kernels.Matern52(domain.size, ARD=True))
        alpha = ExpectedImprovement(model)

        # Now we must specify an optimization algorithm to optimize the acquisition
        # function, each iteration.
        acqopt = SciPyOptimizer(domain)

        # Now create the Bayesian Optimizer
        optimizer = BayesianOptimizer(domain, alpha, optimizer=acqopt, initial=x0, verbose=True)
        # with optimizer.silent():
        r = optimizer.optimize(self.get_batch_yield_gpflow, n_iter=10)
        print(r)

    def benchmark(self, total_calls, n_calls, n_random_starts, manup_scale):
        # default
        x = [8, 15, 30, 75, 150, 30, 37, 43, 47, 51, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80,
             22, 30, 35, 34, 33, 32, 31, 30, 29, 23,
             30, 42, 55, 60, 75, 65, 60,
             0.6, 0.7, 0.8, 0.9, 1.1, 1, 0.9, 0.9,
             0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 0,
             0, 500, 100, 0, 400, 150, 250, 0, 100]

        # top 1
        # x0 = [8, 15, 31, 76, 144, 32, 39, 44, 44, 48, 53, 57, 60, 79, 82, 74, 87, 96, 105, 99, 88, 23, 30, 37, 32, 34,
        #       34, 27, 29, 30, 25, 32, 39, 56, 64, 76, 61, 54, 0.5751669730340376, 0.7161287257301532,
        #       0.7465629211339744, 0.9488068117056693, 1.130591290628027, 1.068799517372909, 0.81, 0.8428951880325852, 1,
        #       3673, 1, 4158, 1, 4073, 0, 4229, 1, 3821, 1, 3869, 1, 4129, 1, 3906, 1, 4328, 1, 1, 1, 547, 93, 1, 394,
        #       145, 225, 1, 101]
        x0 = [8.8, 16.5, 31.624252044238425, 80.13474509247575, 165.0, 30.763382203093368, 40.7, 47.3,
              48.05716368821474, 55.15358172418276, 51.4398158471111, 61.6493571632683, 65.67022327424863,
              77.96346763446553, 71.3782077191417, 77.23046926307548, 92.4, 90.79850035453829, 114.03714549578332, 99.0,
              88.0, 21.96813367025686, 33.0, 38.5, 36.24626215254306, 29.7, 35.2, 28.2419828653085, 27.286406931581023,
              29.426747426092, 25.3, 32.432773357247996, 40.027278748110305, 50.19868927568485, 66.0, 78.05107828935165,
              64.45022060032773, 57.95664924901469, 0.56851933895408, 0.7699999999999999, 0.72, 0.8263110434540011,
              1.1403701584391397, 1.0823399050137648, 0.8481730193616307, 0.9519224931750232, 0.0009510521763949335,
              3803.1322001216763, 0.0004744015345583332, 4169.273038967871, 0.0007940386276741609, 4245.208859117792,
              0.0, 3600.0, 0.001, 4193.088586277349, 0.0004293134419313078, 4166.552905585846, 0.0, 3746.7115958942272,
              0.0, 3806.434973903112, 0.0006842283439804432, 3983.074516003572, 0.00041511053507232776,
              0.0006349914867663446, 0.0006841858874726173, 453.56974515739904, 100.11317471293015,
              0.00034233634278272326, 363.3829129190667, 161.6677187701564, 229.2738817253559, 0.000524502413540419,
              90.58189370332654]
        # top 5
        # x0 = [[8, 15, 30, 67, 135, 32, 33, 46, 48, 51, 51, 64, 58, 64, 68, 83, 75, 98, 106, 99, 88, 23, 33, 38, 30, 34, 30, 30, 29, 26, 25, 27, 37, 49, 64, 72, 62, 62, 0.5937612142062656, 0.7369915508687243, 0.8800000000000001, 0.8876627396891, 0.9900000000000001, 1.1, 0.8912359957349242, 0.99, 0, 4400, 1, 4400, 1, 3847, 1, 3600, 0, 4102, 0, 3789, 0, 4400, 1, 4001, 1, 3600, 0, 1, 0, 550, 93, 0, 360, 142, 275, 1, 99], [8, 15, 32, 80, 138, 28, 40, 44, 42, 50, 52, 62, 71, 71, 78, 76, 79, 91, 117, 99, 88, 22, 33, 34, 33, 36, 33, 33, 33, 26, 25, 30, 42, 51, 64, 72, 66, 58, 0.5600260382388, 0.6775755933903049, 0.8360947394917699, 0.873097356580318, 1.1833480885181498, 0.9013045598167755, 0.8381264271897598, 0.8155621834786863, 0, 3933, 1, 4093, 0, 4160, 1, 3999, 0, 3977, 0, 3633, 0, 4056, 1, 4320, 1, 3907, 0, 1, 0, 470, 101, 0, 393, 141, 254, 1, 97], [7, 15, 32, 67, 135, 33, 40, 45, 51, 56, 58, 54, 63, 76, 68, 88, 92, 91, 104, 99, 88, 19, 33, 37, 37, 29, 32, 33, 31, 27, 25, 33, 46, 49, 66, 82, 61, 54, 0.5887538683908805, 0.7699999999999999, 0.79869450478647, 0.9129025907063256, 1.176962859932942, 1.1, 0.99, 0.99, 0, 3799, 1, 3600, 1, 4359, 0, 3987, 1, 4164, 0, 4400, 1, 4400, 0, 4238, 1, 3600, 0, 1, 1, 550, 102, 0, 417, 164, 225, 1, 90], [8, 14, 27, 77, 165, 29, 40, 40, 44, 56, 58, 65, 61, 69, 83, 80, 92, 99, 106, 99, 88, 20, 28, 36, 30, 33, 35, 28, 33, 28, 25, 27, 46, 54, 66, 82, 71, 55, 0.6599999999999999, 0.63, 0.8800000000000001, 0.8321717898263531, 1.1044391884714464, 0.9, 0.8521349624603344, 0.81, 1, 3778, 1, 3978, 1, 4036, 0, 3842, 0, 4400, 1, 3777, 1, 4203, 0, 3801, 0, 4400, 1, 1, 0, 462, 110, 1, 367, 141, 275, 1, 109], [8, 15, 31, 76, 144, 32, 39, 44, 44, 48, 53, 57, 60, 79, 82, 74, 87, 96, 105, 99, 88, 23, 30, 37, 32, 34, 34, 27, 29, 30, 25, 32, 39, 56, 64, 76, 61, 54, 0.5751669730340376, 0.7161287257301532, 0.7465629211339744, 0.9488068117056693, 1.130591290628027, 1.068799517372909, 0.81, 0.8428951880325852, 1, 3673, 1, 4158, 1, 4073, 0, 4229, 1, 3821, 1, 3869, 1, 4129, 1, 3906, 1, 4328, 1, 1, 1, 547, 93, 1, 394, 145, 225, 1, 101]]

        num_iter = 0

        space = []

        while total_calls > 0:
            # print(f"=== running iter {num_iter}")
            # print(f"=== x: {x}")
            # manup_scale = 0.08 * math.exp(-0.27 * num_iter) + 0.02
            # print(f"=== manup_scale: {manup_scale}")
            total_calls -= n_calls

            recipe_Fs_sp, recipe_Foil_sp, recipe_Fg_sp, \
            recipe_pres_sp, recipe_discharge_sp, recipe_water_sp = self.split(x)

            recipe_Fs_sp_opt, recipe_Foil_sp_opt, recipe_Fg_sp_opt, \
            recipe_pres_sp_opt, recipe_discharge_sp_opt, recipe_water_sp_opt = self.split(x0)

            if num_iter == 0:
                for Fs, Fs_opt in zip(recipe_Fs_sp, recipe_Fs_sp_opt):
                    lower_bound, upper_bound = self.get_bound(Fs, Fs_opt, manup_scale)
                    space.append(Real(lower_bound, upper_bound))

                for Foil, Foil_opt in zip(recipe_Foil_sp, recipe_Foil_sp_opt):
                    lower_bound, upper_bound = self.get_bound(Foil, Foil_opt, manup_scale)
                    space.append(Real(lower_bound, upper_bound))

                for Fg, Fg_opt in zip(recipe_Fg_sp, recipe_Fg_sp_opt):
                    lower_bound, upper_bound = self.get_bound(Fg, Fg_opt, manup_scale)
                    space.append(Real(lower_bound, upper_bound))

                for pres, pres_opt in zip(recipe_pres_sp, recipe_pres_sp_opt):
                    lower_bound, upper_bound = self.get_bound(pres, pres_opt, manup_scale)
                    space.append(Real(lower_bound, upper_bound))

                for discharge, discharge_opt in zip(recipe_discharge_sp, recipe_discharge_sp_opt):
                    if discharge != 0:
                        lower_bound, upper_bound = self.get_bound(discharge, discharge_opt, manup_scale)
                        space.append(Real(lower_bound, upper_bound))
                    else:
                        space.append(Real(0, 0.001))

                for water, water_opt in zip(recipe_water_sp, recipe_water_sp_opt):
                    if water != 0:
                        lower_bound, upper_bound = self.get_bound(water, water_opt, manup_scale)
                        space.append(Real(lower_bound, upper_bound))
                    else:
                        space.append(Real(0, 0.001))

            num_iter += 1
            res_gp = gp_minimize(self.get_batch_yield,
                                 space,
                                 x0=x0,
                                 n_calls=n_calls,
                                 n_random_starts=n_random_starts,
                                 random_state=np.random.randint(1000),
                                 n_jobs=-1)

            print(f"=== mean is {np.mean(res_gp.func_vals)}")
            x = res_gp.x

        return res_gp.func_vals.tolist(), res_gp.x_iters


yields, recipes = [], []
recipe_builder = RecipeBuilder(random_int=274)
# yields, recipes = recipe_builder.benchmark(total_calls=5, n_calls=5, n_random_starts=1, manup_scale=0.1)
# print(yields)
# print(recipes)

recipe_builder.botorch()
