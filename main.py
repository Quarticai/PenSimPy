import time
from pensimpy.pensim_classes.Recipe import Recipe
import numpy as np
from pensimpy.pensim_methods.indpensim import indpensim
from pensimpy.env_setup.peni_env_setup import PenSimEnv
import math
from pensimpy.pensim_classes.Constants import raman_wavenumber
from pensimpy.pensim_methods.create_batch import create_batch
from pensimpy.helper.show_params import show_params

'''
env = PenSimEnv()
done = False
observation = env.reset()
recipe_agent = Recipe() # strictly follows the recipe
time_stamp = 0
batch_yield = 0 # keep track of the yield
while not done:
    time_stamp += 1
    action = recipe_agent(t, observation)
    reward, observation, done = env.step(action)
    batch_yield += reward
'''

if __name__ == "__main__":
    t = time.time()

    env = PenSimEnv()
    xinterp, x0, H, Batch_lenght, param_list, ctrl_flags = env.reset()
    recipe = Recipe()

    time_stamp = 0
    x = create_batch(H, Batch_lenght)
    tmp = 0
    yield_sum = 0
    while time_stamp != int(Batch_lenght / H):
        # time is from 1 to 1150
        time_stamp += 1

        # Get action from recipe agent based on time
        Fs, Foil, Fg, Fpres, Fdischarge, Fw, Fpaa = recipe.run(time_stamp - 1)

        x = indpensim(time_stamp, x, xinterp, x0, H, Batch_lenght, param_list, ctrl_flags,
                      Fs, Foil, Fg, Fpres, Fdischarge, Fw, Fpaa)

        peni_yield = x.V.y[time_stamp - 1] * x.P.y[time_stamp - 1] / 1000
        yield_per = peni_yield - tmp - x.Fremoved.y[time_stamp-1] * x.P.y[time_stamp-1] * H / 1000
        print(f"=== yield: {yield_per}")
        yield_sum += yield_per
        tmp = peni_yield

    print(yield_sum)

    # convert to pH from H+ concentration
    x.pH.y = [-math.log(pH) / math.log(10) if pH != 0 else pH for pH in x.pH.y]
    x.Q.y = [Q / 1000 for Q in x.Q.y]
    x.Raman_Spec.Wavenumber = raman_wavenumber

    print(f"=== cost: {int(time.time() - t)} s")
    penicillin_yield_total = (x.V.y[-1] * x.P.y[-1] - np.dot(x.Fremoved.y, x.P.y) * H) / 1000
    print(f"=== penicillin_yield: {penicillin_yield_total}")

    # # Plot the last res
    # show_params(x)
