import time
from pensimpy.pensim_classes.Recipe import Recipe
from pensimpy.env_setup.peni_env_setup import PenSimEnv
from pensimpy.helper.show_params import show_params
import numpy as np

if __name__ == "__main__":
    yield_records = []
    t = time.time()

    # Random_seed_ref from 0 to 1000
    env = PenSimEnv(random_seed_ref=np.random.randint(1000))
    done = False
    observation, batch_data = env.reset()

    # default
    x = [8, 15, 30, 75, 150, 30, 37, 43, 47, 51, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80,
         22, 30, 35, 34, 33, 32, 31, 30, 29, 23,
         30, 42, 55, 60, 75, 65, 60,
         0.6, 0.7, 0.8, 0.9, 1.1, 1, 0.9, 0.9,
         0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 0,
         0, 500, 100, 0, 400, 150, 250, 0, 100]

    recipe = Recipe(x)

    time_stamp, batch_yield, yield_pre = 0, 0, 0
    while not done:
        # time is from 1 to 1150
        time_stamp += 1

        # Get action from recipe agent based on time
        Fs, Foil, Fg, pressure, Fremoved, Fw, Fpaa = recipe.run(time_stamp)

        # Run and get the reward
        # observation is a class which contains all the variables, e.g. observation.Fs.y[k], observation.Fs.t[k]
        # are the Fs value and corresponding time at k
        observation, batch_data, reward, done = env.step(time_stamp,
                                                         batch_data,
                                                         Fs, Foil, Fg, pressure, Fremoved, Fw, Fpaa)
        batch_yield += reward

    print(f"=== Time cost: {int(time.time() - t)} s")
    print(f"=== Yield: {batch_yield}")
    yield_records.append(batch_yield)

    # # Plot
    # show_params(batch_data)
