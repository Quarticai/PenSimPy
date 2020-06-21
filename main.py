import time
from pensimpy.pensim_classes.Recipe import Recipe
from pensimpy.env_setup.peni_env_setup import PenSimEnv
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
    done = False
    observation = env.reset()
    recipe = Recipe()

    time_stamp, batch_yield, yield_pre = 0, 0, 0
    while not done:
        # time is from 1 to 1150
        time_stamp += 1

        # Get action from recipe agent based on time
        Fs, Foil, Fg, Fpres, Fdischarge, Fw, Fpaa = recipe.run(time_stamp)

        # Run and get the reward
        # observation is a class which contains all the variables, e.g. observation.Fs.y[k], observation.Fs.t[k]
        # are the Fs value and corresponding time at k
        observation, reward, done = env.step(time_stamp,
                                             observation,
                                             Fs, Foil, Fg, Fpres, Fdischarge, Fw, Fpaa)

        batch_yield += reward

    print(f"=== cost: {int(time.time() - t)} s")
    print(f"=== batch_yield: {batch_yield}")

    # # check
    # from pensimpy.pensim_classes.Constants import H
    # import numpy as np
    # penicillin_yield_total = (observation.V.y[-1] * observation.P.y[-1]
    #                           - np.dot(observation.Fremoved.y, observation.P.y) * H) / 1000
    # print(f"=== penicillin_yield: {penicillin_yield_total}")

    # # Plot
    # show_params(observation)
