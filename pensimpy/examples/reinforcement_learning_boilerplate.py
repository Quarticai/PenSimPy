import time
import random
import numpy as np
from pensimpy.data.recipe import Recipe
from pensimpy.peni_env_setup import PenSimEnv
from pensimpy.constants import STEP_IN_MINUTES


class Agent:
    """
    This is a simple agent samples random actions.
    """

    def __init__(self, act_dim):
        self.act_dim = act_dim

    def sample_actions(self):
        return np.clip([random.uniform(-1.5, 1.5) for _ in range(self.act_dim)], -0.01, 0.01)


def run(episodes=1000):
    """
    This is a boilerplate to simulate penicillin yield with reinforcement learning. The random agent can be replaced by a self-defined agent.
    :param episodes: Number of episodes to learn, the default number is 1000.
    :return: A list of penicillin batch yield.
    """
    agent = Agent(act_dim=7)

    batch_yield_list = []
    t = time.time()

    for e in range(episodes):
        recipe = Recipe.get_default()
        env = PenSimEnv(recipe=recipe)
        done = False
        observation, batch_data = env.reset()
        k_timestep, batch_yield, yield_pre = 0, 0, 0

        while not done:
            k_timestep += 1

            actions = agent.sample_actions()

            """add adjustment to each action"""
            Fs_a, Foil_a, Fg_a, pres_a, discharge_a, Fw_a, Fpaa_a = actions

            """Get action from recipe agent based on k_timestep"""
            Fs, Foil, Fg, pressure, discharge, Fw, Fpaa = recipe.get_values_at(k_timestep * STEP_IN_MINUTES)

            """update recipe actions with agent actions"""
            Fs *= (1 + Fs_a)
            Foil *= (1 + Foil_a)
            Fg *= (1 + Fg_a)
            pressure *= (1 + pres_a)
            discharge *= (1 + discharge_a)
            Fw *= (1 + Fw_a)
            Fpaa *= (1 + Fpaa_a)

            observation, batch_data, reward, done = env.step(k_timestep,
                                                             batch_data,
                                                             Fs, Foil, Fg, pressure, discharge, Fw, Fpaa)
            batch_yield += reward
        print(f"episode: {e}, elapsed time: {int(time.time() - t)} s, batch_yield: {batch_yield}")
        batch_yield_list.append(batch_yield)
    return batch_yield_list
