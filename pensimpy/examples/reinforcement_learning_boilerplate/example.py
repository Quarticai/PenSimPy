import time
from pensimpy.data.recipe import Recipe
from pensimpy.peni_env_setup import PenSimEnv
from .agent import Agent


def run(episodes=1000):
    """
    This is a boilerplate to simulate penicillin yield with reinforcement learning. The random agent can be replaced by a self-defined agent.
    :param episodes: Number of episodes to learn, the default number is 1000.
    :return: A list of penicillin batch yield.
    """
    agent = Agent(act_dim=7)

    batch_yield_list = []
    t = time.time()

    MINS_PER_TIME_STEP = 12  # minutes

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
            Fs_a, Foil_a, Fg_a, pres_a, Fremoved_a, Fw_a, Fpaa_a = actions

            """Get action from recipe agent based on k_timestep"""
            Fs, Foil, Fg, pressure, Fremoved, Fw, Fpaa = recipe.get_values_at(k_timestep * MINS_PER_TIME_STEP)

            """update recipe actions with agent actions"""
            Fs *= (1 + Fs_a)
            Foil *= (1 + Foil_a)
            Fg *= (1 + Fg_a)
            pressure *= (1 + pres_a)
            Fremoved *= (1 + Fremoved_a)
            Fw *= (1 + Fw_a)
            Fpaa *= (1 + Fpaa_a)

            observation, batch_data, reward, done = env.step(k_timestep,
                                                             batch_data,
                                                             Fs, Foil, Fg, pressure, Fremoved, Fw, Fpaa)
            batch_yield += reward
        print(f"episode: {e}, elapsed time: {int(time.time() - t)} s, batch_yield: {batch_yield}")
        batch_yield_list.append(batch_yield)
    return batch_yield_list
