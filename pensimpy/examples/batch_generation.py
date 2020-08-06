from pensimpy.peni_env_setup import PenSimEnv
from pensimpy.data.recipe import Recipe, Setpoint


def run():
    """
    Basic batch generation example which simulates the Sequential Batch Control.
    :return: batch data and Raman spectra in pandas dataframe
    """
    recipe = Recipe.get_default()
    # User can either add new setpoints (such as Fs, Foil, Fg, pres, discharge, water) or modify the default by giving
    # a `time_until` in Setpoint(time_until, value) class
    # e.g. add a Setpoint with value of 1000 at the 3rd hour for `Fs`
    # --- recipe.update_process_variable_setpoints('Fs', Setpoint(3, 1000))
    env = PenSimEnv(recipe=recipe)

    return env.get_batches(random_seed=1, include_raman=False)
