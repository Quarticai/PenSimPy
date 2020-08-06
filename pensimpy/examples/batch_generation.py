from pensimpy.peni_env_setup import PenSimEnv
from pensimpy.data.recipe import Recipe


def run():
    """
    Basic batch generation example which simulates the Sequential Batch Control.
    :return: batch data and Raman spectra in pandas dataframe
    """
    # User can manually provide recipes (Fs, Foil, Fg, pres, discharge, water) instead of getting the default ones.

    recipe = Recipe.get_default()
    env = PenSimEnv(recipe=recipe)

    return env.get_batches(random_seed=1, include_raman=False)
