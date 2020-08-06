from pensimpy.peni_env_setup import PenSimEnv
from pensimpy.data.recipe import Recipe

if __name__ == "__main__":
    # Provide recipes for Fs, Foil, Fg, pres, discharge, water
    # Recipe time range: 0 < t <= 230
    recipe = Recipe.get_default()
    env = PenSimEnv(recipe=recipe)
    df, df_raman = env.get_batches(random_seed=1, include_raman=False)
