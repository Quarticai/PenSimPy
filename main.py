from pensimpy.env_setup.peni_env_setup import PenSimEnv

if __name__ == "__main__":
    # Provide recipes for Fs, Foil, Fg, pres, discharge, water
    # Recipe time range: 0 < t <= 230
    setpoints_dict = {'water': [(100, 6100)]}

    random_seed = 0
    env = PenSimEnv(setpoints=setpoints_dict, random_seed=2)
    env.get_batches()
