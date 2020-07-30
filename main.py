from pensimpy.env_setup.peni_env_setup import PenSimEnv

if __name__ == "__main__":
    # Provide recipes for Fs, Foil, Fg, pres, discharge, water
    # Recipe time range: 0 < t <= 230
    setpoints_dict = {'water': [(100, 11)]}

    env = PenSimEnv()
    df, df_raman = env.get_batches(random_seed=1, setpoints=setpoints_dict, include_raman=False)
