from pensimpy.env_setup.peni_env_setup import PenSimEnv

if __name__ == "__main__":
    # Provide recipes for Fs, Foil, Fg, pres, discharge, water
    # Recipe time range: 0 < t <= 230
    setpoints_dict = {'water': [(100, 121)]}

    env = PenSimEnv()
    env.get_batches(random_seed=1, setpoints=setpoints_dict, num_batches=3, include_raman=False, plot_batch=False)
