import math


def get_observation_data(observation, t):
    """
    Get observation data at t.
    """
    vars = ['Foil', 'Fw', 'Fs', 'Fa', 'Fb', 'Fc', 'Fh', 'Fg', 'Wt', 'Fremoved', 'DO2', 'T',
            'O2', 'pressure']

    # convert to pH from H+ concentration
    pH = observation.pH.y[t]
    pH = -math.log(pH) / math.log(10) if pH != 0 else pH
    return [[var, eval(f"observation.{var}.y[t]", {'observation': observation, 't': t})] for var in vars] + \
           [['pH', pH]]
