import numpy as np
import pandas as pd
import math
from scipy.signal import lfilter
from pensimpy.constants import NUM_STEPS, STEP_IN_HOURS


def pid_controller(uk1, ek, ek1, yk, yk1, yk2, u_min, u_max, Kp, Ti, Td, h):
    """
    PID controller
    :param uk1:
    :param ek:
    :param ek1:
    :param yk:
    :param yk1:
    :param yk2:
    :param u_min:
    :param u_max:
    :param Kp:
    :param Ti:
    :param Td:
    :param h:
    :return:
    """
    # proportional component
    P = ek - ek1
    # checks if the integral time constant is defined
    I = ek * h / Ti if Ti > 1e-7 else 0
    # derivative component
    D = -Td / h * (yk - 2 * yk1 + yk2) if Td > 0.001 else 0
    # computes and saturates the control signal
    uu = uk1 + Kp * (P + I + D)
    uu = u_max if uu > u_max else uu
    uu = u_min if uu < u_min else uu

    return uu


def smooth(y, width):
    """
    Realize Matlab smooth() func.
    :param y: list
    :param width:
    :return: list
    """
    n = len(y)
    b1 = np.ones(width) / width
    c = lfilter(b1, [1], y, axis=0)
    cbegin = np.cumsum(y[0:width - 2])
    cbegin = cbegin[::2] / np.arange(1, width - 1, 2)
    cend = np.cumsum(y[n - width + 2:n][::-1])
    cend = cend[::-2] / np.arange(1, width - 1)[::-2]
    c_new = []
    c_new.extend(cbegin)
    c_new.extend(c[width - 1:].tolist())
    c_new.extend(cend)
    return c_new


def get_dataframe(batch_data, include_raman):
    """
    Construct pandas dataframes from batch features
    """
    df = pd.DataFrame(data={"Volume": batch_data.V.y,
                            "Penicillin Concentration": batch_data.P.y,
                            "Discharge rate": batch_data.discharge.y,
                            "Sugar feed rate": batch_data.Fs.y,
                            "Soil bean feed rate": batch_data.Foil.y,
                            "Aeration rate": batch_data.Fg.y,
                            "Back pressure": batch_data.pressure.y,
                            "Water injection/dilution": batch_data.Fw.y,
                            "Phenylacetic acid flow-rate": batch_data.Fpaa.y,
                            "pH": batch_data.pH.y,
                            "Temperature": batch_data.T.y,
                            "Acid flow rate": batch_data.Fa.y,
                            "Base flow rate": batch_data.Fb.y,
                            "Cooling water": batch_data.Fc.y,
                            "Heating water": batch_data.Fh.y,
                            "Vessel Weight": batch_data.Wt.y,
                            "Dissolved oxygen concentration": batch_data.DO2.y,
                            "Oxygen in percent in off-gas": batch_data.O2.y, })
    df = df.set_index([[t * STEP_IN_HOURS for t in range(1, NUM_STEPS + 1)]])

    df_raman = pd.DataFrame()
    if include_raman:
        wavenumber = batch_data.Raman_Spec.Wavenumber
        df_raman = pd.DataFrame(batch_data.Raman_Spec.Intensity, columns=wavenumber)
        df_raman = df_raman[df_raman.columns[::-1]]
        df_raman = df_raman.set_index([[t * STEP_IN_HOURS for t in range(1, NUM_STEPS + 1)]])
        return df, df_raman

    return df, df_raman


def get_observation_data(observation, t):
    """
    Get observation data at t.
    """
    vars = ['Foil', 'Fw', 'Fs', 'Fa', 'Fb', 'Fc', 'Fh', 'Fg', 'Wt', 'discharge', 'DO2', 'T', 'O2', 'pressure']
    # convert to pH from H+ concentration
    pH = observation.pH.y[t]
    pH = -math.log(pH) / math.log(10) if pH != 0 else pH
    return [[var, eval(f"observation.{var}.y[t]", {'observation': observation, 't': t})] for var in vars] + [['pH', pH]]