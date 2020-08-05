import numpy as np
import itertools
import pandas as pd
import math
from scipy.signal import lfilter
from scipy.signal import savgol_filter
from pensimpy.data.x import X
from pensimpy.data.channel import Channel


def create_batch(h, T):
    """
    Create the batch data
    :param h:
    :param T:
    :return:
    """
    t = np.zeros((int(T / h), 1), dtype=float)
    y = np.zeros((int(T / h), 1), dtype=float)

    x = X()
    # pensim manipulated variables
    x.Fg = Channel(**{'name': 'Aeration rate', 'yUnit': 'L/h', 'tUnit': 'h', 'time': t, 'value': y})
    x.RPM = Channel(**{'name': 'Agitator RPM', 'yUnit': 'RPM', 'tUnit': 'h', 'time': t, 'value': y})
    x.Fs = Channel(**{'name': 'Sugar feed rate', 'yUnit': 'L/h', 'tUnit': 'h', 'time': t, 'value': y})
    x.sc = Channel(**{'name': 'Substrate feed concen.', 'yUnit': 'g/L', 'tUnit': 'h', 'time': t, 'value': y})
    x.abc = Channel(**{'name': 'Acid/base feed concen.', 'yUnit': 'moles', 'tUnit': 'h', 'time': t, 'value': y})
    x.Fa = Channel(**{'name': 'Acid flow rate', 'yUnit': 'L/h', 'tUnit': 'h', 'time': t, 'value': y})
    x.Fb = Channel(**{'name': 'Base flow rate', 'yUnit': 'L/h', 'tUnit': 'h', 'time': t, 'value': y})
    x.Fc = Channel(**{'name': 'Heating/cooling water flowrate', 'yUnit': 'L/h', 'tUnit': 'h', 'time': t, 'value': y})
    x.Fh = Channel(**{'name': 'Heating water flowrate', 'yUnit': 'L/h', 'tUnit': 'h', 'time': t, 'value': y})

    # indpensim manipulated variables
    x.Fw = Channel(**{'name': 'Water for injection/dilution', 'yUnit': 'L/h', 'tUnit': 'h', 'time': t, 'value': y})
    x.pressure = Channel(**{'name': 'Air head pressure', 'yUnit': 'bar', 'tUnit': 'h', 'time': t, 'value': y})
    x.Fremoved = Channel(**{'name': 'Dumped broth flow', 'yUnit': 'L/h', 'tUnit': 'h', 'time': t, 'value': y})

    # pensim states
    x.S = Channel(**{'name': 'Substrate concen.', 'yUnit': 'g/L', 'tUnit': 'h', 'time': t, 'value': y})
    x.DO2 = Channel(**{'name': 'Dissolved oxygen concen.', 'yUnit': 'mg/L', 'tUnit': 'h', 'time': t, 'value': y})
    x.X = Channel(**{'name': 'Biomass concen.', 'yUnit': 'g/L', 'tUnit': 'h', 'time': t, 'value': y})
    x.P = Channel(**{'name': 'Penicillin concen.', 'yUnit': 'g/L', 'tUnit': 'h', 'time': t, 'value': y})
    x.V = Channel(**{'name': 'Vessel Volume', 'yUnit': 'L', 'tUnit': 'h', 'time': t, 'value': y})
    x.Wt = Channel(**{'name': 'Vessel Weight', 'yUnit': 'Kg', 'tUnit': 'h', 'time': t, 'value': y})
    x.pH = Channel(**{'name': 'pH', 'yUnit': 'pH', 'tUnit': 'h', 'time': t, 'value': y})
    x.T = Channel(**{'name': 'Temperature', 'yUnit': 'K', 'tUnit': 'h', 'time': t, 'value': y})
    x.Q = Channel(**{'name': 'Generated heat', 'yUnit': 'kJ', 'tUnit': 'h', 'time': t, 'value': y})

    # indpensim states
    x.a0 = Channel(**{'name': 'type a0 biomass concen.', 'yUnit': 'g/L', 'tUnit': 'h', 'time': t, 'value': y})
    x.a1 = Channel(**{'name': 'type a1 biomass concen.', 'yUnit': 'g/L', 'tUnit': 'h', 'time': t, 'value': y})
    x.a3 = Channel(**{'name': 'type a3 biomass concen.', 'yUnit': 'g/L', 'tUnit': 'h', 'time': t, 'value': y})
    x.a4 = Channel(**{'name': 'type a4 biomass concen.', 'yUnit': 'g/L', 'tUnit': 'h', 'time': t, 'value': y})
    x.n0 = Channel(**{'name': 'state n0', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.n1 = Channel(**{'name': 'state n1', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.n2 = Channel(**{'name': 'state n2', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.n3 = Channel(**{'name': 'state n3', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.n4 = Channel(**{'name': 'state n4', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.n5 = Channel(**{'name': 'state n5', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.n6 = Channel(**{'name': 'state n6', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.n7 = Channel(**{'name': 'state n7', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.n8 = Channel(**{'name': 'state n8', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.n9 = Channel(**{'name': 'state n9', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.nm = Channel(**{'name': 'state nm', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.phi0 = Channel(**{'name': 'state phi0', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.CO2outgas = Channel(**{'name': 'CO2 percent in off-gas', 'yUnit': '%', 'tUnit': 'h', 'time': t, 'value': y})
    x.Culture_age = Channel(**{'name': 'Cell culture age', 'yUnit': 'h', 'tUnit': 'h', 'time': t, 'value': y})
    x.Fpaa = Channel(**{'name': 'PAA flow', 'yUnit': 'PAA flow (L/h)', 'tUnit': 'h', 'time': t, 'value': y})
    x.PAA = Channel(**{'name': 'PAA concen.', 'yUnit': 'PAA (g L^{-1})', 'tUnit': 'h', 'time': t, 'value': y})
    x.PAA_offline = Channel(
        **{'name': 'PAA concen. offline', 'yUnit': 'PAA (g L^{-1})', 'tUnit': 'h', 'time': t, 'value': y})
    x.Foil = Channel(**{'name': 'Oil flow', 'yUnit': 'L/hr', 'tUnit': 'h', 'time': t, 'value': y})
    x.NH3 = Channel(**{'name': 'NH_3 concen.', 'yUnit': 'NH3 (g L^{-1})', 'tUnit': 'h', 'time': t, 'value': y})
    x.NH3_offline = Channel(
        **{'name': 'NH_3 concen. off-line', 'yUnit': 'NH3 (g L^{-1})', 'tUnit': 'h', 'time': t, 'value': y})
    x.O2 = Channel(**{'name': 'Oxygen in percent in off-gas', 'yUnit': 'O2 (%)', 'tUnit': 'h', 'time': t, 'value': y})
    x.mup = Channel(
        **{'name': 'Specific growth rate of Penicillin', 'yUnit': 'mu_P (h^{-1})', 'tUnit': 'h', 'time': t, 'value': y})
    x.mux = Channel(
        **{'name': 'Specific growth rate of Biomass', 'yUnit': 'mu_X (h^{-1})', 'tUnit': 'h', 'time': t, 'value': y})
    x.P_offline = Channel(
        **{'name': 'Offline Penicillin concen.', 'yUnit': 'P(g L^{-1})', 'tUnit': 'h', 'time': t, 'value': y})
    x.X_CER = Channel(
        **{'name': 'Biomass concen. from CER', 'yUnit': 'g min^{-1}', 'tUnit': 'h', 'time': t, 'value': y})
    x.X_offline = Channel(
        **{'name': 'Offline Biomass concen.', 'yUnit': 'X(g L^{-1})', 'tUnit': 'h', 'time': t, 'value': y})
    x.CER = Channel(**{'name': 'Carbon evolution rate', 'yUnit': 'g/h', 'tUnit': 'h', 'time': t, 'value': y})
    x.mu_X_calc = Channel(
        **{'name': 'Biomass specific growth rate', 'yUnit': 'hr^{-1}', 'tUnit': 'h', 'time': t, 'value': y})
    x.mu_P_calc = Channel(
        **{'name': 'Penicillin specific growth rate', 'yUnit': 'hr^{-1}', 'tUnit': 'h', 'time': t, 'value': y})
    x.F_discharge_cal = Channel(**{'name': 'Discharge rate', 'yUnit': 'L hr^{-1}', 'tUnit': 'h', 'time': t, 'value': y})
    x.NH3_shots = Channel(**{'name': 'Ammonia shots', 'yUnit': 'kgs', 'tUnit': 'h', 'time': t, 'value': y})
    x.OUR = Channel(**{'name': 'Oxygen Uptake Rate', 'yUnit': '(g min^{-1})', 'tUnit': 'h', 'time': t, 'value': y})
    x.CO2_d = Channel(**{'name': 'Dissolved CO_2', 'yUnit': '(mg L^{-1})', 'tUnit': 'h', 'time': t, 'value': y})
    x.Viscosity = Channel(**{'name': 'Viscosity', 'yUnit': 'centPoise', 'tUnit': 'h', 'time': t, 'value': y})
    x.Viscosity_offline = Channel(
        **{'name': 'Viscosity Offline', 'yUnit': 'centPoise', 'tUnit': 'h', 'time': t, 'value': y})
    x.Fault_ref = Channel(**{'name': 'Fault reference', 'yUnit': 'Fault ref', 'tUnit': 'h', 'time': t, 'value': y})
    x.Control_ref = Channel(
        **{'name': '0-Recipe driven, 1-Operator controlled', 'yUnit': 'Control ref', 'tUnit': 'Batch number', 'time': t,
           'value': y})
    x.PAT_ref = Channel(
        **{'name': '1-No Raman spec, 1-Raman spec recorded, 2-PAT control', 'yUnit': 'PAT ref', 'tUnit': 'Batch number',
           'time': t, 'value': y})
    x.Batch_ref = Channel(
        **{'name': 'Batch reference', 'yUnit': 'Batch ref', 'tUnit': 'Batch ref', 'time': t, 'value': y})
    x.PAA_pred = Channel(
        **{'name': 'PAA Prediction.', 'yUnit': 'PAA_pred (g L^{-1})', 'tUnit': 'h', 'time': t, 'value': y})

    # Raman Spectra: Wavenumber & Intensity
    Wavenumber = np.zeros((2200, 1), dtype=float)
    Intensity = np.zeros((int(T / h), 2200), dtype=float)
    x.Raman_Spec = Channel(
        **{'name': 'Raman Spectra', 'yUnit': 'a.u', 'tUnit': 'cm^-1', 'Wavenumber': Wavenumber, 'Intensity': Intensity})

    return x


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


def predict_substrate(k, x, model_data):
    Raman_Spec_sg = savgol_filter(x.Raman_Spec.Intensity[k-2, :], 5, 2)
    Raman_Spec_sg_d = np.diff(Raman_Spec_sg)
    PAA_peaks_Spec = []
    PAA_peaks_Spec.extend(Raman_Spec_sg_d[349:500])
    PAA_peaks_Spec.extend(Raman_Spec_sg_d[799:860])
    x.PAA_pred.y[k - 2] = np.dot(np.array([PAA_peaks_Spec]), np.array([model_data]).T)[0]
    if k > 21:
        x.PAA_pred.y[k - 2] = (x.PAA_pred.y[k - 3] + x.PAA_pred.y[k - 4] + x.PAA_pred.y[k - 2]) / 3
    return x


def get_recipe_trend(recipe_list):
    """
    Get the recipe trend data
    :param recipe:
    :param recipe_sp:
    :return:
    """
    recipe = [int(ele / 0.2) for ele in recipe_list[0]]
    recipe = [recipe[0]] + np.diff(recipe).tolist()
    recipe_sp = [[ele] for ele in recipe_list[1]]
    res_default = [x * y for x, y in zip(recipe, recipe_sp)]
    return list(itertools.chain(*res_default))[0:1150]


def get_dataframe(batch_data, include_raman):
    df = pd.DataFrame(data={"Volume": batch_data.V.y,
                            "Penicillin Concentration": batch_data.P.y,
                            "Discharge rate": batch_data.Fremoved.y,
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
    df = df.set_index([[t * 0.2 for t in range(1, 1151)]])

    df_raman = pd.DataFrame()
    if include_raman:
        wavenumber = batch_data.Raman_Spec.Wavenumber
        df_raman = pd.DataFrame(batch_data.Raman_Spec.Intensity, columns=wavenumber)
        df_raman = df_raman[df_raman.columns[::-1]]
        df_raman = df_raman.set_index([[t * 0.2 for t in range(1, 1151)]])
        return df, df_raman

    return df, df_raman


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