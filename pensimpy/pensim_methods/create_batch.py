import numpy as np
from pensimpy.pensim_classes.X import X
from pensimpy.pensim_classes.Channel import Channel
from pensimpy.pensim_methods.create_channel import create_channel


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
    channel = Channel()
    create_channel(channel, **{'name': 'Aeration rate', 'yUnit': 'L/h', 'tUnit': 'h', 'time': t, 'value': y})
    x.Fg = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Agitator RPM', 'yUnit': 'RPM', 'tUnit': 'h', 'time': t, 'value': y})
    x.RPM = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Sugar feed rate', 'yUnit': 'L/h', 'tUnit': 'h', 'time': t, 'value': y})
    x.Fs = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Substrate feed concen.', 'yUnit': 'g/L', 'tUnit': 'h', 'time': t, 'value': y})
    x.sc = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Acid/base feed concen.', 'yUnit': 'moles', 'tUnit': 'h', 'time': t, 'value': y})
    x.abc = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Acid flow rate', 'yUnit': 'L/h', 'tUnit': 'h', 'time': t, 'value': y})
    x.Fa = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Base flow rate', 'yUnit': 'L/h', 'tUnit': 'h', 'time': t, 'value': y})
    x.Fb = channel

    channel = Channel()
    create_channel(channel,
                   **{'name': 'Heating/cooling water flowrate', 'yUnit': 'L/h', 'tUnit': 'h', 'time': t, 'value': y})
    x.Fc = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Heating water flowrate', 'yUnit': 'L/h', 'tUnit': 'h', 'time': t, 'value': y})
    x.Fh = channel

    # indpensim manipulated variables
    channel = Channel()
    create_channel(channel,
                   **{'name': 'Water for injection/dilution', 'yUnit': 'L/h', 'tUnit': 'h', 'time': t, 'value': y})
    x.Fw = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Air head pressure', 'yUnit': 'bar', 'tUnit': 'h', 'time': t, 'value': y})
    x.pressure = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Dumped broth flow', 'yUnit': 'L/h', 'tUnit': 'h', 'time': t, 'value': y})
    x.Fremoved = channel

    # pensim states
    channel = Channel()
    create_channel(channel, **{'name': 'Substrate concen.', 'yUnit': 'g/L', 'tUnit': 'h', 'time': t, 'value': y})
    x.S = channel

    channel = Channel()
    create_channel(channel,
                   **{'name': 'Dissolved oxygen concen.', 'yUnit': 'mg/L', 'tUnit': 'h', 'time': t, 'value': y})
    x.DO2 = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Biomass concen.', 'yUnit': 'g/L', 'tUnit': 'h', 'time': t, 'value': y})
    x.X = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Penicillin concen.', 'yUnit': 'g/L', 'tUnit': 'h', 'time': t, 'value': y})
    x.P = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Vessel Volume', 'yUnit': 'L', 'tUnit': 'h', 'time': t, 'value': y})
    x.V = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Vessel Weight', 'yUnit': 'Kg', 'tUnit': 'h', 'time': t, 'value': y})
    x.Wt = channel

    channel = Channel()
    create_channel(channel, **{'name': 'pH', 'yUnit': 'pH', 'tUnit': 'h', 'time': t, 'value': y})
    x.pH = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Temperature', 'yUnit': 'K', 'tUnit': 'h', 'time': t, 'value': y})
    x.T = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Generated heat', 'yUnit': 'kJ', 'tUnit': 'h', 'time': t, 'value': y})
    x.Q = channel

    # indpensim states
    channel = Channel()
    create_channel(channel, **{'name': 'type a0 biomass concen.', 'yUnit': 'g/L', 'tUnit': 'h', 'time': t, 'value': y})
    x.a0 = channel

    channel = Channel()
    create_channel(channel, **{'name': 'type a1 biomass concen.', 'yUnit': 'g/L', 'tUnit': 'h', 'time': t, 'value': y})
    x.a1 = channel

    channel = Channel()
    create_channel(channel, **{'name': 'type a3 biomass concen.', 'yUnit': 'g/L', 'tUnit': 'h', 'time': t, 'value': y})
    x.a3 = channel

    channel = Channel()
    create_channel(channel, **{'name': 'type a4 biomass concen.', 'yUnit': 'g/L', 'tUnit': 'h', 'time': t, 'value': y})
    x.a4 = channel

    channel = Channel()
    create_channel(channel, **{'name': 'state n0', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.n0 = channel

    channel = Channel()
    create_channel(channel, **{'name': 'state n1', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.n1 = channel

    channel = Channel()
    create_channel(channel, **{'name': 'state n2', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.n2 = channel

    channel = Channel()
    create_channel(channel, **{'name': 'state n3', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.n3 = channel

    channel = Channel()
    create_channel(channel, **{'name': 'state n4', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.n4 = channel

    channel = Channel()
    create_channel(channel, **{'name': 'state n5', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.n5 = channel

    channel = Channel()
    create_channel(channel, **{'name': 'state n6', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.n6 = channel

    channel = Channel()
    create_channel(channel, **{'name': 'state n7', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.n7 = channel

    channel = Channel()
    create_channel(channel, **{'name': 'state n8', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.n8 = channel

    channel = Channel()
    create_channel(channel, **{'name': 'state n9', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.n9 = channel

    channel = Channel()
    create_channel(channel, **{'name': 'state nm', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.nm = channel

    channel = Channel()
    create_channel(channel, **{'name': 'state phi0', 'yUnit': '-', 'tUnit': 'h', 'time': t, 'value': y})
    x.phi0 = channel

    channel = Channel()
    create_channel(channel, **{'name': 'carbon dioxide percent in off-gas', 'yUnit': '%', 'tUnit': 'h', 'time': t, 'value': y})
    x.CO2outgas = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Cell culture age', 'yUnit': 'h', 'tUnit': 'h', 'time': t, 'value': y})
    x.Culture_age = channel

    channel = Channel()
    create_channel(channel, **{'name': 'PAA flow', 'yUnit': 'PAA flow (L/h)', 'tUnit': 'h', 'time': t, 'value': y})
    x.Fpaa = channel

    channel = Channel()
    create_channel(channel, **{'name': 'PAA concen.', 'yUnit': 'PAA (g L^{-1})', 'tUnit': 'h', 'time': t, 'value': y})
    x.PAA = channel

    channel = Channel()
    create_channel(channel, **{'name': 'PAA concen. offline', 'yUnit': 'PAA (g L^{-1})', 'tUnit': 'h', 'time': t, 'value': y})
    x.PAA_offline = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Oil flow', 'yUnit': 'L/hr', 'tUnit': 'h', 'time': t, 'value': y})
    x.Foil = channel

    channel = Channel()
    create_channel(channel, **{'name': 'NH_3 concen.', 'yUnit': 'NH3 (g L^{-1})', 'tUnit': 'h', 'time': t, 'value': y})
    x.NH3 = channel

    channel = Channel()
    create_channel(channel, **{'name': 'NH_3 concen. off-line', 'yUnit': 'NH3 (g L^{-1})', 'tUnit': 'h', 'time': t, 'value': y})
    x.NH3_offline = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Oxygen in percent in off-gas', 'yUnit': 'O2 (%)', 'tUnit': 'h', 'time': t, 'value': y})
    x.O2 = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Specific growth rate of Penicillin', 'yUnit': 'mu_P (h^{-1})', 'tUnit': 'h', 'time': t, 'value': y})
    x.mup = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Specific growth rate of Biomass', 'yUnit': 'mu_X (h^{-1})', 'tUnit': 'h', 'time': t, 'value': y})
    x.mux = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Offline Penicillin concen.', 'yUnit': 'P(g L^{-1})', 'tUnit': 'h', 'time': t, 'value': y})
    x.P_offline = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Biomass concen. from CER', 'yUnit': 'g min^{-1}', 'tUnit': 'h', 'time': t, 'value': y})
    x.X_CER = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Offline Biomass concen.', 'yUnit': 'X(g L^{-1})', 'tUnit': 'h', 'time': t, 'value': y})
    x.X_offline = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Carbon evolution rate', 'yUnit': 'g/h', 'tUnit': 'h', 'time': t, 'value': y})
    x.CER = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Biomass specific growth rate', 'yUnit': 'hr^{-1}', 'tUnit': 'h', 'time': t, 'value': y})
    x.mu_X_calc = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Penicillin specific growth rate', 'yUnit': 'hr^{-1}', 'tUnit': 'h', 'time': t, 'value': y})
    x.mu_P_calc = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Discharge rate', 'yUnit': 'L hr^{-1}', 'tUnit': 'h', 'time': t, 'value': y})
    x.F_discharge_cal = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Ammonia shots', 'yUnit': 'kgs', 'tUnit': 'h', 'time': t, 'value': y})
    x.NH3_shots = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Oxygen Uptake Rate', 'yUnit': '(g min^{-1})', 'tUnit': 'h', 'time': t, 'value': y})
    x.OUR = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Dissolved CO_2', 'yUnit': '(mg L^{-1})', 'tUnit': 'h', 'time': t, 'value': y})
    x.CO2_d = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Viscosity', 'yUnit': 'centPoise', 'tUnit': 'h', 'time': t, 'value': y})
    x.Viscosity = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Viscosity Offline', 'yUnit': 'centPoise', 'tUnit': 'h', 'time': t, 'value': y})
    x.Viscosity_offline = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Fault reference', 'yUnit': 'Fault ref', 'tUnit': 'h', 'time': t, 'value': y})
    x.Fault_ref = channel

    channel = Channel()
    create_channel(channel, **{'name': '0-Recipe driven, 1-Operator controlled', 'yUnit': 'Control ref', 'tUnit': 'Batch number', 'time': t, 'value': y})
    x.Control_ref = channel

    channel = Channel()
    create_channel(channel, **{'name': '1-No Raman spec, 1-Raman spec recorded, 2-PAT control', 'yUnit': 'PAT ref', 'tUnit': 'Batch number', 'time': t, 'value': y})
    x.PAT_ref = channel

    channel = Channel()
    create_channel(channel, **{'name': 'Batch reference', 'yUnit': 'Batch ref', 'tUnit': 'Batch ref', 'time': t, 'value': y})
    x.Batch_ref = channel

    channel = Channel()
    create_channel(channel, **{'name': 'PAA Prediction.', 'yUnit': 'PAA_pred (g L^{-1})', 'tUnit': 'h', 'time': t, 'value': y})
    x.PAA_pred = channel

    # Raman Spectra: Wavenumber & Intensity
    Wavenumber = np.zeros((2200, 1), dtype=float)
    Intensity = np.zeros((int(T / h), 2200), dtype=float)
    channel = Channel()
    create_channel(channel, **{'name': 'Raman Spectra', 'yUnit': 'a.u', 'tUnit': 'cm^-1', 'Wavenumber': Wavenumber, 'Intensity': Intensity})
    x.Raman_Spec = channel

    return x
