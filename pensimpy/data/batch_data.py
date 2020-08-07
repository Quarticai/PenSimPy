import numpy as np
from pensimpy.data.channel import Channel
from pensimpy.constants import NUM_STEPS, WAVENUMBER_LENGTH
from scipy.signal import lfilter


class X:
    """
    Batch data. All features included.
    """

    def __init__(self):
        t = np.zeros((NUM_STEPS, 1), dtype=float)
        y = np.zeros((NUM_STEPS, 1), dtype=float)
        self.Fg = Channel(**{'name': 'Aeration rate', 'y_unit': 'L/h', 't_unit': 'h', 'time': t, 'value': y})
        self.RPM = Channel(**{'name': 'Agitator RPM', 'y_unit': 'RPM', 't_unit': 'h', 'time': t, 'value': y})
        self.Fs = Channel(**{'name': 'Sugar feed rate', 'y_unit': 'L/h', 't_unit': 'h', 'time': t, 'value': y})
        self.sc = Channel(**{'name': 'Substrate feed concen.', 'y_unit': 'g/L', 't_unit': 'h', 'time': t, 'value': y})
        self.abc = Channel(**{'name': 'Acid/base feed concen.', 'y_unit': 'moles', 't_unit': 'h', 'time': t, 'value': y})
        self.Fa = Channel(**{'name': 'Acid flow rate', 'y_unit': 'L/h', 't_unit': 'h', 'time': t, 'value': y})
        self.Fb = Channel(**{'name': 'Base flow rate', 'y_unit': 'L/h', 't_unit': 'h', 'time': t, 'value': y})
        self.Fc = Channel(**{'name': 'Heating/cooling water flowrate', 'y_unit': 'L/h', 't_unit': 'h', 'time': t, 'value': y})
        self.Fh = Channel(**{'name': 'Heating water flowrate', 'y_unit': 'L/h', 't_unit': 'h', 'time': t, 'value': y})
        self.Fw = Channel(**{'name': 'Water for injection/dilution', 'y_unit': 'L/h', 't_unit': 'h', 'time': t, 'value': y})
        self.pressure = Channel(**{'name': 'Air head pressure', 'y_unit': 'bar', 't_unit': 'h', 'time': t, 'value': y})
        self.discharge = Channel(**{'name': 'Dumped broth flow', 'y_unit': 'L/h', 't_unit': 'h', 'time': t, 'value': y})
        self.S = Channel(**{'name': 'Substrate concen.', 'y_unit': 'g/L', 't_unit': 'h', 'time': t, 'value': y})
        self.DO2 = Channel(**{'name': 'Dissolved oxygen concen.', 'y_unit': 'mg/L', 't_unit': 'h', 'time': t, 'value': y})
        self.X = Channel(**{'name': 'Biomass concen.', 'y_unit': 'g/L', 't_unit': 'h', 'time': t, 'value': y})
        self.P = Channel(**{'name': 'Penicillin concen.', 'y_unit': 'g/L', 't_unit': 'h', 'time': t, 'value': y})
        self.V = Channel(**{'name': 'Vessel Volume', 'y_unit': 'L', 't_unit': 'h', 'time': t, 'value': y})
        self.Wt = Channel(**{'name': 'Vessel Weight', 'y_unit': 'Kg', 't_unit': 'h', 'time': t, 'value': y})
        self.pH = Channel(**{'name': 'pH', 'y_unit': 'pH', 't_unit': 'h', 'time': t, 'value': y})
        self.T = Channel(**{'name': 'Temperature', 'y_unit': 'K', 't_unit': 'h', 'time': t, 'value': y})
        self.Q = Channel(**{'name': 'Generated heat', 'y_unit': 'kJ', 't_unit': 'h', 'time': t, 'value': y})
        self.a0 = Channel(**{'name': 'type a0 biomass concen.', 'y_unit': 'g/L', 't_unit': 'h', 'time': t, 'value': y})
        self.a1 = Channel(**{'name': 'type a1 biomass concen.', 'y_unit': 'g/L', 't_unit': 'h', 'time': t, 'value': y})
        self.a3 = Channel(**{'name': 'type a3 biomass concen.', 'y_unit': 'g/L', 't_unit': 'h', 'time': t, 'value': y})
        self.a4 = Channel(**{'name': 'type a4 biomass concen.', 'y_unit': 'g/L', 't_unit': 'h', 'time': t, 'value': y})
        self.n0 = Channel(**{'name': 'state n0', 'y_unit': '-', 't_unit': 'h', 'time': t, 'value': y})
        self.n1 = Channel(**{'name': 'state n1', 'y_unit': '-', 't_unit': 'h', 'time': t, 'value': y})
        self.n2 = Channel(**{'name': 'state n2', 'y_unit': '-', 't_unit': 'h', 'time': t, 'value': y})
        self.n3 = Channel(**{'name': 'state n3', 'y_unit': '-', 't_unit': 'h', 'time': t, 'value': y})
        self.n4 = Channel(**{'name': 'state n4', 'y_unit': '-', 't_unit': 'h', 'time': t, 'value': y})
        self.n5 = Channel(**{'name': 'state n5', 'y_unit': '-', 't_unit': 'h', 'time': t, 'value': y})
        self.n6 = Channel(**{'name': 'state n6', 'y_unit': '-', 't_unit': 'h', 'time': t, 'value': y})
        self.n7 = Channel(**{'name': 'state n7', 'y_unit': '-', 't_unit': 'h', 'time': t, 'value': y})
        self.n8 = Channel(**{'name': 'state n8', 'y_unit': '-', 't_unit': 'h', 'time': t, 'value': y})
        self.n9 = Channel(**{'name': 'state n9', 'y_unit': '-', 't_unit': 'h', 'time': t, 'value': y})
        self.nm = Channel(**{'name': 'state nm', 'y_unit': '-', 't_unit': 'h', 'time': t, 'value': y})
        self.phi0 = Channel(**{'name': 'state phi0', 'y_unit': '-', 't_unit': 'h', 'time': t, 'value': y})
        self.CO2outgas = Channel(**{'name': 'CO2 percent in off-gas', 'y_unit': '%', 't_unit': 'h', 'time': t, 'value': y})
        self.Culture_age = Channel(**{'name': 'Cell culture age', 'y_unit': 'h', 't_unit': 'h', 'time': t, 'value': y})
        self.Fpaa = Channel(**{'name': 'PAA flow', 'y_unit': 'PAA flow (L/h)', 't_unit': 'h', 'time': t, 'value': y})
        self.PAA = Channel(**{'name': 'PAA concen.', 'y_unit': 'PAA (g L^{-1})', 't_unit': 'h', 'time': t, 'value': y})
        self.PAA_offline = Channel(**{'name': 'PAA concen. offline', 'y_unit': 'PAA (g L^{-1})', 't_unit': 'h', 'time': t, 'value': y})
        self.Foil = Channel(**{'name': 'Oil flow', 'y_unit': 'L/hr', 't_unit': 'h', 'time': t, 'value': y})
        self.NH3 = Channel(**{'name': 'NH_3 concen.', 'y_unit': 'NH3 (g L^{-1})', 't_unit': 'h', 'time': t, 'value': y})
        self.NH3_offline = Channel(**{'name': 'NH_3 concen. off-line', 'y_unit': 'NH3 (g L^{-1})', 't_unit': 'h', 'time': t, 'value': y})
        self.OUR = Channel(**{'name': 'Oxygen Uptake Rate', 'y_unit': '(g min^{-1})', 't_unit': 'h', 'time': t, 'value': y})
        self.O2 = Channel(**{'name': 'Oxygen in percent in off-gas', 'y_unit': 'O2 (%)', 't_unit': 'h', 'time': t, 'value': y})
        self.mup = Channel(**{'name': 'Specific growth rate of Penicillin', 'y_unit': 'mu_P (h^{-1})', 't_unit': 'h', 'time': t, 'value': y})
        self.mux = Channel(**{'name': 'Specific growth rate of Biomass', 'y_unit': 'mu_X (h^{-1})', 't_unit': 'h', 'time': t, 'value': y})
        self.P_offline = Channel(**{'name': 'Offline Penicillin concen.', 'y_unit': 'P(g L^{-1})', 't_unit': 'h', 'time': t, 'value': y})
        self.X_CER = Channel(**{'name': 'Biomass concen. from CER', 'y_unit': 'g min^{-1}', 't_unit': 'h', 'time': t, 'value': y})
        self.X_offline = Channel(**{'name': 'Offline Biomass concen.', 'y_unit': 'X(g L^{-1})', 't_unit': 'h', 'time': t, 'value': y})
        self.CER = Channel(**{'name': 'Carbon evolution rate', 'y_unit': 'g/h', 't_unit': 'h', 'time': t, 'value': y})
        self.mu_X_calc = Channel(**{'name': 'Biomass specific growth rate', 'y_unit': 'hr^{-1}', 't_unit': 'h', 'time': t, 'value': y})
        self.mu_P_calc = Channel(**{'name': 'Penicillin specific growth rate', 'y_unit': 'hr^{-1}', 't_unit': 'h', 'time': t, 'value': y})
        self.F_discharge_cal = Channel(**{'name': 'Discharge rate', 'y_unit': 'L hr^{-1}', 't_unit': 'h', 'time': t, 'value': y})
        self.NH3_shots = Channel(**{'name': 'Ammonia shots', 'y_unit': 'kgs', 't_unit': 'h', 'time': t, 'value': y})
        self.CO2_d = Channel(**{'name': 'Dissolved CO_2', 'y_unit': '(mg L^{-1})', 't_unit': 'h', 'time': t, 'value': y})
        self.Viscosity = Channel(**{'name': 'Viscosity', 'y_unit': 'centPoise', 't_unit': 'h', 'time': t, 'value': y})
        self.Viscosity_offline = Channel(**{'name': 'Viscosity Offline', 'y_unit': 'centPoise', 't_unit': 'h', 'time': t, 'value': y})
        self.Fault_ref = Channel(**{'name': 'Fault reference', 'y_unit': 'Fault ref', 't_unit': 'h', 'time': t, 'value': y})
        self.Control_ref = Channel(**{'name': '0-Recipe driven, 1-Operator controlled', 'y_unit': 'Control ref', 't_unit': 'Batch number', 'time': t, 'value': y})
        self.PAT_ref = Channel(**{'name': '1-No Raman spec, 1-Raman spec recorded, 2-PAT control', 'y_unit': 'PAT ref', 't_unit': 'Batch number', 'time': t, 'value': y})
        self.Batch_ref = Channel(**{'name': 'Batch reference', 'y_unit': 'Batch ref', 't_unit': 'Batch ref', 'time': t, 'value': y})
        self.PAA_pred = Channel(**{'name': 'PAA Prediction.', 'y_unit': 'PAA_pred (g L^{-1})', 't_unit': 'h', 'time': t, 'value': y})
        # extra
        self.PRBS_noise_addition = [0] * NUM_STEPS
        # Raman Spectra: Wavenumber & Intensity
        Wavenumber = np.zeros((WAVENUMBER_LENGTH, 1), dtype=float)
        Intensity = np.zeros((NUM_STEPS, WAVENUMBER_LENGTH), dtype=float)
        self.Raman_Spec = Channel(**{'name': 'Raman Spectra', 'y_unit': 'a.u', 't_unit': 'cm^-1', 'Wavenumber': Wavenumber, 'Intensity': Intensity})


class X0:
    """
    Initialize key features of the batch data.
    """

    def __init__(self, random_seed_ref, intial_conds):
        random_state = np.random.RandomState(random_seed_ref)
        self.mux = 0.41 + 0.025 * random_state.randn(1)[0]

        random_seed_ref += 1
        random_state = np.random.RandomState(random_seed_ref)
        self.mup = 0.041 + 0.0025 * random_state.randn(1)[0]

        random_seed_ref += 1
        random_state = np.random.RandomState(random_seed_ref)
        self.S = 1 + 0.1 * random_state.randn(1)[0]

        random_seed_ref += 1
        random_state = np.random.RandomState(random_seed_ref)
        self.DO2 = 15 + 0.5 * random_state.randn(1)[0]

        random_seed_ref += 1
        random_state = np.random.RandomState(random_seed_ref)
        self.X = intial_conds + 0.1 * random_state.randn(1)[0]
        self.P = 0

        random_seed_ref += 1
        random_state = np.random.RandomState(random_seed_ref)
        self.V = 5.800e+04 + 500 * random_state.randn(1)[0]

        random_seed_ref += 1
        random_state = np.random.RandomState(random_seed_ref)
        self.Wt = 6.2e+04 + 500 * random_state.randn(1)[0]

        random_seed_ref += 1
        random_state = np.random.RandomState(random_seed_ref)
        self.CO2outgas = 0.038 + 0.001 * random_state.randn(1)[0]

        random_seed_ref += 1
        random_state = np.random.RandomState(random_seed_ref)
        self.O2 = 0.20 + 0.05 * random_state.randn(1)[0]

        random_seed_ref += 1
        random_state = np.random.RandomState(random_seed_ref)
        self.pH = 6.5 + 0.1 * random_state.randn(1)[0]
        # converts from pH to H+ conc.
        self.pH = 10 ** (-self.pH)

        random_seed_ref += 1
        random_state = np.random.RandomState(random_seed_ref)
        self.T = 297 + 0.5 * random_state.randn(1)[0]

        random_seed_ref += 1
        self.a0 = intial_conds * 0.3333333333333333
        self.a1 = intial_conds * 0.6666666666666666
        self.a3 = 0
        self.a4 = 0
        self.Culture_age = 0

        random_seed_ref += 1
        random_state = np.random.RandomState(random_seed_ref)
        self.PAA = 1400 + 50 * random_state.randn(1)[0]

        random_seed_ref += 1
        random_state = np.random.RandomState(random_seed_ref)
        self.NH3 = 1700 + 50 * random_state.randn(1)[0]


class U:
    """
    Sequential batch control and PID control variables.
    """

    def __init__(self):
        self.Fault_ref = 0
        self.Fs = 0
        self.Foil = 0
        self.Fg = 0
        self.pressure = 0
        self.Fa = 0
        self.Fb = 0
        self.Fc = 0
        self.Fh = 0
        self.Fw = 0
        self.discharge = 0
        self.Fpaa = 0
        self.RPM = 0
        self.viscosity = 0
        self.NH3_shots = 0


class Xinterp:
    """
    Add filtered disturbance to batch data.
    """

    def __init__(self, random_seed_ref, batch_time):
        b1 = [0.005]
        a1 = [1, -0.995]

        random_state = np.random.RandomState(random_seed_ref)
        distMuP = lfilter(b1, a1, 0.03 * random_state.randn(NUM_STEPS + 1, 1), axis=0)
        self.distMuP = Channel(**{'name': 'Penicillin specific growth rate disturbance',
                                  'y_unit': 'g/Lh',
                                  't_unit': 'h',
                                  'time': batch_time,
                                  'value': distMuP})

        random_state = np.random.RandomState(random_seed_ref)
        distMuX = lfilter(b1, a1, 0.25 * random_state.randn(NUM_STEPS + 1, 1), axis=0)
        self.distMuX = Channel(**{'name': 'Biomass specific  growth rate disturbance',
                                  'y_unit': 'hr^{-1}',
                                  't_unit': 'h',
                                  'time': batch_time,
                                  'value': distMuX})

        random_state = np.random.RandomState(random_seed_ref)
        distcs = lfilter(b1, a1, 1500 * random_state.randn(NUM_STEPS + 1, 1), axis=0)
        self.distcs = Channel(**{'name': 'Substrate concentration disturbance',
                                 'y_unit': 'gL^{-1}',
                                 't_unit': 'h',
                                 'time': batch_time,
                                 'value': distcs})

        random_state = np.random.RandomState(random_seed_ref)
        distcoil = lfilter(b1, a1, 300 * random_state.randn(NUM_STEPS + 1, 1), axis=0)
        self.distcoil = Channel(**{'name': 'Oil inlet concentration disturbance',
                                   'y_unit': 'g L^{-1}',
                                   't_unit': 'h',
                                   'time': batch_time,
                                   'value': distcoil})

        random_state = np.random.RandomState(random_seed_ref)
        distabc = lfilter(b1, a1, 0.2 * random_state.randn(NUM_STEPS + 1, 1), axis=0)
        self.distabc = Channel(**{'name': 'Acid/Base molar inlet concentration disturbance',
                                  'y_unit': 'mol L^{-1}',
                                  't_unit': 'h',
                                  'time': batch_time,
                                  'value': distabc})

        random_state = np.random.RandomState(random_seed_ref)
        distPAA = lfilter(b1, a1, 300000 * random_state.randn(NUM_STEPS + 1, 1), axis=0)
        self.distPAA = Channel(**{'name': 'Phenylacetic acid concentration disturbance',
                                  'y_unit': 'g L^{-1}',
                                  't_unit': 'h',
                                  'time': batch_time,
                                  'value': distPAA})

        random_state = np.random.RandomState(random_seed_ref)
        distTcin = lfilter(b1, a1, 100 * random_state.randn(NUM_STEPS + 1, 1), axis=0)
        self.distTcin = Channel(**{'name': 'Coolant temperature inlet concentration disturbance',
                                   'y_unit': 'K',
                                   't_unit': 'h',
                                   'time': batch_time,
                                   'value': distTcin})

        random_state = np.random.RandomState(random_seed_ref)
        distO_2in = lfilter(b1, a1, 0.02 * random_state.randn(NUM_STEPS + 1, 1), axis=0)
        self.distO_2in = Channel(**{'name': 'Oxygen inlet concentration',
                                    'y_unit': '%',
                                    't_unit': 'h',
                                    'time': batch_time,
                                    'value': distO_2in})

        # extra
        self.NH3_shots = Channel()
        # hard code
        self.NH3_shots.y = [0] * NUM_STEPS