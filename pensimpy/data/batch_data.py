import numpy as np
from pensimpy.data.channel import Channel
from scipy.signal import lfilter


class X:
    """
    Batch data. All features included.
    """

    def __init__(self):
        self.Fg = ''
        self.RPM = ''
        self.Fs = ''
        self.sc = ''
        self.abc = ''
        self.Fa = ''
        self.Fb = ''
        self.Fc = ''
        self.Fh = ''
        self.Fw = ''
        self.pressure = ''
        self.Fremoved = ''
        self.S = ''
        self.DO2 = ''
        self.X = ''
        self.P = ''
        self.V = ''
        self.Wt = ''
        self.pH = ''
        self.T = ''
        self.Q = ''
        self.a0 = ''
        self.a1 = ''
        self.a3 = ''
        self.a4 = ''
        self.n0 = ''
        self.n1 = ''
        self.n2 = ''
        self.n3 = ''
        self.n4 = ''
        self.n5 = ''
        self.n6 = ''
        self.n7 = ''
        self.n8 = ''
        self.n9 = ''
        self.nm = ''
        self.phi0 = ''
        self.CO2outgas = ''
        self.Culture_age = ''
        self.Fpaa = ''
        self.PAA = ''
        self.PAA_offline = ''
        self.NH3 = ''
        self.NH3_offline = ''
        self.OUR = ''
        self.O2 = ''
        self.mup = ''
        self.mux = ''
        self.P_offline = ''
        self.X_CER = ''
        self.X_offline = ''
        self.CER = ''
        self.mu_X_calc = ''
        self.mu_P_calc = ''
        self.F_discharge_cal = ''
        self.NH3_shots = ''
        self.CO2_d = ''
        self.Viscosity = ''
        self.Viscosity_offline = ''
        self.Fault_ref = ''
        self.Control_ref = ''
        self.PAT_ref = ''
        self.Batch_ref = ''
        # extra
        self.PRBS_noise_addition = [0] * 1750


class X0:
    """
    Initialize the batch data.
    """

    def __init__(self, Seed_ref, intial_conds):
        random_state = np.random.RandomState(Seed_ref)
        self.mux = 0.41 + 0.025 * random_state.randn(1)[0]

        Seed_ref += 1
        random_state = np.random.RandomState(Seed_ref)
        self.mup = 0.041 + 0.0025 * random_state.randn(1)[0]

        Seed_ref += 1
        random_state = np.random.RandomState(Seed_ref)
        self.S = 1 + 0.1 * random_state.randn(1)[0]

        Seed_ref += 1
        random_state = np.random.RandomState(Seed_ref)
        self.DO2 = 15 + 0.5 * random_state.randn(1)[0]

        Seed_ref += 1
        random_state = np.random.RandomState(Seed_ref)
        self.X = intial_conds + 0.1 * random_state.randn(1)[0]
        self.P = 0

        Seed_ref += 1
        random_state = np.random.RandomState(Seed_ref)
        self.V = 5.800e+04 + 500 * random_state.randn(1)[0]

        Seed_ref += 1
        random_state = np.random.RandomState(Seed_ref)
        self.Wt = 6.2e+04 + 500 * random_state.randn(1)[0]

        Seed_ref += 1
        random_state = np.random.RandomState(Seed_ref)
        self.CO2outgas = 0.038 + 0.001 * random_state.randn(1)[0]

        Seed_ref += 1
        random_state = np.random.RandomState(Seed_ref)
        self.O2 = 0.20 + 0.05 * random_state.randn(1)[0]

        Seed_ref += 1
        random_state = np.random.RandomState(Seed_ref)
        self.pH = 6.5 + 0.1 * random_state.randn(1)[0]
        # converts from pH to H+ conc.
        self.pH = 10 ** (-self.pH)

        Seed_ref += 1
        random_state = np.random.RandomState(Seed_ref)
        self.T = 297 + 0.5 * random_state.randn(1)[0]

        Seed_ref += 1
        self.a0 = intial_conds * 0.3333333333333333
        self.a1 = intial_conds * 0.6666666666666666
        self.a3 = 0
        self.a4 = 0
        self.Culture_age = 0

        Seed_ref += 1
        random_state = np.random.RandomState(Seed_ref)
        self.PAA = 1400 + 50 * random_state.randn(1)[0]

        Seed_ref += 1
        random_state = np.random.RandomState(Seed_ref)
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
        self.Fremoved = 0
        self.Fpaa = 0
        self.RPM = 0
        self.viscosity = 0
        self.NH3_shots = 0


class Xinterp:
    """
    Add disturbance to batch data.
    """

    def __init__(self, random_seed_ref, T, h, batch_time):
        b1 = [0.005]
        a1 = [1, -0.995]

        random_state = np.random.RandomState(random_seed_ref)
        distMuP = lfilter(b1, a1, 0.03 * random_state.randn(int(T / h + 1), 1), axis=0)
        self.distMuP = Channel(**{'name': 'Penicillin specific growth rate disturbance',
                                  'y_unit': 'g/Lh',
                                  't_unit': 'h',
                                  'time': batch_time,
                                  'value': distMuP})

        random_state = np.random.RandomState(random_seed_ref)
        distMuX = lfilter(b1, a1, 0.25 * random_state.randn(int(T / h + 1), 1), axis=0)
        self.distMuX = Channel(**{'name': 'Biomass specific  growth rate disturbance',
                                  'y_unit': 'hr^{-1}',
                                  't_unit': 'h',
                                  'time': batch_time,
                                  'value': distMuX})

        random_state = np.random.RandomState(random_seed_ref)
        distcs = lfilter(b1, a1, 1500 * random_state.randn(int(T / h + 1), 1), axis=0)
        self.distcs = Channel(**{'name': 'Substrate concentration disturbance',
                                 'y_unit': 'gL^{-1}',
                                 't_unit': 'h',
                                 'time': batch_time,
                                 'value': distcs})

        random_state = np.random.RandomState(random_seed_ref)
        distcoil = lfilter(b1, a1, 300 * random_state.randn(int(T / h + 1), 1), axis=0)
        self.distcoil = Channel(**{'name': 'Oil inlet concentration disturbance',
                                   'y_unit': 'g L^{-1}',
                                   't_unit': 'h',
                                   'time': batch_time,
                                   'value': distcoil})

        random_state = np.random.RandomState(random_seed_ref)
        distabc = lfilter(b1, a1, 0.2 * random_state.randn(int(T / h + 1), 1), axis=0)
        self.distabc = Channel(**{'name': 'Acid/Base molar inlet concentration disturbance',
                                  'y_unit': 'mol L^{-1}',
                                  't_unit': 'h',
                                  'time': batch_time,
                                  'value': distabc})

        random_state = np.random.RandomState(random_seed_ref)
        distPAA = lfilter(b1, a1, 300000 * random_state.randn(int(T / h + 1), 1), axis=0)
        self.distPAA = Channel(**{'name': 'Phenylacetic acid concentration disturbance',
                                  'y_unit': 'g L^{-1}',
                                  't_unit': 'h',
                                  'time': batch_time,
                                  'value': distPAA})

        random_state = np.random.RandomState(random_seed_ref)
        distTcin = lfilter(b1, a1, 100 * random_state.randn(int(T / h + 1), 1), axis=0)
        self.distTcin = Channel(**{'name': 'Coolant temperature inlet concentration disturbance',
                                   'y_unit': 'K',
                                   't_unit': 'h',
                                   'time': batch_time,
                                   'value': distTcin})

        random_state = np.random.RandomState(random_seed_ref)
        distO_2in = lfilter(b1, a1, 0.02 * random_state.randn(int(T / h + 1), 1), axis=0)
        self.distO_2in = Channel(**{'name': 'Oxygen inlet concentration',
                                    'y_unit': '%',
                                    't_unit': 'h',
                                    'time': batch_time,
                                    'value': distO_2in})

        # extra
        self.NH3_shots = Channel()
        # hard code
        self.NH3_shots.y = [0] * 2000