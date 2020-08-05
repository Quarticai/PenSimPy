from pensimpy.data.channel import Channel
import numpy as np
from scipy.signal import lfilter


class Xinterp:
    """
    Noise added by randn and filtered be a low pass filter
    """

    def __init__(self, Random_seed_ref, T, h, Batch_time):
        b1 = [0.005]
        a1 = [1, -0.995]

        np.random.seed(Random_seed_ref)
        distMuP = lfilter(b1, a1, 0.03 * np.random.randn(int(T / h + 1), 1), axis=0)
        self.distMuP = Channel(**{'name': 'Penicillin specific growth rate disturbance',
                                  'yUnit': 'g/Lh',
                                  'tUnit': 'h',
                                  'time': Batch_time,
                                  'value': distMuP})

        np.random.seed(Random_seed_ref)
        distMuX = lfilter(b1, a1, 0.25 * np.random.randn(int(T / h + 1), 1), axis=0)
        self.distMuX = Channel(**{'name': 'Biomass specific  growth rate disturbance',
                                  'yUnit': 'hr^{-1}',
                                  'tUnit': 'h',
                                  'time': Batch_time,
                                  'value': distMuX})

        np.random.seed(Random_seed_ref)
        distcs = lfilter(b1, a1, 1500 * np.random.randn(int(T / h + 1), 1), axis=0)
        self.distcs = Channel(**{'name': 'Substrate concentration disturbance',
                                 'yUnit': 'gL^{-1}',
                                 'tUnit': 'h',
                                 'time': Batch_time,
                                 'value': distcs})

        np.random.seed(Random_seed_ref)
        distcoil = lfilter(b1, a1, 300 * np.random.randn(int(T / h + 1), 1), axis=0)
        self.distcoil = Channel(**{'name': 'Oil inlet concentration disturbance',
                                   'yUnit': 'g L^{-1}',
                                   'tUnit': 'h',
                                   'time': Batch_time,
                                   'value': distcoil})

        np.random.seed(Random_seed_ref)
        distabc = lfilter(b1, a1, 0.2 * np.random.randn(int(T / h + 1), 1), axis=0)
        self.distabc = Channel(**{'name': 'Acid/Base molar inlet concentration disturbance',
                                  'yUnit': 'mol L^{-1}',
                                  'tUnit': 'h',
                                  'time': Batch_time,
                                  'value': distabc})

        np.random.seed(Random_seed_ref)
        distPAA = lfilter(b1, a1, 300000 * np.random.randn(int(T / h + 1), 1), axis=0)
        self.distPAA = Channel(**{'name': 'Phenylacetic acid concentration disturbance',
                                  'yUnit': 'g L^{-1}',
                                  'tUnit': 'h',
                                  'time': Batch_time,
                                  'value': distPAA})

        np.random.seed(Random_seed_ref)
        distTcin = lfilter(b1, a1, 100 * np.random.randn(int(T / h + 1), 1), axis=0)
        self.distTcin = Channel(**{'name': 'Coolant temperature inlet concentration disturbance',
                                   'yUnit': 'K',
                                   'tUnit': 'h',
                                   'time': Batch_time,
                                   'value': distTcin})

        np.random.seed(Random_seed_ref)
        distO_2in = lfilter(b1, a1, 0.02 * np.random.randn(int(T / h + 1), 1), axis=0)
        self.distO_2in = Channel(**{'name': 'Oxygen inlet concentration',
                                    'yUnit': '%',
                                    'tUnit': 'h',
                                    'time': Batch_time,
                                    'value': distO_2in})

        # extra
        self.NH3_shots = Channel()
        # hard code
        self.NH3_shots.y = [0] * 2000
