from pensimpy.pensim_classes.Channel import Channel
from pensimpy.pensim_methods.create_channel import create_channel
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
        channel = Channel()
        create_channel(channel, **{'name': 'Penicillin specific growth rate disturbance',
                                   'yUnit': 'g/Lh',
                                   'tUnit': 'h',
                                   'time': Batch_time,
                                   'value': distMuP})
        self.distMuP = channel

        distMuX = lfilter(b1, a1, 0.25 * np.random.randn(int(T / h + 1), 1), axis=0)
        channel = Channel()
        create_channel(channel, **{'name': 'Biomass specific  growth rate disturbance',
                                   'yUnit': 'hr^{-1}',
                                   'tUnit': 'h',
                                   'time': Batch_time,
                                   'value': distMuX})
        self.distMuX = channel

        distcs = lfilter(b1, a1, 1500 * np.random.randn(int(T / h + 1), 1), axis=0)
        channel = Channel()
        create_channel(channel, **{'name': 'Substrate concentration disturbance',
                                   'yUnit': 'gL^{-1}',
                                   'tUnit': 'h',
                                   'time': Batch_time,
                                   'value': distcs})
        self.distcs = channel

        distcoil = lfilter(b1, a1, 300 * np.random.randn(int(T / h + 1), 1), axis=0)
        channel = Channel()
        create_channel(channel, **{'name': 'Oil inlet concentration disturbance',
                                   'yUnit': 'g L^{-1}',
                                   'tUnit': 'h',
                                   'time': Batch_time,
                                   'value': distcoil})
        self.distcoil = channel

        distabc = lfilter(b1, a1, 0.2 * np.random.randn(int(T / h + 1), 1), axis=0)
        channel = Channel()
        create_channel(channel, **{'name': 'Acid/Base molar inlet concentration disturbance',
                                   'yUnit': 'mol L^{-1}',
                                   'tUnit': 'h',
                                   'time': Batch_time,
                                   'value': distabc})
        self.distabc = channel

        distPAA = lfilter(b1, a1, 300000 * np.random.randn(int(T / h + 1), 1), axis=0)
        channel = Channel()
        create_channel(channel, **{'name': 'Phenylacetic acid concentration disturbance',
                                   'yUnit': 'g L^{-1}',
                                   'tUnit': 'h',
                                   'time': Batch_time,
                                   'value': distPAA})
        self.distPAA = channel

        distTcin = lfilter(b1, a1, 100 * np.random.randn(int(T / h + 1), 1), axis=0)
        channel = Channel()
        create_channel(channel, **{'name': 'Coolant temperature inlet concentration disturbance',
                                   'yUnit': 'K',
                                   'tUnit': 'h',
                                   'time': Batch_time,
                                   'value': distTcin})
        self.distTcin = channel

        distO_2in = lfilter(b1, a1, 0.02 * np.random.randn(int(T / h + 1), 1), axis=0)
        channel = Channel()
        create_channel(channel, **{'name': 'Oxygen inlet concentration',
                                   'yUnit': '%',
                                   'tUnit': 'h',
                                   'time': Batch_time,
                                   'value': distO_2in})
        self.distO_2in = channel

        # extra
        self.NH3_shots = Channel()
        # hard code
        self.NH3_shots.y = [0] * 2000
