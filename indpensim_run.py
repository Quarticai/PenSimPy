from CtrlFlags import CtrlFlags
import numpy as np
from scipy.signal import lfilter
from create_channel import create_channel
from Channel import Channel
from parameter_list import parameter_list
from X0 import X0
from Xinterp import Xinterp
from indpensim import indpensim


def indpensim_run(Batch_no, batch_run_flags):
    """
    Initialize the params and call the simulator
    :param Batch_no:
    :param batch_run_flags:
    :return:
    """
    parms = {'PRBS': batch_run_flags.Control_strategy[0][Batch_no - 1],
             'Fixed_Batch_length': batch_run_flags.Batch_length[0][Batch_no - 1],
             'Faults': batch_run_flags.Batch_fault_order_reference[Batch_no - 1][0],
             'Raman_spec': batch_run_flags.Raman_spec[0][Batch_no - 1],
             'Batch_Num': Batch_no}
    ctrl_flags = CtrlFlags(**parms)

    # Standard batch simulation with randomised initial conditions and batch
    if ctrl_flags.IC == 0:
        ctrl_flags.SBC = 0
        ctrl_flags.Vis = 0
        Optimum_Batch_lenght = 230
        # set a fixed batch length
        T = Optimum_Batch_lenght
        # if ctrl_flags.Fixed_Batch_length == 1:
        #     Batch_length_variation = 25 * np.random.randn(1)[0]
        #     T = int(round(Optimum_Batch_lenght + Batch_length_variation))
        # elif ctrl_flags.Fixed_Batch_length == 0:
        #     T = Optimum_Batch_lenght

        # Enbaling seed for repeatable random numbers for different batches
        Random_seed_ref = int(np.ceil(np.random.rand(1)[0] * 1000))
        Seed_ref = 31 + Random_seed_ref
        Rand_ref = 1

        # Defining consistent seed for random number generator for each variable
        np.random.seed(Seed_ref + Batch_no + Rand_ref)
        Rand_ref = Rand_ref + 1
        intial_conds = 0.5 + 0.05 * np.random.randn(1)[0]
        np.random.seed(Seed_ref + Batch_no + Rand_ref)
        Rand_ref = Rand_ref + 1

        # create x0
        x0 = X0()
        x0.mux = 0.41 + 0.025 * np.random.randn(1)[0]
        np.random.seed(Seed_ref + Batch_no + Rand_ref)
        Rand_ref = Rand_ref + 1
        x0.mup = 0.041 + 0.0025 * np.random.randn(1)[0]
        h = 0.2

        #  Initialising simulation
        np.random.seed(Seed_ref + Batch_no + Rand_ref)
        Rand_ref = Rand_ref + 1
        x0.S = 1 + 0.1 * np.random.randn(1)[0]

        np.random.seed(Seed_ref + Batch_no + Rand_ref)
        Rand_ref = Rand_ref + 1
        x0.DO2 = 15 + 0.5 * np.random.randn(1)[0]

        np.random.seed(Seed_ref + Batch_no + Rand_ref)
        Rand_ref = Rand_ref + 1
        x0.X = intial_conds + 0.1 * np.random.randn(1)[0]

        x0.P = 0
        np.random.seed(Seed_ref + Batch_no + Rand_ref)
        Rand_ref = Rand_ref + 1
        x0.V = 5.800e+04 + 500 * np.random.randn(1)[0]

        np.random.seed(Seed_ref + Batch_no + Rand_ref)
        Rand_ref = Rand_ref + 1
        x0.Wt = 6.2e+04 + 500 * np.random.randn(1)[0]

        np.random.seed(Seed_ref + Batch_no + Rand_ref)
        Rand_ref = Rand_ref + 1
        x0.CO2outgas = 0.038 + 0.001 * np.random.randn(1)[0]

        np.random.seed(Seed_ref + Batch_no + Rand_ref)
        Rand_ref = Rand_ref + 1
        x0.O2 = 0.20 + 0.05 * np.random.randn(1)[0]

        np.random.seed(Seed_ref + Batch_no + Rand_ref)
        Rand_ref = Rand_ref + 1
        x0.pH = 6.5 + 0.1 * np.random.randn(1)[0]

        np.random.seed(Seed_ref + Batch_no + Rand_ref)
        Rand_ref = Rand_ref + 1
        x0.T = 297 + 0.5 * np.random.randn(1)[0]

        np.random.seed(Seed_ref + Batch_no + Rand_ref)
        Rand_ref = Rand_ref + 1
        x0.a0 = intial_conds * (1 / 3)
        x0.a1 = intial_conds * (2 / 3)
        x0.a3 = 0
        x0.a4 = 0
        x0.Culture_age = 0
        np.random.seed(Seed_ref + Batch_no + Rand_ref)
        Rand_ref = Rand_ref + 1

        x0.PAA = 1400 + 50 * np.random.randn(1)[0]
        np.random.seed(Seed_ref + Batch_no + Rand_ref)
        Rand_ref = Rand_ref + 1

        x0.NH3 = 1700 + 50 * np.random.randn(1)[0]
        np.random.seed(Seed_ref + Batch_no + Rand_ref)
        Rand_ref = Rand_ref + 1

        alpha_kla = 85 + 10 * np.random.randn(1)[0]
        np.random.seed(Seed_ref + Batch_no + Rand_ref)
        Rand_ref = Rand_ref + 1

        PAA_c = 530000 + 20000 * np.random.randn(1)[0]
        np.random.seed(Seed_ref + Batch_no + Rand_ref)

        N_conc_paa = 2 * 75000 + 2000 * np.random.randn(1)[0]
        Batch_time = np.arange(0, T + h, h)
        ctrl_flags.T_sp = 298
        ctrl_flags.pH_sp = 6.5

    np.random.seed(Random_seed_ref + Batch_no)
    # Creates process disturbances on growth rates as well as process inputs
    # using a low pass filter
    b1 = [0.005]
    a1 = [1, -0.995]

    # Penicillin specific growth rate disturbance: with SD of +/- 0.0009 [hr^{-1}]
    v = np.random.randn(int(T / h + 1), 1)
    distMuP = lfilter(b1, a1, 0.03 * v, axis=0)
    channel = Channel()
    xinterp = Xinterp()
    create_channel(channel, **{'name': 'Penicillin specific growth rate disturbance',
                               'yUnit': 'g/Lh',
                               'tUnit': 'h',
                               'time': Batch_time,
                               'value': distMuP})
    xinterp.distMuP = channel

    # Biomass specific growth rate disturbance: with SD  +/- 0.011 [hr^{-1}]
    v = np.random.randn(int(T / h + 1), 1)
    distMuX = lfilter(b1, a1, 0.25 * v, axis=0)
    channel = Channel()
    create_channel(channel, **{'name': 'Biomass specific  growth rate disturbance',
                               'yUnit': 'hr^{-1}',
                               'tUnit': 'h',
                               'time': Batch_time,
                               'value': distMuX})
    xinterp.distMuX = channel

    # Substrate inlet concentration disturbance: +/- 15 [g L^{-1}]
    v = np.random.randn(int(T / h + 1), 1)
    distcs = lfilter(b1, a1, 5 * 300 * v, axis=0)
    channel = Channel()
    create_channel(channel, **{'name': 'Substrate concentration disturbance',
                               'yUnit': 'gL^{-1}',
                               'tUnit': 'h',
                               'time': Batch_time,
                               'value': distcs})
    xinterp.distcs = channel

    # Oil inlet concentration disturbance: +/- 15 [g L^{-1}]
    v = np.random.randn(int(T / h + 1), 1)
    distcoil = lfilter(b1, a1, 300 * v, axis=0)
    channel = Channel()
    create_channel(channel, **{'name': 'Oil inlet concentration disturbance',
                               'yUnit': 'g L^{-1}',
                               'tUnit': 'h',
                               'time': Batch_time,
                               'value': distcoil})
    xinterp.distcoil = channel

    # Acid/Base molar inlet concentration disturbance: +/- 0.004 [mol L^{-1}]
    v = np.random.randn(int(T / h + 1), 1)
    distabc = lfilter(b1, a1, 0.2 * v, axis=0)
    channel = Channel()
    create_channel(channel, **{'name': 'Acid/Base molar inlet concentration disturbance',
                               'yUnit': 'mol L^{-1}',
                               'tUnit': 'h',
                               'time': Batch_time,
                               'value': distabc})
    xinterp.distabc = channel

    # Phenylacetic acid concentration disturbance: g L^{-1}
    v = np.random.randn(int(T / h + 1), 1)
    distPAA = lfilter(b1, a1, 300000 * v, axis=0)
    channel = Channel()
    create_channel(channel, **{'name': 'Phenylacetic acid concentration disturbance',
                               'yUnit': 'g L^{-1}',
                               'tUnit': 'h',
                               'time': Batch_time,
                               'value': distPAA})
    xinterp.distPAA = channel

    # Coolant temperature inlet concentration disturbance: +/- 3 [K]
    v = np.random.randn(int(T / h + 1), 1)
    distTcin = lfilter(b1, a1, 100 * v, axis=0)
    channel = Channel()
    create_channel(channel, **{'name': 'Coolant temperature inlet concentration disturbance',
                               'yUnit': 'K',
                               'tUnit': 'h',
                               'time': Batch_time,
                               'value': distTcin})
    xinterp.distTcin = channel

    # Oxygen inlet concentration: +/- 0.009 [%]
    v = np.random.randn(int(T / h + 1), 1)
    distO_2in = lfilter(b1, a1, 0.02 * v, axis=0)
    channel = Channel()
    create_channel(channel, **{'name': 'Oxygen inlet concentration',
                               'yUnit': '%',
                               'tUnit': 'h',
                               'time': Batch_time,
                               'value': distO_2in})
    xinterp.distO_2in = channel

    # Import parameter list
    param_list = parameter_list(x0, alpha_kla, N_conc_paa, PAA_c)

    # Run simulation
    print("=== Run simulation...")
    Xref = indpensim(xinterp, x0, h, T, 2, param_list, ctrl_flags)
    return Xref, h
