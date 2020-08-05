import numpy as np


class X0:
    """
    Initial states for each batch, also contains some process params ('mux', 'mup')
    """

    def __init__(self, Seed_ref, intial_conds):
        np.random.seed(Seed_ref)
        self.mux = 0.41 + 0.025 * np.random.randn(1)[0]

        Seed_ref += 1
        np.random.seed(Seed_ref)
        self.mup = 0.041 + 0.0025 * np.random.randn(1)[0]

        Seed_ref += 1
        np.random.seed(Seed_ref)
        self.S = 1 + 0.1 * np.random.randn(1)[0]

        Seed_ref += 1
        np.random.seed(Seed_ref)
        self.DO2 = 15 + 0.5 * np.random.randn(1)[0]

        Seed_ref += 1
        np.random.seed(Seed_ref)
        self.X = intial_conds + 0.1 * np.random.randn(1)[0]
        self.P = 0

        Seed_ref += 1
        np.random.seed(Seed_ref)
        self.V = 5.800e+04 + 500 * np.random.randn(1)[0]

        Seed_ref += 1
        np.random.seed(Seed_ref)
        self.Wt = 6.2e+04 + 500 * np.random.randn(1)[0]

        Seed_ref += 1
        np.random.seed(Seed_ref)
        self.CO2outgas = 0.038 + 0.001 * np.random.randn(1)[0]

        Seed_ref += 1
        np.random.seed(Seed_ref)
        self.O2 = 0.20 + 0.05 * np.random.randn(1)[0]

        Seed_ref += 1
        np.random.seed(Seed_ref)
        self.pH = 6.5 + 0.1 * np.random.randn(1)[0]
        # converts from pH to H+ conc.
        self.pH = 10 ** (-self.pH)

        Seed_ref += 1
        np.random.seed(Seed_ref)
        self.T = 297 + 0.5 * np.random.randn(1)[0]

        Seed_ref += 1
        np.random.seed(Seed_ref)
        self.a0 = intial_conds * 0.3333333333333333
        self.a1 = intial_conds * 0.6666666666666666
        self.a3 = 0
        self.a4 = 0
        self.Culture_age = 0

        Seed_ref += 1
        np.random.seed(Seed_ref)
        self.PAA = 1400 + 50 * np.random.randn(1)[0]

        Seed_ref += 1
        np.random.seed(Seed_ref)
        self.NH3 = 1700 + 50 * np.random.randn(1)[0]
