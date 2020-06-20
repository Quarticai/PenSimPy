from pensimpy.pensim_classes.CtrlFlags import CtrlFlags
import numpy as np
from pensimpy.pensim_classes.Constants import H, Batch_lenght
from pensimpy.pensim_methods.parameter_list import parameter_list
from pensimpy.pensim_classes.X0 import X0
from pensimpy.pensim_classes.Xinterp import Xinterp


class PenSimEnv():
    def reset(self):
        ctrl_flags = CtrlFlags()

        # Enbaling seed for repeatable random numbers for different batches
        Random_seed_ref = int(np.ceil(np.random.rand(1)[0] * 1000))
        Seed_ref = 31 + Random_seed_ref
        intial_conds = 0.5 + 0.05 * np.random.randn(1)[0]

        # create x0
        x0 = X0(Seed_ref, intial_conds)

        # alpha_kla
        Seed_ref += 14
        np.random.seed(Seed_ref)
        alpha_kla = 85 + 10 * np.random.randn(1)[0]

        # PAA_c
        Seed_ref += 1
        np.random.seed(Seed_ref)
        PAA_c = 530000 + 20000 * np.random.randn(1)[0]

        # N_conc_paa
        Seed_ref += 1
        np.random.seed(Seed_ref)
        N_conc_paa = 150000 + 2000 * np.random.randn(1)[0]

        # create xinterp
        xinterp = Xinterp(Random_seed_ref, Batch_lenght, H, np.arange(0, Batch_lenght + H, H))

        # param list
        param_list = parameter_list(x0.mup, x0.mux, alpha_kla, N_conc_paa, PAA_c)

        return xinterp, x0, H, Batch_lenght, param_list, ctrl_flags
