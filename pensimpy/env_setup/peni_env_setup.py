from pensimpy.pensim_classes.CtrlFlags import CtrlFlags
from pensimpy.pensim_classes.Constants import H, Batch_length
from pensimpy.pensim_methods.parameter_list import parameter_list
from pensimpy.pensim_classes.X0 import X0
from pensimpy.pensim_classes.Xinterp import Xinterp
import numpy as np
from pensimpy.pensim_methods.create_batch import create_batch
from pensimpy.pensim_methods.fctrl_indpensim import fctrl_indpensim
from scipy.integrate import odeint
from pensimpy.pensim_methods.indpensim_ode_py import indpensim_ode_py
from pensimpy.pensim_methods.raman_sim import raman_sim
from pensimpy.pensim_classes.Constants import raman_spectra
import math
from pensimpy.pensim_classes.Constants import raman_wavenumber


class PenSimEnv():
    def __init__(self):
        self.x = create_batch(H, Batch_length)
        self.xinterp = None
        self.x0 = None
        self.param_list = None
        self.ctrl_flags = CtrlFlags()
        self.yield_pre = 0

    def reset(self):
        """
        Setup the envs and return the observation class x.
        """
        # Enbaling seed for repeatable random numbers for different batches
        Random_seed_ref = int(np.ceil(np.random.rand(1)[0] * 1000))
        Seed_ref = 31 + Random_seed_ref
        intial_conds = 0.5 + 0.05 * np.random.randn(1)[0]

        # create x0
        self.x0 = X0(Seed_ref, intial_conds)

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
        self.xinterp = Xinterp(Random_seed_ref, Batch_length, H, np.arange(0, Batch_length + H, H))

        # param list
        self.param_list = parameter_list(self.x0.mup, self.x0.mux, alpha_kla, N_conc_paa, PAA_c)

        return self.x

    def step(self, k, x, Fs, Foil, Fg, Fpres, Fdischarge, Fw, Fpaa):
        """
        Simulate the fermentation process by solving ODE
        """
        # simulation timing init
        h_ode = H / 20
        t = np.arange(0, Batch_length + H, H)

        # fills the batch with just the initial conditions so the control system
        # can provide the first input. These will be overwritten after
        # the ODEs are integrated.
        if k == 1:
            x.S.y[0] = self.x0.S
            x.DO2.y[0] = self.x0.DO2
            x.X.y[0] = self.x0.X
            x.P.y[0] = self.x0.P
            x.V.y[0] = self.x0.V
            x.CO2outgas.y[0] = self.x0.CO2outgas
            x.pH.y[0] = self.x0.pH
            x.T.y[0] = self.x0.T

        # gets MVs
        u, x = fctrl_indpensim(x, self.xinterp, k, H, self.ctrl_flags, Fs, Foil, Fg, Fpres, Fdischarge, Fw, Fpaa)

        # builds initial conditions and control vectors specific to
        # indpensim_ode using ode45
        if k == 1:
            x00 = [self.x0.S,
                   self.x0.DO2,
                   self.x0.O2,
                   self.x0.P,
                   self.x0.V,
                   self.x0.Wt,
                   self.x0.pH,
                   self.x0.T,
                   0,
                   4,
                   self.x0.Culture_age,
                   self.x0.a0,
                   self.x0.a1,
                   self.x0.a3,
                   self.x0.a4,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   self.x0.CO2outgas,
                   0,
                   self.x0.PAA,
                   self.x0.NH3,
                   0,
                   0]
        else:
            x00 = [x.S.y[k - 2],
                   x.DO2.y[k - 2],
                   x.O2.y[k - 2],
                   x.P.y[k - 2],
                   x.V.y[k - 2],
                   x.Wt.y[k - 2],
                   x.pH.y[k - 2],
                   x.T.y[k - 2],
                   x.Q.y[k - 2],
                   x.Viscosity.y[k - 2],
                   x.Culture_age.y[k - 2],
                   x.a0.y[k - 2],
                   x.a1.y[k - 2],
                   x.a3.y[k - 2],
                   x.a4.y[k - 2],
                   x.n0.y[k - 2],
                   x.n1.y[k - 2],
                   x.n2.y[k - 2],
                   x.n3.y[k - 2],
                   x.n4.y[k - 2],
                   x.n5.y[k - 2],
                   x.n6.y[k - 2],
                   x.n7.y[k - 2],
                   x.n8.y[k - 2],
                   x.n9.y[k - 2],
                   x.nm.y[k - 2],
                   x.phi0.y[k - 2],
                   x.CO2outgas.y[k - 2],
                   x.CO2_d.y[k - 2],
                   x.PAA.y[k - 2],
                   x.NH3.y[k - 2],
                   0,
                   0]

        # Process disturbances
        distMuP = self.xinterp.distMuP.y[k - 1]
        distMuX = self.xinterp.distMuX.y[k - 1]
        distcs = self.xinterp.distcs.y[k - 1]
        distcoil = self.xinterp.distcoil.y[k - 1]
        distabc = self.xinterp.distabc.y[k - 1]
        distPAA = self.xinterp.distPAA.y[k - 1]
        distTcin = self.xinterp.distTcin.y[k - 1]
        distO_2in = self.xinterp.distO_2in.y[k - 1]

        u00 = [self.ctrl_flags.Inhib,
               u.Fs,
               u.Fg,
               u.RPM,
               u.Fc,
               u.Fh,
               u.Fb,
               u.Fa,
               h_ode,
               u.Fw,
               u.pressure,
               u.viscosity,
               u.Fremoved,
               u.Fpaa,
               u.Foil,
               u.NH3_shots,
               self.ctrl_flags.Dis,
               distMuP,
               distMuX,
               distcs,
               distcoil,
               distabc,
               distPAA,
               distTcin,
               distO_2in,
               self.ctrl_flags.Vis]

        # To account for inability of growth rates of biomass and penicillin to
        # return to normal after continuous periods of suboptimal pH and temperature conditions
        # If the Temperature or pH results is off set-point for k> 100 mu_p(max) is reduced to current value
        if self.ctrl_flags.Inhib == 1 or self.ctrl_flags.Inhib == 2:
            if k > 65:
                a1 = np.diff(x.mu_X_calc.y[k - 66:k - 1])
                a2 = [1 if x < 0 else 0 for x in a1]
                if sum(a2) >= 63:
                    self.param_list[1] = x.mu_X_calc.y[k - 2] * 5

        # Solver selection and calling indpensim_ode
        t_start = t[k - 1]
        t_end = t[k]
        t_span = np.arange(t_start, t_end + h_ode, h_ode).tolist()

        par = self.param_list.copy()
        par.extend(u00)

        # todo
        y_sol = odeint(indpensim_ode_py, x00, t_span, tfirst=True, args=(par,))
        y_sol = y_sol[-1]
        t_tmp = t_span[-1]

        # Defining minimum value for all variables for numerical stability
        y_sol[0:31] = [0.001 if ele <= 0 else ele for ele in y_sol[0:31]]

        # Saving all manipulated variables
        x.Fg.t[k - 1] = t_tmp
        x.Fg.y[k - 1] = u.Fg
        x.RPM.t[k - 1] = t_tmp
        x.RPM.y[k - 1] = u.RPM
        x.Fpaa.t[k - 1] = t_tmp
        x.Fpaa.y[k - 1] = u.Fpaa
        x.Fs.t[k - 1] = t_tmp
        x.Fs.y[k - 1] = u.Fs
        x.Fa.t[k - 1] = t_tmp
        x.Fa.y[k - 1] = u.Fa
        x.Fb.t[k - 1] = t_tmp
        x.Fb.y[k - 1] = u.Fb
        x.Fc.t[k - 1] = t_tmp
        x.Fc.y[k - 1] = u.Fc
        x.Foil.t[k - 1] = t_tmp
        x.Foil.y[k - 1] = u.Foil
        x.Fh.t[k - 1] = t_tmp
        x.Fh.y[k - 1] = u.Fh
        x.Fw.t[k - 1] = t_tmp
        x.Fw.y[k - 1] = u.Fw
        x.pressure.t[k - 1] = t_tmp
        x.pressure.y[k - 1] = u.pressure
        x.Fremoved.t[k - 1] = t_tmp
        x.Fremoved.y[k - 1] = u.Fremoved

        # Saving all the  IndPenSim states
        x.S.y[k - 1] = y_sol[0]
        x.S.t[k - 1] = t_tmp
        x.DO2.y[k - 1] = y_sol[1]

        # Required for numerical stability
        x.DO2.y[k - 1] = 1 if x.DO2.y[k - 1] < 2 else x.DO2.y[k - 1]

        x.DO2.t[k - 1] = t_tmp
        x.O2.y[k - 1] = y_sol[2]
        x.O2.t[k - 1] = t_tmp
        x.P.y[k - 1] = y_sol[3]
        x.P.t[k - 1] = t_tmp
        x.V.y[k - 1] = y_sol[4]
        x.V.t[k - 1] = t_tmp
        x.Wt.y[k - 1] = y_sol[5]
        x.Wt.t[k - 1] = t_tmp
        x.pH.y[k - 1] = y_sol[6]
        x.pH.t[k - 1] = t_tmp
        x.T.y[k - 1] = y_sol[7]
        x.T.t[k - 1] = t_tmp
        x.Q.y[k - 1] = y_sol[8]
        x.Q.t[k - 1] = t_tmp
        x.Viscosity.y[k - 1] = y_sol[9]
        x.Viscosity.t[k - 1] = t_tmp
        x.Culture_age.y[k - 1] = y_sol[10]
        x.Culture_age.t[k - 1] = t_tmp
        x.a0.y[k - 1] = y_sol[11]
        x.a0.t[k - 1] = t_tmp
        x.a1.y[k - 1] = y_sol[12]
        x.a1.t[k - 1] = t_tmp
        x.a3.y[k - 1] = y_sol[13]
        x.a3.t[k - 1] = t_tmp
        x.a4.y[k - 1] = y_sol[14]
        x.a4.t[k - 1] = t_tmp
        x.n0.y[k - 1] = y_sol[15]
        x.n0.t[k - 1] = t_tmp
        x.n1.y[k - 1] = y_sol[16]
        x.n1.t[k - 1] = t_tmp
        x.n2.y[k - 1] = y_sol[17]
        x.n2.t[k - 1] = t_tmp
        x.n3.y[k - 1] = y_sol[18]
        x.n3.t[k - 1] = t_tmp
        x.n4.y[k - 1] = y_sol[19]
        x.n4.t[k - 1] = t_tmp
        x.n5.y[k - 1] = y_sol[20]
        x.n5.t[k - 1] = t_tmp
        x.n6.y[k - 1] = y_sol[21]
        x.n6.t[k - 1] = t_tmp
        x.n7.y[k - 1] = y_sol[22]
        x.n7.t[k - 1] = t_tmp
        x.n8.y[k - 1] = y_sol[23]
        x.n8.t[k - 1] = t_tmp
        x.n9.y[k - 1] = y_sol[24]
        x.n9.t[k - 1] = t_tmp
        x.nm.y[k - 1] = y_sol[25]
        x.nm.t[k - 1] = t_tmp
        x.phi0.y[k - 1] = y_sol[26]
        x.phi0.t[k - 1] = t_tmp
        x.CO2outgas.y[k - 1] = y_sol[27]
        x.CO2outgas.t[k - 1] = t_tmp
        x.CO2_d.t[k - 1] = t_tmp
        x.CO2_d.y[k - 1] = y_sol[28]
        x.PAA.y[k - 1] = y_sol[29]
        x.PAA.t[k - 1] = t_tmp
        x.NH3.y[k - 1] = y_sol[30]
        x.NH3.t[k - 1] = t_tmp
        x.mu_P_calc.y[k - 1] = y_sol[31]
        x.mu_P_calc.t[k - 1] = t_tmp
        x.mu_X_calc.y[k - 1] = y_sol[32]
        x.mu_X_calc.t[k - 1] = t_tmp
        x.X.y[k - 1] = x.a0.y[k - 1] + x.a1.y[k - 1] + x.a3.y[k - 1] + x.a4.y[k - 1]
        x.X.t[k - 1] = t_tmp
        x.Fault_ref.y[k - 1] = u.Fault_ref
        x.Fault_ref.t[k - 1] = t_tmp
        x.Control_ref.y[k - 1] = self.ctrl_flags.PRBS
        x.Control_ref.t[k - 1] = self.ctrl_flags.Batch_Num
        x.PAT_ref.y[k - 1] = self.ctrl_flags.Raman_spec
        x.PAT_ref.t[k - 1] = self.ctrl_flags.Batch_Num
        x.Batch_ref.t[k - 1] = self.ctrl_flags.Batch_Num
        x.Batch_ref.y[k - 1] = self.ctrl_flags.Batch_Num

        # oxygen in air
        O2_in = 0.204

        # Calculating the OUR/ CER
        x.OUR.y[k - 1] = (1.4285714285714286 * x.Fg.y[k - 1]) * \
                         (O2_in - x.O2.y[k - 1] * (0.7902 / (1 - x.O2.y[k - 1] - x.CO2outgas.y[k - 1] / 100)))
        x.OUR.t[k - 1] = t_tmp

        # Calculating the CER
        x.CER.y[k - 1] = (1.9642857142857144 * x.Fg.y[k - 1]) * (
                (0.0065 * x.CO2outgas.y[k - 1]) * (0.7902 / (1 - O2_in - x.CO2outgas.y[k - 1] / 100) - 0.0330))
        x.CER.t[k - 1] = t_tmp

        # Adding in Raman Spectra
        # todo
        if k > 10:
            if self.ctrl_flags.Raman_spec == 1:
                x = raman_sim(k, x, H, Batch_length, raman_spectra)
            elif self.ctrl_flags.Raman_spec == 2:
                x = raman_sim(k, x, H, Batch_length, raman_spectra)

        # Off-line measurements recorded
        if np.remainder(t_tmp, self.ctrl_flags.Off_line_m) == 0 or t_tmp == 1 or t_tmp == Batch_length:
            delay = self.ctrl_flags.Off_line_delay
            x.NH3_offline.y[k - 1] = x.NH3.y[k - delay - 1]
            x.NH3_offline.t[k - 1] = x.NH3.t[k - delay - 1]
            x.Viscosity_offline.y[k - 1] = x.Viscosity.y[k - delay - 1]
            x.Viscosity_offline.t[k - 1] = x.Viscosity.t[k - delay - 1]
            x.PAA_offline.y[k - 1] = x.PAA.y[k - delay - 1]
            x.PAA_offline.t[k - 1] = x.PAA.t[k - delay - 1]
            x.P_offline.y[k - 1] = x.P.y[k - delay - 1]
            x.P_offline.t[k - 1] = x.P.t[k - delay - 1]
            x.X_offline.y[k - 1] = x.X.y[k - delay - 1]
            x.X_offline.t[k - 1] = x.X.t[k - delay - 1]
        else:
            x.NH3_offline.y[k - 1] = float('nan')
            x.NH3_offline.t[k - 1] = float('nan')
            x.Viscosity_offline.y[k - 1] = float('nan')
            x.Viscosity_offline.t[k - 1] = float('nan')
            x.PAA_offline.y[k - 1] = float('nan')
            x.PAA_offline.t[k - 1] = float('nan')
            x.P_offline.y[k - 1] = float('nan')
            x.P_offline.t[k - 1] = float('nan')
            x.X_offline.y[k - 1] = float('nan')
            x.X_offline.t[k - 1] = float('nan')

        peni_yield = x.V.y[k - 1] * x.P.y[k - 1] / 1000
        # peni_yield is accumulated penicillin
        # yield_pre is previous yield
        # x.Fremoved.y[k - 1] * x.P.y[k - 1] * h / 1000  is the discharged
        yield_per_run = peni_yield - self.yield_pre - x.Fremoved.y[k - 1] * x.P.y[k - 1] * H / 1000
        self.yield_pre = peni_yield

        done = True if k == Batch_length / H else False
        if done:
            # post process
            # convert to pH from H+ concentration
            x.pH.y = [-math.log(pH) / math.log(10) if pH != 0 else pH for pH in x.pH.y]
            x.Q.y = [Q / 1000 for Q in x.Q.y]
            x.Raman_Spec.Wavenumber = raman_wavenumber

        return x, yield_per_run, done
