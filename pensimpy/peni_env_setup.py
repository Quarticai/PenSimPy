import numpy as np
import math
import time
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from pensimpy.data.ctrl_flags import CtrlFlags
from pensimpy.data.batch_data import X0, Xinterp, U, X
from pensimpy.constants import RAMAN_SPECTRA, RAMAN_WAVENUMBER, STEP_IN_MINUTES, BATCH_LENGTH_IN_HOURS, STEP_IN_HOURS, NUM_STEPS, WAVENUMBER_LENGTH
from pensimpy.ode.indpensim_ode_py import indpensim_ode_py
from pensimpy.utils import pid_controller, smooth, get_dataframe, get_observation_data
import fastodeint


class PenSimEnv:
    """
    Class for setting up the simulation environment, simulating the penicillin yield process with Raman spectra, and
    generating the batch data and Raman spectra data in pandas dataframe.
    """
    def __init__(self, recipe, fast=True):
        self.xinterp = None
        self.x0 = None
        self.param_list = None
        self.ctrl_flags = CtrlFlags()
        self.yield_pre = 0
        self.random_seed_ref = 0
        self.fast = fast
        self.recipe = recipe

    def reset(self):
        """
        Setup the envs and return the observation class x.
        """
        # Enbaling seed for repeatable random numbers for different batches
        seed_ref = 31 + self.random_seed_ref
        random_state = np.random.RandomState(seed_ref)
        intial_conds = 0.5 + 0.05 * random_state.randn(1)[0]

        # create x0
        self.x0 = X0(seed_ref, intial_conds)

        # alpha_kla
        seed_ref += 14
        random_state = np.random.RandomState(seed_ref)
        alpha_kla = 85 + 10 * random_state.randn(1)[0]

        # PAA_c
        seed_ref += 1
        random_state = np.random.RandomState(seed_ref)
        PAA_c = 530000 + 20000 * random_state.randn(1)[0]

        # N_conc_paa
        seed_ref += 1
        random_state = np.random.RandomState(seed_ref)
        N_conc_paa = 150000 + 2000 * random_state.randn(1)[0]

        # create xinterp
        self.xinterp = Xinterp(self.random_seed_ref, np.arange(0, BATCH_LENGTH_IN_HOURS + STEP_IN_HOURS, STEP_IN_HOURS))

        # param list
        # self.param_list = parameter_list(self.x0.mup, self.x0.mux, alpha_kla, N_conc_paa, PAA_c)
        self.param_list = [self.x0.mup, self.x0.mux, alpha_kla, N_conc_paa, PAA_c]

        # create the observation class
        x = X()

        # get observation
        observation = get_observation_data(x, 0)
        return observation, x

    def step(self, k, x, Fs, Foil, Fg, pressure, discharge, Fw, Fpaa):
        """
        Simulate the fermentation process by solving ODE.
        """
        # simulation timing init
        h_ode = STEP_IN_HOURS / 40
        t = np.arange(0, BATCH_LENGTH_IN_HOURS + STEP_IN_HOURS, STEP_IN_HOURS)

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

        # apply PID and interpolations
        u, x = self.integrate_control_strategy(x, k, Fs, Foil, Fg, pressure, discharge, Fw, Fpaa)

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
               u.discharge,
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

        if self.fast:
            y_sol = fastodeint.integrate(x00, par, t_start, t_end + h_ode, h_ode)
            t_tmp = t_end + h_ode
        else:
            y_sol = odeint(indpensim_ode_py, x00, t_span, tfirst=True, args=(par,))
            y_sol = y_sol[-1]
            t_tmp = t_span[-1]

        # # Defining minimum value for all variables for numerical stability
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
        x.discharge.t[k - 1] = t_tmp
        x.discharge.y[k - 1] = u.discharge

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
        if k > 10:
            if self.ctrl_flags.Raman_spec == 1:
                x = self.raman_sim(k, x)
            elif self.ctrl_flags.Raman_spec == 2:
                x = self.raman_sim(k, x)

        # Off-line measurements recorded
        if np.remainder(t_tmp, self.ctrl_flags.Off_line_m) == 0 or t_tmp == 1 or t_tmp == BATCH_LENGTH_IN_HOURS:
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

        x.V.y[k - 1] = np.nan_to_num(x.V.y[k - 1])
        x.P.y[k - 1] = np.nan_to_num(x.P.y[k - 1])
        peni_yield = x.V.y[k - 1] * x.P.y[k - 1] / 1000
        # peni_yield is accumulated penicillin
        # yield_pre is previous yield
        # x.discharge.y[k - 1] * x.P.y[k - 1] * h / 1000  is the discharged
        yield_per_run = peni_yield - self.yield_pre - x.discharge.y[k - 1] * x.P.y[k - 1] * STEP_IN_HOURS / 1000
        self.yield_pre = peni_yield

        observation = get_observation_data(x, k - 1)

        done = True if k == NUM_STEPS else False
        if done:
            # post process
            # convert to pH from H+ concentration
            x.pH.y = [-math.log(pH) / math.log(10) if pH != 0 else pH for pH in x.pH.y]
            x.Q.y = [Q / 1000 for Q in x.Q.y]
            x.Raman_Spec.Wavenumber = RAMAN_WAVENUMBER

        return observation, x, yield_per_run, done

    def integrate_control_strategy(self, x, k, Fs_k, Foil_k, Fg_k, pressure_k, discharge_k, Fw_k, Fpaa_k):
        """
        Control strategies: Sequential batch control and PID control.
        """
        # pH controller
        u = U()
        pH_sensor_error = 0
        if self.ctrl_flags.Faults == 8:
            pH_sensor_error = 0.1
            ramp_function = [[0, 0],
                             [200, 0],
                             [800, pH_sensor_error],
                             [1750, pH_sensor_error]]
            ramp_function = np.array(ramp_function)
            t_interp = np.arange(1, 1751)
            f = interp1d(ramp_function[:, 0], ramp_function[:, 1], kind='linear', fill_value='extrapolate')
            ramp_function_interp = f(t_interp)
            pH_sensor_error = ramp_function_interp[k - 1]
            u.Fault_ref = 1

        # builds the error history. Samples 1 and 2 are calculated separately
        # because there is only instance of the error available
        pH_sp = self.ctrl_flags.pH_sp
        if k == 1 or k == 2:
            ph_err = pH_sp - (-math.log(x.pH.y[0]) / math.log(10)) + pH_sensor_error
            ph_err1 = pH_sp - (-math.log(x.pH.y[0]) / math.log(10)) + pH_sensor_error
        else:
            ph_err = pH_sp - (-math.log(x.pH.y[k - 2]) / math.log(10)) + pH_sensor_error
            ph_err1 = - (-math.log(x.pH.y[k - 3]) / math.log(10)) + pH_sensor_error

        # builds the pH history of the current and previous two samples
        if k == 1 or k == 2:
            ph = -math.log(x.pH.y[0]) / math.log(10)
            ph1 = -math.log(x.pH.y[0]) / math.log(10)
            ph2 = -math.log(x.pH.y[0]) / math.log(10)
        elif k == 3:
            ph = -math.log(x.pH.y[1]) / math.log(10)
            ph1 = -math.log(x.pH.y[0]) / math.log(10)
            ph2 = -math.log(x.pH.y[0]) / math.log(10)
        else:
            ph = -math.log(x.pH.y[k - 2]) / math.log(10)
            ph1 = -math.log(x.pH.y[k - 3]) / math.log(10)
            ph2 = -math.log(x.pH.y[k - 4]) / math.log(10)

        # pH has decreased under 0.05 from set-point, add some base solution
        if ph_err >= -0.05:
            ph_on_off = 1
            if k == 1:
                Fb = pid_controller(x.Fb.y[0], ph_err, ph_err1, ph, ph1, ph2, 0, 225, 8e-2, 4.0000e-05, 8, STEP_IN_HOURS)
            else:
                Fb = pid_controller(x.Fb.y[k - 2], ph_err, ph_err1, ph, ph1, ph2, 0, 225, 8e-2, 4.0000e-05, 8, STEP_IN_HOURS)
            Fa = 0
        elif ph_err <= -0.05:
            ph_on_off = 1
            if k == 1:
                Fa = pid_controller(x.Fa.y[0], ph_err, ph_err1, ph, ph1, ph2, 0, 225, 8e-2, 12.5, 0.125, STEP_IN_HOURS)
                Fb = 0
            else:
                Fa = pid_controller(x.Fa.y[k - 2], ph_err, ph_err1, ph, ph1, ph2, 0, 225, 8e-2, 12.5, 0.125, STEP_IN_HOURS)
                Fb = x.Fb.y[k - 2] * 0.5
        else:
            ph_on_off = 0
            Fb = 0
            Fa = 0

        # Temperature controller
        T_sensor_error = 0
        if self.ctrl_flags.Faults == 7:
            T_sensor_error = 0.4
            ramp_function = [[0, 0],
                             [200, 0],
                             [800, T_sensor_error],
                             [1750, T_sensor_error]]
            ramp_function = np.array(ramp_function)
            t_interp = np.arange(1, 1751)
            f = interp1d(ramp_function[:, 0], ramp_function[:, 1], kind='linear', fill_value='extrapolate')
            ramp_function_interp = f(t_interp)
            T_sensor_error = ramp_function_interp[k - 1]
            u.Fault_ref = 1

        # builds the error history.  Samples 1 and 2 are calculated separately
        # because there is only instance of the error available
        T_sp = self.ctrl_flags.T_sp
        if k == 1 or k == 2:
            temp_err = T_sp - x.T.y[0] + T_sensor_error
            temp_err1 = T_sp - x.T.y[0] + T_sensor_error
        else:
            temp_err = T_sp - x.T.y[k - 2] + T_sensor_error
            temp_err1 = T_sp - x.T.y[k - 3] + T_sensor_error

        # builds the temperature history of current and previous two samples.
        if k == 1 or k == 2:
            temp = x.T.y[0]
            temp1 = x.T.y[0]
            temp2 = x.T.y[0]
        elif k == 3:
            temp = x.T.y[1]
            temp1 = x.T.y[0]
            temp2 = x.T.y[0]
        else:
            temp = x.T.y[k - 2]
            temp1 = x.T.y[k - 3]
            temp2 = x.T.y[k - 4]

        # Threshold for heating. Heating is activated only if the temperature drop
        # is more than 1 degree celsius
        if temp_err <= 0.05:
            temp_on_off = 0
            if k == 1:
                Fc = pid_controller(x.Fc.y[0], temp_err, temp_err1, temp, temp1, temp2, 0, 1.5e3, -300, 1.6, 0.005, STEP_IN_HOURS)
                Fh = 0
            else:
                Fc = pid_controller(x.Fc.y[k - 2], temp_err, temp_err1, temp, temp1, temp2, 0, 1.5e3, -300, 1.6, 0.005, STEP_IN_HOURS)
                Fh = x.Fh.y[k - 2] * 0.1
        else:
            temp_on_off = 1
            if k == 1:
                Fh = pid_controller(x.Fc.y[0], temp_err, temp_err1, temp, temp1, temp2, 0, 1.5e3, 50, 0.050, 1, STEP_IN_HOURS)
                Fc = 0
            else:
                Fh = pid_controller(x.Fc.y[k - 2], temp_err, temp_err1, temp, temp1, temp2, 0, 1.5e3, 50, 0.050, 1, STEP_IN_HOURS)
                Fc = x.Fc.y[k - 2] * 0.3
        Fc = 1e-4 if Fc < 1e-4 else Fc
        Fh = 1e-4 if Fh < 1e-4 else Fh

        # Sequential Batch control strategy
        # If Sequential Batch Control (SBC) = 1, operator controlled
        if self.ctrl_flags.SBC == 1:
            Foil = self.xinterp.Foil.y[k - 1]
            Fdischarge = self.xinterp.Fdischarge_cal.y[k - 1]
            pressure = self.xinterp.pressure.y[k - 1]
            Fpaa = self.xinterp.Fpaa.y[k - 1]
            Fw = self.xinterp.Fw.y[k - 1]
            viscosity = self.xinterp.viscosity.y[k - 1]
            Fg = self.xinterp.Fg.y[k - 1]
            Fs = self.xinterp.Fs.y[k - 1]

        # SBC - Fs
        if self.ctrl_flags.SBC == 0:
            viscosity = 4
            Fs = Fs_k
            Foil = Foil_k
            Fg = Fg_k
            pressure = pressure_k
            discharge = -discharge_k
            Fw = Fw_k
            Fpaa = Fpaa_k

            if self.ctrl_flags.PRBS == 1:
                if k > 500 and np.remainder(k, 100) == 0:
                    random_number = np.random.randint(1, 4)
                    noise_factor = 15
                    if random_number == 1:
                        random_noise = 0
                    elif random_number == 2:
                        random_noise = noise_factor
                    else:
                        random_noise = -noise_factor
                    x.PRBS_noise_addition[k - 1] = random_noise
                Fs = x.Fs.y[k - 2] if k > 475 else Fs
                if k > 500 and np.remainder(k, 100) == 0:
                    Fs = x.Fs.y[k - 2] + x.PRBS_noise_addition[-1]
            else:
                x.PRBS_noise_addition[k - 1] = 0

            # Add PRBS to substrate flow rate  (Fpaa)
            if self.ctrl_flags.PRBS == 1:
                if k > 500 and np.remainder(k, 100) == 0:
                    random_number = np.random.randint(1, 4)
                    noise_factor = 1
                    if random_number == 1:
                        random_noise = 0
                    elif random_number == 2:
                        random_noise = noise_factor
                    else:
                        random_noise = -noise_factor
                    x.PRBS_noise_addition[k - 1] = random_noise
                Fpaa = x.Fpaa.y[k - 2] if k > 475 else Fpaa
                if k > 500 and np.remainder(k, 100) == 0:
                    Fpaa = x.Fpaa.y[k - 2] + x.PRBS_noise_addition[-1]
            else:
                x.PRBS_noise_addition[k - 1] = 0
            self.xinterp.NH3_shots.y[k - 1] = 0

        # Process faults
        # 0 - No Faults
        # 1 - Aeration rate fault
        # 2 - Vessel back pressure  fault
        # 3 - Substrate feed rate fault
        # 4 - Base flow-rate fault
        # 5 - Coolant flow-rate fault
        # 6 - All of the above faults

        # Aeration fault
        if self.ctrl_flags.Faults == 1 or self.ctrl_flags.Faults == 6:
            if 100 <= k <= 120:
                Fg = 20
                u.Fault_ref = 1
            if 500 <= k <= 550:
                Fg = 20
                u.Fault_ref = 1

        # Pressure fault
        if self.ctrl_flags.Faults == 2 or self.ctrl_flags.Faults == 6:
            if 500 <= k <= 520:
                pressure = 2
                u.Fault_ref = 1
            if 1000 <= k <= 1200:
                pressure = 2
                u.Fault_ref = 1

        # Substrate feed fault
        if self.ctrl_flags.Faults == 3 or self.ctrl_flags.Faults == 6:
            if 100 <= k <= 150:
                Fs = 2
                u.Fault_ref = 1
            if 380 <= k <= 460:
                Fs = 20
                u.Fault_ref = 1
            if 1000 <= k <= 1070:
                Fs = 20
                u.Fault_ref = 1

        # Base flow-rate  fault
        if self.ctrl_flags.Faults == 4 or self.ctrl_flags.Faults == 6:
            if 400 <= k <= 420:
                Fb = 5
                u.Fault_ref = 1
            if 700 <= k <= 800:
                Fb = 10
                u.Fault_ref = 1

        # Coolant water flow-rate fault
        if self.ctrl_flags.Faults == 5 or self.ctrl_flags.Faults == 6:
            if 350 <= k <= 450:
                Fc = 2
                u.Fault_ref = 1
            if 1200 <= k <= 1350:
                Fc = 10
                u.Fault_ref = 1

        # Bulidng PID controller for PAA
        # builds the error history.  Samples 1 and 2 are calculated separately
        # because there is only instance of the error available
        if self.ctrl_flags.Raman_spec == 2:
            PAA_sp = 1200
            if k == 1 or k == 2:
                PAA_err = PAA_sp - x.PAA.y[0]
                PAA_err1 = PAA_sp - x.PAA.y[0]
            else:
                PAA_err = PAA_sp - x.PAA.y[k - 2]
                PAA_err1 = PAA_sp - x.PAA.y[k - 3]

            # builds the temperature history of current and previous two samples
            if k * STEP_IN_HOURS >= 10:
                if k == 1 or k == 2:
                    temp = x.PAA_pred.y[0]
                    temp1 = x.PAA_pred.y[0]
                    temp2 = x.PAA_pred.y[0]
                elif k == 3:
                    temp = x.PAA_pred.y[1]
                    temp1 = x.PAA_pred.y[0]
                    temp2 = x.PAA_pred.y[0]
                else:
                    temp = x.PAA_pred.y[k - 3]
                    temp1 = x.PAA_pred.y[k - 4]
                    temp2 = x.PAA_pred.y[k - 5]

                if k == 1:
                    Fpaa = pid_controller(x.Fpaa.y[0], PAA_err, PAA_err1, temp, temp1, temp2, 0, 150, 0.1, 0.50, 0, STEP_IN_HOURS)
                else:
                    Fpaa = pid_controller(x.Fpaa.y[k - 2], PAA_err, PAA_err1, temp, temp1, temp2, 0, 150, 0.1, 0.50, 0, STEP_IN_HOURS)

        # Controller vector
        u.Fg = Fg
        u.RPM = 100
        u.Fs = Fs
        u.Fa = Fa
        u.Fb = Fb
        u.Fc = Fc
        u.Fh = Fh
        u.d1 = ph_on_off
        u.tfl = temp_on_off
        u.Fw = Fw
        u.pressure = pressure
        u.viscosity = viscosity
        u.discharge = discharge
        u.Fpaa = Fpaa
        u.Foil = Foil
        u.NH3_shots = self.xinterp.NH3_shots.y[k - 1]

        return u, x

    def raman_sim(self, k, x):
        # Building history of Raman Spectra
        Intensity_shift1 = np.ones((WAVENUMBER_LENGTH, 1), dtype=int)

        Intensity_shift1[:, 0] = np.exp((np.arange(WAVENUMBER_LENGTH) + 1) / 1100) - 0.5

        a = -0.000178143846614472
        b = 1.05644816081515
        c = -0.00681439987249108
        d = -0.02
        Product_S = x.P.y[k - 1] / 40
        Biomass_S = x.X.y[k - 1] / 40
        Viscosity_S = x.Viscosity.y[k - 1] / 100
        Time_S = k / NUM_STEPS
        Intensity_increase1 = a * Biomass_S + b * Product_S + c * Viscosity_S + d * Time_S
        scaling_factor = 370000
        Gluc_increase = 1714.2857142857142
        PAA_increase = 1700
        Prod_increase = 100000

        # Loading in the reference Raman Spectral file
        New_Spectra = Intensity_increase1 * scaling_factor * Intensity_shift1 + np.array([RAMAN_SPECTRA]).T
        x.Raman_Spec.Intensity[k - 1, :] = np.squeeze(New_Spectra).tolist()

        random_noise = [50] * WAVENUMBER_LENGTH
        random_number = np.random.randint(-1, 2, size=(WAVENUMBER_LENGTH, 1))
        random_noise = np.multiply(random_noise, random_number.T)[0]

        random_noise_summed = np.cumsum(random_noise)
        random_noise_summed_smooth = smooth(random_noise_summed, 25)

        New_Spectra_noise = New_Spectra + 10 * np.array([random_noise_summed_smooth]).T

        x.Raman_Spec.Intensity[k - 1, :] = np.squeeze(New_Spectra_noise).tolist()

        # Aim 3. Creating the bell curve response for Glucose
        Glucose_raw_peaks_G_peaka = np.zeros((WAVENUMBER_LENGTH, 1), dtype=float)
        Glucose_raw_peaks_G_peakb = np.zeros((WAVENUMBER_LENGTH, 1), dtype=float)
        Glucose_raw_peaks_G_peakc = np.zeros((WAVENUMBER_LENGTH, 1), dtype=float)
        PAA_raw_peaks_G_peaka = np.zeros((WAVENUMBER_LENGTH, 1), dtype=float)
        PAA_raw_peaks_G_peakb = np.zeros((WAVENUMBER_LENGTH, 1), dtype=float)
        Product_raw_peaka = np.zeros((WAVENUMBER_LENGTH, 1), dtype=float)
        Product_raw_peakb = np.zeros((WAVENUMBER_LENGTH, 1), dtype=float)

        # Glucose peaks
        # Peak A
        Glucose_raw_peaks_G_peaka[78: 359, 0] = 0.011398350868612364 * np.exp(-0.0004081632653061224 * np.arange(-140, 141) ** 2)
        # Peak B
        Glucose_raw_peaks_G_peakb[598: 679, 0] = 0.009277727451196111 * np.exp(-0.005 * np.arange(-40, 41) ** 2)
        # Peak C
        Glucose_raw_peaks_G_peakc[852: 1253, 0] = 0.007978845608028654 * np.exp(-0.0002 * np.arange(-200, 201) ** 2)
        # PAA  peaks
        # Peak A
        PAA_raw_peaks_G_peaka[298: 539, 0] = 0.01329807601338109 * np.exp(-0.0005555555555555556 * np.arange(-120, 121) ** 2)
        # Peak B
        PAA_raw_peaks_G_peakb[808: 869, 0] = 0.01237030326826148 * np.exp(-0.008888888888888889 * np.arange(-30, 31) ** 2)
        # Adding in  Peak aPen G Peak
        Product_raw_peaka[679: 920, 0] = 0.02659615202676218 * np.exp(-0.0022222222222222222 * np.arange(-120, 121) ** 2)
        # Adding in  Peak b for Pen G Peak
        Product_raw_peakb[299: 2100, 0] = 0.02659615202676218 * np.exp(-0.0022222222222222222 * np.arange(-900, 901) ** 2)

        total_peaks_G = Glucose_raw_peaks_G_peaka + Glucose_raw_peaks_G_peakb + Glucose_raw_peaks_G_peakc
        total_peaks_PAA = PAA_raw_peaks_G_peaka + PAA_raw_peaks_G_peakb
        total_peaks_P = Product_raw_peakb + Product_raw_peaka
        K_G = 0.005

        Substrate_raman = x.S.y[k - 1]
        PAA_raman = x.PAA.y[k - 1]

        term1 = total_peaks_G * Gluc_increase * Substrate_raman / (K_G + Substrate_raman)
        term2 = total_peaks_PAA * PAA_increase * PAA_raman
        term3 = total_peaks_P * Prod_increase * x.P.y[k - 1]

        x.Raman_Spec.Intensity[k - 1, :] = np.squeeze(New_Spectra_noise + term1 + term2 + term3).tolist()

        return x

    def get_batches(self, random_seed=0, include_raman=False):
        """
        Generate batch data in pandas dataframes.
        """
        self.random_seed_ref = random_seed

        t = time.time()
        done = False
        observation, batch_data = self.reset()
        k_timestep, batch_yield, yield_pre = 0, 0, 0

        self.yield_pre = 0
        while not done:
            k_timestep += 1
            # Get action from recipe agent based on time
            Fs, Foil, Fg, pressure, discharge, Fw, Fpaa = self.recipe.get_values_at(k_timestep * STEP_IN_MINUTES)

            # Run and get the reward
            # observation is a class which contains all the variables, e.g. observation.Fs.y[k], observation.Fs.t[k]
            # are the Fs value and corresponding time at k
            observation, batch_data, reward, done = self.step(k_timestep,
                                                              batch_data,
                                                              Fs, Foil, Fg, pressure, discharge, Fw, Fpaa)
            batch_yield += reward

        print(f"=== Yield {round(batch_yield, 6)} Kg in {round(time.time() - t, 3)}s")
        return get_dataframe(batch_data, include_raman)
