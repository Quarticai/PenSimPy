import numpy as np
from create_batch import create_batch
from fctrl_indpensim import fctrl_indpensim
from scipy.integrate import odeint
import math
from indpensim_ode_py import indpensim_ode_py


def indpensim(xd, x0, h, T, solv, param_list, ctrl_flags):
    """
    Simulate the fermentation process by solving ODE
    :param xd:
    :param x0:
    :param h:
    :param T:
    :param solv:
    :param param_list:
    :param ctrl_flags:
    :return:
    """
    # simulation timing init
    N = int(T / h)
    h_ode = h / 20
    t = np.arange(0, T + h, h)
    # creates batch structured
    x = create_batch(h, T)

    # User control inputs
    # converts from pH to H+ conc.
    x0.pH = 10 ** (-x0.pH)

    # Recipe_Fs = [15, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 800, 1750]
    Recipe_Fs_sp = [8, 15, 30, 75, 150, 30, 37, 43, 47, 51, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80]
    # Recipe_Fs_sp = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    # main loop
    for k in range(1, N + 1):
        # fills the batch with just the initial conditions so the control system
        # can provide the first input. These will be overwritten after
        # the ODEs are integrated.
        if k == 1:
            x.S.y[0] = x0.S
            x.DO2.y[0] = x0.DO2
            x.X.y[0] = x0.X
            x.P.y[0] = x0.P
            x.V.y[0] = x0.V
            x.CO2outgas.y[0] = x0.CO2outgas
            x.pH.y[0] = x0.pH
            x.T.y[0] = x0.T

        # gets MVs
        # print(f"=== k: {k}")
        # if k == 20:
        #     Recipe_Fs_sp_new = input()
        #     Recipe_Fs_sp_new = [int(ele) for ele in Recipe_Fs_sp_new.split(',')]
        #     Recipe_Fs_sp = Recipe_Fs_sp_new
        # print(f"=== Recipe_Fs_sp: {Recipe_Fs_sp}")

        u, x = fctrl_indpensim(x, xd, k, h, T, ctrl_flags, Recipe_Fs_sp)

        # builds initial conditions and control vectors specific to
        # indpensim_ode using ode45
        if k == 1:
            x00 = [x0.S,
                   x0.DO2,
                   x0.O2,
                   x0.P,
                   x0.V,
                   x0.Wt,
                   x0.pH,
                   x0.T,
                   0,
                   4,
                   x0.Culture_age,
                   x0.a0,
                   x0.a1,
                   x0.a3,
                   x0.a4,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   x0.CO2outgas,
                   0,
                   x0.PAA,
                   x0.NH3,
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
        distMuP = xd.distMuP.y[k - 1]
        distMuX = xd.distMuX.y[k - 1]
        distcs = xd.distcs.y[k - 1]
        distcoil = xd.distcoil.y[k - 1]
        distabc = xd.distabc.y[k - 1]
        distPAA = xd.distPAA.y[k - 1]
        distTcin = xd.distTcin.y[k - 1]
        distO_2in = xd.distO_2in.y[k - 1]

        u00 = [ctrl_flags.Inhib,
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
               ctrl_flags.Dis,
               distMuP,
               distMuX,
               distcs,
               distcoil,
               distabc,
               distPAA,
               distTcin,
               distO_2in,
               ctrl_flags.Vis]

        # To account for inability of growth rates of biomass and penicillin to
        # return to normal after continuous periods of suboptimal pH and temperature conditions
        # If the Temperature or pH results is off set-point for k> 100 mu_p(max) is reduced to current value
        if ctrl_flags.Inhib == 1 or ctrl_flags.Inhib == 2:
            if k > 65:
                a1 = np.diff(x.mu_X_calc.y[k - 66:k - 1])
                a2 = [1 if x < 0 else 0 for x in a1]
                if sum(a2) >= 63:
                    param_list[1] = x.mu_X_calc.y[k - 2] * 5

        # Solver selection and calling indpensim_ode
        if solv == 2:
            t_start = t[k - 1]
            t_end = t[k]
            t_span = np.arange(t_start, t_end + h_ode, h_ode).tolist()

            par = param_list.copy()
            par.extend(u00)

            y_sol = odeint(indpensim_ode_py, x00, t_span, tfirst=True, args=(par,))

        # """
        # Defining minimum value for all variables for numerical stability
        for i in range(1, 32):
            if y_sol[-1][i - 1] <= 0:
                y_sol[-1][i - 1] = 0.001

        # Saving all manipulated variables
        x.Fg.t[k - 1] = t_span[-1]
        x.Fg.y[k - 1] = u.Fg
        x.RPM.t[k - 1] = t_span[-1]
        x.RPM.y[k - 1] = u.RPM
        x.Fpaa.t[k - 1] = t_span[-1]
        x.Fpaa.y[k - 1] = u.Fpaa
        x.Fs.t[k - 1] = t_span[-1]
        x.Fs.y[k - 1] = u.Fs
        x.Fa.t[k - 1] = t_span[-1]
        x.Fa.y[k - 1] = u.Fa
        x.Fb.t[k - 1] = t_span[-1]
        x.Fb.y[k - 1] = u.Fb
        x.Fc.t[k - 1] = t_span[-1]
        x.Fc.y[k - 1] = u.Fc
        x.Foil.t[k - 1] = t_span[-1]
        x.Foil.y[k - 1] = u.Foil
        x.Fh.t[k - 1] = t_span[-1]
        x.Fh.y[k - 1] = u.Fh
        x.Fw.t[k - 1] = t_span[-1]
        x.Fw.y[k - 1] = u.Fw
        x.pressure.t[k - 1] = t_span[-1]
        x.pressure.y[k - 1] = u.pressure
        x.Fremoved.t[k - 1] = t_span[-1]
        x.Fremoved.y[k - 1] = u.Fremoved

        # Saving all the  IndPenSim states
        x.S.y[k - 1] = y_sol[-1][0]
        x.S.t[k - 1] = t_span[-1]
        x.DO2.y[k - 1] = y_sol[-1][1]

        # Required for numerical stability
        x.DO2.y[k - 1] = 1 if x.DO2.y[k - 1] < 2 else x.DO2.y[k - 1]

        x.DO2.t[k - 1] = t_span[-1]
        x.O2.y[k - 1] = y_sol[-1][2]
        x.O2.t[k - 1] = t_span[-1]
        x.P.y[k - 1] = y_sol[-1][3]
        x.P.t[k - 1] = t_span[-1]
        x.V.y[k - 1] = y_sol[-1][4]
        x.V.t[k - 1] = t_span[-1]
        x.Wt.y[k - 1] = y_sol[-1][5]
        x.Wt.t[k - 1] = t_span[-1]
        x.pH.y[k - 1] = y_sol[-1][6]
        x.pH.t[k - 1] = t_span[-1]
        x.T.y[k - 1] = y_sol[-1][7]
        x.T.t[k - 1] = t_span[-1]
        x.Q.y[k - 1] = y_sol[-1][8]
        x.Q.t[k - 1] = t_span[-1]
        x.Viscosity.y[k - 1] = y_sol[-1][9]
        x.Viscosity.t[k - 1] = t_span[-1]
        x.Culture_age.y[k - 1] = y_sol[-1][10]
        x.Culture_age.t[k - 1] = t_span[-1]
        x.a0.y[k - 1] = y_sol[-1][11]
        x.a0.t[k - 1] = t_span[-1]
        x.a1.y[k - 1] = y_sol[-1][12]
        x.a1.t[k - 1] = t_span[-1]
        x.a3.y[k - 1] = y_sol[-1][13]
        x.a3.t[k - 1] = t_span[-1]
        x.a4.y[k - 1] = y_sol[-1][14]
        x.a4.t[k - 1] = t_span[-1]
        x.n0.y[k - 1] = y_sol[-1][15]
        x.n0.t[k - 1] = t_span[-1]
        x.n1.y[k - 1] = y_sol[-1][16]
        x.n1.t[k - 1] = t_span[-1]
        x.n2.y[k - 1] = y_sol[-1][17]
        x.n2.t[k - 1] = t_span[-1]
        x.n3.y[k - 1] = y_sol[-1][18]
        x.n3.t[k - 1] = t_span[-1]
        x.n4.y[k - 1] = y_sol[-1][19]
        x.n4.t[k - 1] = t_span[-1]
        x.n5.y[k - 1] = y_sol[-1][20]
        x.n5.t[k - 1] = t_span[-1]
        x.n6.y[k - 1] = y_sol[-1][21]
        x.n6.t[k - 1] = t_span[-1]
        x.n7.y[k - 1] = y_sol[-1][22]
        x.n7.t[k - 1] = t_span[-1]
        x.n8.y[k - 1] = y_sol[-1][23]
        x.n8.t[k - 1] = t_span[-1]
        x.n9.y[k - 1] = y_sol[-1][24]
        x.n9.t[k - 1] = t_span[-1]
        x.nm.y[k - 1] = y_sol[-1][25]
        x.nm.t[k - 1] = t_span[-1]
        x.phi0.y[k - 1] = y_sol[-1][26]
        x.phi0.t[k - 1] = t_span[-1]
        x.CO2outgas.y[k - 1] = y_sol[-1][27]
        x.CO2outgas.t[k - 1] = t_span[-1]
        x.CO2_d.t[k - 1] = t_span[-1]
        x.CO2_d.y[k - 1] = y_sol[-1][28]
        x.PAA.y[k - 1] = y_sol[-1][29]
        x.PAA.t[k - 1] = t_span[-1]
        x.NH3.y[k - 1] = y_sol[-1][30]
        x.NH3.t[k - 1] = t_span[-1]
        x.mu_P_calc.y[k - 1] = y_sol[-1][31]
        x.mu_P_calc.t[k - 1] = t_span[-1]
        x.mu_X_calc.y[k - 1] = y_sol[-1][32]
        x.mu_X_calc.t[k - 1] = t_span[-1]
        x.X.y[k - 1] = x.a0.y[k - 1] + x.a1.y[k - 1] + x.a3.y[k - 1] + x.a4.y[k - 1]
        x.X.t[k - 1] = t_span[-1]
        x.Fault_ref.y[k - 1] = u.Fault_ref
        x.Fault_ref.t[k - 1] = t_span[-1]
        x.Control_ref.y[k - 1] = ctrl_flags.PRBS
        x.Control_ref.t[k - 1] = ctrl_flags.Batch_Num
        x.PAT_ref.y[k - 1] = ctrl_flags.Raman_spec
        x.PAT_ref.t[k - 1] = ctrl_flags.Batch_Num
        x.Batch_ref.t[k - 1] = ctrl_flags.Batch_Num
        x.Batch_ref.y[k - 1] = ctrl_flags.Batch_Num

        # oxygen in air
        O2_in = 0.204

        # Calculating the OUR/ CER
        x.OUR.y[k - 1] = (32 * x.Fg.y[k - 1] / 22.4) * \
                         (O2_in - x.O2.y[k - 1] * (0.7902 / (1 - x.O2.y[k - 1] - x.CO2outgas.y[k - 1] / 100)))
        x.OUR.t[k - 1] = t_span[-1]

        # Calculating the CER
        x.CER.y[k - 1] = (44 * x.Fg.y[k - 1] / 22.4) * ((0.65 * x.CO2outgas.y[k - 1] / 100) *
                                                        (0.7902 / (1 - O2_in - x.CO2outgas.y[k - 1] / 100) - 0.0330))
        x.CER.t[k - 1] = t_span[-1]

        # Adding in Raman Spectra
        # Todo

        # Off-line measurements recorded
        if np.remainder(t_span[-1], ctrl_flags.Off_line_m) <= 1e-3 or abs(t_span[-1] - 1) <= 1e-3 or abs(t_span[-1] - T) <= 1e-3 :
            delay = ctrl_flags.Off_line_delay
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

        # convert to pH from H+ concentration
    for k in range(0, len(x.pH.y)):
        if x.pH.y[k] != 0:
            x.pH.y[k] = -math.log(x.pH.y[k]) / math.log(10)
        x.Q.y[k] = x.Q.y[k] / 1000

    return x
