import math
import numpy as np
from numba import njit, jit


# @njit(nopython=True)
def indpensim_ode_py(t, y, par):
    """
    ODE for penicillin proecss
    :param t: time span
    :param y: initial condition
    :param par: inputs + params
    :return:
    """
    mu_p = par[0]
    mux_max = par[1]
    ratio_mu_e_mu_b = par[2]
    P_std_dev = par[3]
    mean_P = par[4]
    mu_v = par[5]
    mu_a = par[6]
    mu_diff = par[7]
    beta_1 = par[8]
    K_b = par[9]
    K_diff = par[10]
    K_diff_L = par[11]
    K_e = par[12]
    K_v = par[13]
    delta_r = par[14]
    k_v = par[15]
    D = par[16]
    rho_a0 = par[17]
    rho_d = par[18]
    mu_h = par[19]
    r_0 = par[20]
    delta_0 = par[21]

    # Process related parameters
    Y_sX = par[22]
    Y_sP = par[23]
    m_s = par[24]
    c_oil = par[25]
    c_s = par[26]
    Y_O2_X = par[27]
    Y_O2_P = par[28]
    m_O2_X = par[29]
    alpha_kla = par[30]
    a = par[31]
    b = par[32]
    c = par[33]
    d = par[34]
    Henrys_c = par[35]
    n_imp = par[36]
    r = par[37]
    r_imp = par[38]
    Po = par[39]
    epsilon = par[40]
    g = par[41]
    R = par[42]
    X_crit_DO2 = par[43]
    P_crit_DO2 = par[44]
    A_inhib = par[45]
    Tf = par[46]
    Tw = par[47]
    Tcin = par[48]
    Th = par[49]
    Tair = par[50]
    C_ps = par[51]
    C_pw = par[52]
    dealta_H_evap = par[53]
    U_jacket = par[54]
    A_c = par[55]
    Eg = par[56]
    Ed = par[57]
    k_g = par[58]
    k_d = par[59]
    Y_QX = par[60]
    abc = par[61]
    gamma1 = par[62]
    gamma2 = par[63]
    m_ph = par[64]
    K1 = par[65]
    K2 = par[66]
    N_conc_oil = par[67]
    N_conc_paa = par[68]
    N_conc_shot = par[69]
    Y_NX = par[70]
    Y_NP = par[71]
    m_N = par[72]
    X_crit_N = par[73]
    PAA_c = par[74]
    Y_PAA_P = par[75]
    Y_PAA_X = par[76]
    m_PAA = par[77]
    X_crit_PAA = par[78]
    P_crit_PAA = par[79]
    B_1 = par[80]
    B_2 = par[81]
    B_3 = par[82]
    B_4 = par[83]
    B_5 = par[84]
    delta_c_0 = par[85]
    k3 = par[86]
    k1 = par[87]
    k2 = par[88]
    t1 = par[89]
    t2 = par[90]
    q_co2 = par[91]
    X_crit_CO2 = par[92]
    alpha_evp = par[93]
    beta_T = par[94]
    pho_g = par[95]
    pho_oil = par[96]
    pho_w = par[97]
    pho_paa = par[98]
    O_2_in = par[99]
    N2_in = par[100]
    C_CO2_in = par[101]
    Tv = par[102]
    T0 = par[103]
    alpha_1 = par[104]

    # process inputs
    inhib_flag = par[0 + 105]
    Fs = par[1 + 105]
    Fg = (par[2 + 105] / 60)
    RPM = par[3 + 105]
    Fc = par[4 + 105]
    Fh = par[5 + 105]
    Fb = par[6 + 105]
    Fa = par[7 + 105]
    step1 = par[8 + 105]
    Fw = par[9 + 105]
    Fw = 0 if Fw < 0 else Fw
    pressure = par[10 + 105]
    # Viscosity flag
    if par[25 + 105] == 0:
        viscosity = y[9]
    else:
        viscosity = par[11 + 105]

    F_discharge = par[12 + 105]
    Fpaa = par[13 + 105]
    Foil = par[14 + 105]
    NH3_shots = par[15 + 105]
    dist_flag = par[16 + 105]
    distMuP = par[17 + 105]
    distMuX = par[18 + 105]
    distsc = par[19 + 105]
    distcoil = par[20 + 105]
    distabc = par[21 + 105]
    distPAA = par[22 + 105]
    distTcin = par[23 + 105]
    distO_2_in = par[24 + 105]
    pho_b = (1100 + y[3] + y[11] + y[12] + y[13] + y[14])

    if dist_flag == 1:
        mu_p = mu_p + distMuP
        mux_max = mux_max + distMuX
        c_s = c_s + distsc
        c_oil = c_oil + distcoil
        abc = abc + distabc
        PAA_c = PAA_c + distPAA
        Tcin = Tcin + distTcin
        O_2_in = O_2_in + distO_2_in

    # Process parameters
    # Adding in age-dependant term
    A_t1 = ((y[10]) / (y[11] + y[12] + y[13] + y[14]))

    # Variables
    s = y[0]
    a_1 = y[12]
    a_0 = y[11]
    a_3 = y[13]
    total_X = y[11] + y[12] + y[13] + y[14]  # Total Biomass

    # Calculating liquid height in vessel
    h_b = (y[4] / 1000) / (math.pi * (r ** 2))
    h_b = h_b * (1 - epsilon)

    # Calculating log mean pressure of vessel
    pressure_bottom = 1 + pressure + ((pho_b * h_b) * 9.81 * 1e-5)
    pressure_top = 1 + pressure
    log_mean_pressure = (pressure_bottom - pressure_top) / (math.log(pressure_bottom / pressure_top))
    total_pressure = log_mean_pressure

    # Ensuring minimum value for viscosity
    viscosity = 1 if viscosity < 4 else viscosity
    DOstar_tp = total_pressure * O_2_in / Henrys_c

    # Inhibition flags
    if inhib_flag == 0:
        pH_inhib = 1
        NH3_inhib = 1
        T_inhib = 1
        mu_h = 0.003
        DO_2_inhib_X = 1
        DO_2_inhib_P = 1
        CO2_inhib = 1
        PAA_inhib_X = 1
        PAA_inhib_P = 1

    if inhib_flag == 1:
        pH_inhib = (1 / (1 + (y[6] / K1) + (K2 / y[6])))
        NH3_inhib = 1
        T_inhib = (k_g * math.exp(-(Eg / (R * y[7]))) - k_d * math.exp(-(Ed / (R * y[7])))) * 0 + 1
        CO2_inhib = 1
        DO_2_inhib_X = 0.5 * (1 - math.tanh(A_inhib * (X_crit_DO2 * ((total_pressure * O_2_in) / Henrys_c) - y[1])))
        DO_2_inhib_P = 0.5 * (1 - math.tanh(A_inhib * (P_crit_DO2 * ((total_pressure * O_2_in) / Henrys_c) - y[1])))
        PAA_inhib_X = 1
        PAA_inhib_P = 1
        pH = -math.log10(y[6])
        k4 = math.exp((B_1 + B_2 * pH + B_3 * y[7] + B_4 * (pH ** 2)) + B_5 * (y[7] ** 2))
        mu_h = k4

    if inhib_flag == 2:
        pH_inhib = 1 / (1 + (y[6] / K1) + (K2 / y[6]))
        NH3_inhib = 0.5 * (1 - math.tanh(A_inhib * (X_crit_N - y[30])))
        T_inhib = k_g * math.exp(-(Eg / (R * y[7]))) - k_d * math.exp(-(Ed / (R * y[7])))
        CO2_inhib = 0.5 * (1 + math.tanh(A_inhib * (X_crit_CO2 - y[28] * 1000)))
        DO_2_inhib_X = 0.5 * (1 - math.tanh(A_inhib * (X_crit_DO2 * ((total_pressure * O_2_in) / Henrys_c) - y[1])))
        DO_2_inhib_P = 0.5 * (1 - math.tanh(A_inhib * (P_crit_DO2 * ((total_pressure * O_2_in) / Henrys_c) - y[1])))
        PAA_inhib_X = 0.5 * (1 + (math.tanh((X_crit_PAA - y[29]))))
        PAA_inhib_P = 0.5 * (1 + (math.tanh((-P_crit_PAA + y[29]))))
        pH = -math.log10(y[6])
        k4 = math.exp((B_1 + B_2 * pH + B_3 * y[7] + B_4 * (pH ** 2)) + B_5 * (y[7] ** 2))
        mu_h = k4

    # Main rate equations for kinetic expressions
    # Penicillin inhibition curve
    P_inhib = 2.5 * P_std_dev * (
            (P_std_dev * np.sqrt(2 * math.pi)) ** -1 * math.exp(-0.5 * ((s - mean_P) / P_std_dev) ** 2))

    # Specific growth rates of biomass regions with inhibition effect
    mu_a0 = ratio_mu_e_mu_b * mux_max * pH_inhib * NH3_inhib * T_inhib * DO_2_inhib_X * CO2_inhib * PAA_inhib_X

    # Rate constant for Branching A0
    mu_e = mux_max * pH_inhib * NH3_inhib * T_inhib * DO_2_inhib_X * CO2_inhib * PAA_inhib_X

    # Rate constant for extension A1
    K_diff = par[10] - (A_t1 * beta_1)
    if K_diff < K_diff_L:
        K_diff = K_diff_L

    # Growing A_0 region
    r_b0 = mu_a0 * a_1 * s / (K_b + s)
    r_sb0 = Y_sX * r_b0

    # Non-growing regions A_1 region
    r_e1 = (mu_e * a_0 * s) / (K_e + s)
    r_se1 = Y_sX * r_e1

    # Differentiation (A_0 -> A_1)
    r_d1 = mu_diff * a_0 / (K_diff + s)
    r_m0 = m_s * a_0 / (K_diff + s)

    n = 16
    phi = [0] * 10
    phi[0] = y[26]

    for k in range(2, 11):
        r_mean = 1.5e-4 + (k - 2) * delta_r
        phi[k - 1] = ((4 * math.pi * r_mean ** 3) / 3) * y[n] * delta_r
        n += 1

    # Total vacuole volume
    # v_2 = sum(phi)
    v_2 = 0
    for i in range(len(phi)):
        v_2 += phi[i]

    rho_a1 = (a_1 / ((a_1 / rho_a0) + v_2))
    v_a1 = a_1 / (2 * rho_a1) - v_2

    # Penicillin produced from the non-growing regions  A_1 regions
    r_p = mu_p * rho_a0 * v_a1 * P_inhib * DO_2_inhib_P * PAA_inhib_P - mu_h * y[3]

    # ----- Vacuole formation-------
    r_m1 = (m_s * rho_a0 * v_a1 * s) / (K_v + s)

    # ------ Vacuole degeneration -------------------
    r_d4 = mu_a * a_3

    # ------ Vacuole Volume -------------------
    # n_0 - mean vacoule number density for vacuoles sized ranging from delta_0 -> r_0
    dn0_dt = ((mu_v * v_a1) / (K_v + s)) * ((6 / math.pi) * ((r_0 + delta_0) ** -3)) - k_v * y[15]

    n = 16
    # n_j - mean vacoule number density for vacuoles sized ranging from r_{j}
    dn1_dt = -k_v * ((y[n + 1] - y[n - 1]) / (2 * delta_r)) + D * (y[n + 1] - 2 * y[n] + y[n - 1]) / delta_r ** 2
    n += 1
    dn2_dt = -k_v * ((y[n + 1] - y[n - 1]) / (2 * delta_r)) + D * (y[n + 1] - 2 * y[n] + y[n - 1]) / delta_r ** 2
    n += 1
    dn3_dt = -k_v * ((y[n + 1] - y[n - 1]) / (2 * delta_r)) + D * (y[n + 1] - 2 * y[n] + y[n - 1]) / delta_r ** 2
    n += 1
    dn4_dt = -k_v * ((y[n + 1] - y[n - 1]) / (2 * delta_r)) + D * (y[n + 1] - 2 * y[n] + y[n - 1]) / delta_r ** 2
    n += 1
    dn5_dt = -k_v * ((y[n + 1] - y[n - 1]) / (2 * delta_r)) + D * (y[n + 1] - 2 * y[n] + y[n - 1]) / delta_r ** 2
    n += 1
    dn6_dt = -k_v * ((y[n + 1] - y[n - 1]) / (2 * delta_r)) + D * (y[n + 1] - 2 * y[n] + y[n - 1]) / delta_r ** 2
    n += 1
    dn7_dt = -k_v * ((y[n + 1] - y[n - 1]) / (2 * delta_r)) + D * (y[n + 1] - 2 * y[n] + y[n - 1]) / delta_r ** 2
    n += 1
    dn8_dt = -k_v * ((y[n + 1] - y[n - 1]) / (2 * delta_r)) + D * (y[n + 1] - 2 * y[n] + y[n - 1]) / delta_r ** 2
    n += 1
    dn9_dt = -k_v * ((y[n + 1] - y[n - 1]) / (2 * delta_r)) + D * (y[n + 1] - 2 * y[n] + y[n - 1]) / delta_r ** 2
    n_k = dn9_dt

    # Mean vacoule density for  department k all vacuoles above k in size are assumed constant size
    k = 10
    r_k = r_0 + (k - 2) * delta_r
    k = 12
    r_m = (r_0 + (k - 2) * delta_r)

    # Calculating maximum vacuole volume department
    dn_m_dt = k_v * n_k / (r_m - r_k) - mu_a * y[25]
    n_k = y[24]

    # mean vacuole
    dphi_0_dt = ((mu_v * v_a1) / (K_v + s)) - k_v * y[15] * (math.pi * (r_0 + delta_0) ** 3) / 6

    # Volume and Weight expressions
    F_evp = y[4] * alpha_evp * (math.exp(2.5 * (y[7] - T0) / (Tv - T0)) - 1)
    pho_feed = (c_s / 1000 * pho_g + (1 - c_s / 1000) * pho_w)

    # Dilution term
    dilution = Fs + Fb + Fa + Fw - F_evp + Fpaa

    # Change in Volume
    dV1 = Fs + Fb + Fa + Fw + F_discharge / (pho_b / 1000) - F_evp + Fpaa

    # Change in Weight
    dWt = Fs * pho_feed / 1000 + pho_oil / 1000 * Foil + Fb + Fa + Fw + F_discharge - F_evp + Fpaa * pho_paa / 1000

    # ODE's for Biomass regions
    da_0_dt = r_b0 - r_d1 - y[11] * dilution / y[4]

    # Non growing regions
    da_1_dt = r_e1 - r_b0 + r_d1 - (math.pi * ((r_k + r_m) ** 3) / 6) * rho_d * k_v * n_k - y[12] * dilution / y[4]

    # Degenerated regions
    da_3_dt = (math.pi * ((r_k + r_m) ** 3) / 6) * rho_d * k_v * n_k - r_d4 - y[13] * dilution / y[4]

    # Autolysed regions
    da_4_dt = r_d4 - y[14] * dilution / y[4]

    # Penicillin production
    dP_dt = r_p - y[3] * dilution / y[4]

    # Active Biomass rate
    X_1 = da_0_dt + da_1_dt + da_3_dt + da_4_dt

    # Total biomass
    X_t = y[11] + y[12] + y[13] + y[14]

    Qrxn_X = X_1 * Y_QX * y[4] * Y_O2_X / 1000
    Qrxn_P = dP_dt * Y_QX * y[4] * Y_O2_P / 1000

    Qrxn_t = Qrxn_X + Qrxn_P

    if Qrxn_t < 0:
        Qrxn_t = 0

    N = RPM / 60
    D_imp = 2 * r_imp
    unaerated_power = (n_imp * Po * pho_b * (N ** 3) * (D_imp ** 5))
    P_g = 0.706 * (((unaerated_power ** 2) * (N) * D_imp ** 3) / (Fg ** 0.56)) ** 0.45
    P_n = P_g / unaerated_power
    variable_power = (n_imp * Po * pho_b * (N ** 3) * (D_imp ** 5) * P_n) / 1000

    #
    #   Process parameters
    #
    # Defining the column vector
    # dy = np.zeros((33, 1))
    dy = [0] * 33

    # Substrate utilization
    dy[0] = -r_se1 - r_sb0 - r_m0 - r_m1 - (
            Y_sP * mu_p * rho_a0 * v_a1 * P_inhib * DO_2_inhib_P * PAA_inhib_P) + Fs * c_s / y[4] + Foil * c_oil / \
            y[4] - y[0] * dilution / y[4]

    # Dissolved oxygen
    V_s = Fg / (math.pi * (r ** 2))
    T = y[7]
    V = y[4]
    V_m = y[4] / 1000
    P_air = ((V_s * R * T * V_m / (22.4 * h_b)) * math.log(1 + pho_b * 9.81 * h_b / (pressure_top * 1e5)))
    P_t1 = (variable_power + P_air)
    viscosity = 1 if viscosity <= 4 else viscosity
    vis_scaled = (viscosity / 100)
    oil_f = (Foil / V)
    kla = (alpha_kla * ((V_s ** a) * ((P_t1 / V_m) ** b) * vis_scaled ** c) * (1 - oil_f ** d))
    OUR = (-X_1) * Y_O2_X - m_O2_X * X_t - dP_dt * Y_O2_P
    OTR = kla * (DOstar_tp - y[1])
    dy[1] = OUR + OTR - (y[1] * dilution / y[4])

    # O_2 off-gas
    Vg = epsilon * V_m
    Qfg_in = 60 * Fg * 1000 * 32 / 22.4
    Qfg_out = 60 * Fg * (N2_in / (1 - y[2] - y[27] / 100)) * 1000 * 32 / 22.4
    dy[2] = (Qfg_in * O_2_in - Qfg_out * y[2] - 0.001 * OTR * V_m * 60) / (Vg * 28.97 * 1000 / 22.4)

    # Penicillin production rate
    dy[3] = r_p - y[3] * dilution / y[4]

    # Volume change
    dy[4] = dV1

    # Weight change
    dy[5] = dWt

    # pH
    pH_dis = Fs + Foil + Fb + Fa + F_discharge + Fw
    if -math.log10(y[6]) < 7:
        cb = -abc
        ca = abc
        pH_balance = 0
    else:
        cb = abc
        ca = -abc
        y[6] = (1e-14 / y[6] - y[6])
        pH_balance = 1

    # Calculation of ion addition
    B = (y[6] * y[4] + ca * Fa * step1 + cb * Fb * step1) / (y[4] + Fb * step1 + Fa * step1)
    B = -B

    if pH_balance == 1:
        dy[6] = -gamma1 * (r_b0 + r_e1 + r_d4 + r_d1 + m_ph * total_X) - gamma1 * r_p - gamma2 * pH_dis + (
                (-B - np.sqrt(B ** 2 + 4e-14)) / 2 - y[6])

    if pH_balance == 0:
        dy[6] = gamma1 * (r_b0 + r_e1 + r_d4 + r_d1 + m_ph * total_X) + gamma1 * r_p + gamma2 * pH_dis + (
                (-B + np.sqrt(B ** 2 + 4e-14)) / 2 - y[6])

    # Temperature
    Ws = P_t1
    Qcon = U_jacket * A_c * (y[7] - Tair)
    dQ_dt = Fs * pho_feed * C_ps * (Tf - y[7]) / 1000 + Fw * pho_w * C_pw * (
            Tw - y[7]) / 1000 - F_evp * pho_b * C_pw / 1000 - dealta_H_evap * F_evp * pho_w / 1000 + Qrxn_t + Ws - (
                    alpha_1 / 1000) * Fc ** (beta_T + 1) * (
                    (y[7] - Tcin) / (Fc / 1000 + (alpha_1 * (Fc / 1000) ** beta_T) / 2 * pho_b * C_ps)) - (
                    alpha_1 / 1000) * Fh ** (beta_T + 1) * (
                    (y[7] - Th) / (Fh / 1000 + (alpha_1 * (Fh / 1000) ** beta_T) / 2 * pho_b * C_ps)) - Qcon
    dy[7] = dQ_dt / ((y[4] / 1000) * C_pw * pho_b)

    # Heat generation
    dy[8] = dQ_dt

    # Viscosity
    dy[9] = 3 * (a_0 ** (1 / 3)) * (1 / (1 + math.exp(-k1 * (t - t1)))) * (1 / (1 + math.exp(-k2 * (t - t2)))) - k3 * Fw

    # Total X
    dy[10] = (y[11] + y[12] + y[13] + y[14])

    #
    #   Adding in the ODE's for hyphae
    #
    dy[11] = da_0_dt
    dy[12] = da_1_dt
    dy[13] = da_3_dt
    dy[14] = da_4_dt
    dy[15] = dn0_dt
    dy[16] = dn1_dt
    dy[17] = dn2_dt
    dy[18] = dn3_dt
    dy[19] = dn4_dt
    dy[20] = dn5_dt
    dy[21] = dn6_dt
    dy[22] = dn7_dt
    dy[23] = dn8_dt
    dy[24] = dn9_dt
    dy[25] = dn_m_dt
    dy[26] = dphi_0_dt

    # CO_2
    total_X_CO2 = y[11] + y[12]
    CER = total_X_CO2 * q_co2 * V
    dy[27] = (((60 * Fg * 44 * 1000) / 22.4) * C_CO2_in + CER - ((60 * Fg * 44 * 1000) / 22.4) * y[27]) / (
            Vg * 28.97 * 1000 / 22.4)

    # dissolved CO_2
    Henrys_c_co2 = (math.exp(11.25 - 395.9 / (y[7] - 175.9))) / (44 * 100)
    C_star_CO2 = (total_pressure * y[27]) / Henrys_c_co2
    dy[28] = kla * delta_c_0 * (C_star_CO2 - y[28]) - y[28] * dilution / y[4]

    # PAA
    dy[29] = Fpaa * PAA_c / V - (Y_PAA_P * dP_dt) - Y_PAA_X * X_1 - m_PAA * y[3] - y[29] * dilution / y[4]

    # N
    X_C_nitrogen = (-r_b0 - r_e1 - r_d1 - r_d4) * Y_NX
    P_C_nitrogen = -dP_dt * Y_NP
    dy[30] = (NH3_shots * N_conc_shot) / y[4] + X_C_nitrogen + P_C_nitrogen - m_N * total_X + (
            1 * N_conc_paa * Fpaa / y[4]) + N_conc_oil * Foil / y[4] - y[30] * dilution / y[4]
    dy[31] = mu_p
    dy[32] = mu_e

    return dy
