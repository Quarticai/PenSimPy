def parameter_list(mu_p, mux_max, alpha_kla, N_conc_paa, PAA_c):
    """
    Fixed params for the fermentation process
    :param mu_p:
    :param mux_max:
    :param alpha_kla:
    :param N_conc_paa:
    :param PAA_c:
    :return:
    """
    # Penicillin model parameters
    ratio_mu_e_mu_b = 0.4
    P_std_dev = 0.0015
    mean_P = 0.002
    mu_v = 1.71e-4
    mu_a = 3.5e-3
    mu_diff = 5.36e-3
    beta_1 = 0.006
    K_b = 0.05
    K_diff = 0.75
    K_diff_L = 0.09
    K_e = 0.009
    K_v = 0.05
    delta_r = 0.75e-4
    k_v = 3.22e-5
    D = 2.66e-11
    rho_a0 = 0.35
    rho_d = 0.18
    mu_h = 0.003
    r_0 = 1.5e-4
    delta_0 = 1e-4
    # Process related parameters
    Y_sx = 1.85
    Y_sP = 0.9
    m_s = 0.029
    c_oil = 1000
    c_s = 600
    Y_O2_X = 650
    Y_O2_P = 160
    m_O2_X = 17.5
    a = 0.38
    b = 0.34
    c = -0.38
    d = 0.25
    Henrys_c = 0.0251
    n_imp = 3
    r = 2.1
    r_imp = 0.85
    Po = 5
    epsilon = 0.1
    g = 9.81
    R = 8.314
    X_crit_DO2 = 0.1
    P_crit_DO2 = 0.3
    A_inhib = 1
    Tf = 288
    Tw = 288
    Tcin = 285
    Th = 333
    Tair = 290
    C_ps = 5.9
    C_pw = 4.18
    dealta_H_evap = 2430.7
    U_jacket = 36
    A_c = 105
    Eg = 14880
    Ed = 173250
    k_g = 450
    k_d = 2.5e+29
    Y_QX = 25
    abc = 0.033
    gamma1 = 0.0325e-5
    gamma2 = 2.5e-11
    m_ph = 0.0025
    K1 = 1e-5
    K2 = 2.5e-8
    N_conc_oil = 20000
    N_conc_shot = 400000
    Y_NX = 10
    Y_NP = 80
    m_N = 0.03
    X_crit_N = 150
    Y_PAA_P = 187.5
    Y_PAA_X = 45
    m_PAA = 1.05
    X_crit_PAA = 2400
    P_crit_PAA = 200
    B_1 = -64.29
    B_2 = -1.825
    B_3 = 0.3649
    B_4 = 0.1280
    B_5 = -4.9496e-04
    delta_c_o = 0.89
    k_3 = 0.005
    k1 = 0.001
    k2 = 0.0001
    t1 = 1
    t2 = 250
    q_co2 = 0.1353
    X_crit_CO2 = 7570
    alpha_evp = 5.2400e-4
    beta_T = 2.88
    pho_g = 1540
    pho_oil = 900
    pho_w = 1000
    pho_paa = 1000
    O_2_in = 0.21
    N2_in = 0.79
    C_CO2_in = 0.033
    Tv = 373
    T0 = 273
    alpha_1 = 2451.8
    return [mu_p, mux_max, ratio_mu_e_mu_b, P_std_dev, mean_P, mu_v, mu_a, mu_diff, beta_1, K_b, K_diff, K_diff_L, K_e,
            K_v, delta_r, k_v, D, rho_a0, rho_d, mu_h, r_0, delta_0, Y_sx, Y_sP, m_s, c_oil, c_s, Y_O2_X, Y_O2_P,
            m_O2_X, alpha_kla, a, b, c, d, Henrys_c, n_imp, r, r_imp, Po, epsilon, g, R, X_crit_DO2, P_crit_DO2,
            A_inhib, Tf, Tw, Tcin, Th, Tair, C_ps, C_pw, dealta_H_evap, U_jacket, A_c, Eg, Ed, k_g, k_d, Y_QX, abc,
            gamma1, gamma2, m_ph, K1, K2, N_conc_oil, N_conc_paa, N_conc_shot, Y_NX, Y_NP, m_N, X_crit_N, PAA_c,
            Y_PAA_P, Y_PAA_X, m_PAA, X_crit_PAA, P_crit_PAA, B_1, B_2, B_3, B_4, B_5, delta_c_o, k_3, k1, k2, t1, t2,
            q_co2, X_crit_CO2, alpha_evp, beta_T, pho_g, pho_oil, pho_w, pho_paa, O_2_in, N2_in, C_CO2_in, Tv, T0,
            alpha_1]
