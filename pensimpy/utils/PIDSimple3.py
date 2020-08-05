def PIDSimple3(uk1, ek, ek1, yk, yk1, yk2, u_min, u_max, Kp, Ti, Td, h):
    """
    PID controller
    :param uk1:
    :param ek:
    :param ek1:
    :param yk:
    :param yk1:
    :param yk2:
    :param u_min:
    :param u_max:
    :param Kp:
    :param Ti:
    :param Td:
    :param h:
    :return:
    """
    # proportional component
    P = ek - ek1
    # checks if the integral time constant is defined
    I = ek * h / Ti if Ti > 1e-7 else 0
    # derivative component
    D = -Td / h * (yk - 2 * yk1 + yk2) if Td > 0.001 else 0
    # computes and saturates the control signal
    uu = uk1 + Kp * (P + I + D)
    uu = u_max if uu > u_max else uu
    uu = u_min if uu < u_min else uu

    return uu
