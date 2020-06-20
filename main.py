import time
from pensimpy.pensim_classes.Recipe import Recipe
import numpy as np
from pensimpy.pensim_methods.indpensim import indpensim
from pensimpy.env_setup.peni_env_setup import PenSimEnv
import math
from pensimpy.pensim_classes.Constants import raman_wavenumber
from pensimpy.pensim_methods.create_batch import create_batch

if __name__ == "__main__":
    penicillin_predictions = []
    sum_intensity = np.zeros(2200)

    t = time.time()

    env = PenSimEnv()
    xinterp, x0, H, Batch_lenght, param_list, ctrl_flags = env.reset()
    recipe = Recipe()

    k = 1
    x = create_batch(H, Batch_lenght)
    tmp = 0
    while k != int(Batch_lenght / H) + 1:
        x = indpensim(k, x, xinterp, x0, H, Batch_lenght, param_list, ctrl_flags, recipe)
        peni_yield = x.V.y[k - 1] * x.P.y[k - 1] / 1000
        print(f"=== yield: {peni_yield - tmp}")
        tmp = peni_yield
        k += 1

    # print(yield_sum)

    # convert to pH from H+ concentration
    x.pH.y = [-math.log(pH) / math.log(10) if pH != 0 else pH for pH in x.pH.y]
    x.Q.y = [Q / 1000 for Q in x.Q.y]
    x.Raman_Spec.Wavenumber = raman_wavenumber

    print(f"=== cost: {int(time.time() - t)} s")
    penicillin_yield_total = (x.V.y[-1] * x.P.y[-1] - np.dot(x.Fremoved.y, x.P.y) * H) / 1000
    print(f"=== penicillin_yield: {penicillin_yield_total}")

    # # Plot the last res
    # show_params(Xref)
