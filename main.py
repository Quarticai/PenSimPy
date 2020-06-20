import numpy as np
import time

from pensimpy.pensim_classes.Constants import H
from pensimpy.pensim_classes.Recipe import Recipe
from pensimpy.pensim_methods.indpensim_run import indpensim_run

if __name__ == "__main__":
    recipe = Recipe()

    penicillin_predictions = []
    sum_intensity = np.zeros(2200)

    t = time.time()
    for result in indpensim_run(recipe):
        if result['type'] == 'raman_update':
            pass
            # get intensity
            intensity = result['Intensity']
            sum_intensity += intensity

            # kth 12 minutes
            k = result['k']

            # apply ML model for prediction:  penicillin = Ml(odeintensity)
            penicillin_predictions.append(1)

            # TODO: feed intensities to model to get pensim concentration prediction
            # TODO: send concentration prediction as well as k via websocket
        elif result['type'] == 'batch_end':
            # for returning final accuracy and average intensity
            # TODO: send final accuracy and averaged intensity via websocket

            # avg_intensity = sum_intensity / 1150
            #
            # # lower/upper bound is based on the interpolation method
            # lower_bound = 1
            # upper_bound = 1
            # res = penicillin_yields[(penicillin_yields > lower_bound) & (penicillin_yields < upper_bound)]
            # accuracy = round(len(res) / len(penicillin_yields) * 100, 2)
            Xref = result['x']
        else:
            raise ValueError("Unknown flag")

    print(f"=== cost: {int(time.time() - t)} s")

    penicillin_yield_total = (Xref.V.y[-1] * Xref.P.y[-1] - np.dot(Xref.Fremoved.y, Xref.P.y) * H) / 1000
    print(f"=== penicillin_yield: {penicillin_yield_total}")

    # # Plot the last res
    # show_params(Xref)
