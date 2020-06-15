from pensimpy.pensim_classes.BatchRunFlags import BatchRunFlags
import numpy as np
import time

from pensimpy.pensim_classes.Constants import H
from pensimpy.pensim_classes.Recipe import Recipe
from pensimpy.pensim_methods.indpensim_run import indpensim_run
import statistics
from pensimpy.helper.show_params import show_params
from pensimpy.helper.save_csv import save_csv
import argparse


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--total_runs', type=int)
    p.add_argument('--num_of_batches', type=int)
    p.add_argument('--plot_res', type=int)
    p.add_argument('--save_res', type=int)
    args = p.parse_args()

    total_runs = args.total_runs
    num_of_batches = args.num_of_batches
    plot_res = args.plot_res
    save_res = args.save_res

    batch_run_flags = BatchRunFlags(num_of_batches)
    peni_products = []

    recipe = Recipe()

    print("=== Run simulation...")
    for run_id in range(total_runs):
        print(f"=== run_id: {run_id}")
        # For backend APIs
        penicillin_yields = []
        avg_pHs = []
        avg_Ts = []
        pHs = []
        Ts = []
        median_pH = []
        median_T = []
        penicillin_predictions = []
        sum_intensity = np.zeros(2200)

        for Batch_no in range(1, num_of_batches + 1):
            print(f"==== Batch_no: {Batch_no}")
            t = time.time()
            for result in indpensim_run(Batch_no, batch_run_flags, recipe):
                if result['type'] == 'raman_update':
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

            # penicillin_harvested_during_batch = sum([a * b for a, b in zip(Xref.Fremoved.y, Xref.P.y)]) * H
            penicillin_harvested_during_batch = np.dot(Xref.Fremoved.y, Xref.P.y) * H
            penicillin_harvested_end_of_batch = Xref.V.y[-1] * Xref.P.y[-1]
            penicillin_yield_total = penicillin_harvested_end_of_batch - penicillin_harvested_during_batch

            penicillin_yields.append(penicillin_yield_total / 1000)
            avg_pH = sum(Xref.pH.y) / len(Xref.pH.y)
            avg_pHs.append(avg_pH)
            avg_T = sum(Xref.T.y) / len(Xref.T.y)
            avg_Ts.append(avg_T)

            pHs.append(Xref.pH.y)
            Ts.append(Xref.T.y)
            print(f"=== penicillin_yield_total: {penicillin_yield_total / 1000}")

        pHs = np.array(pHs)
        median_pH = [statistics.median(pHs[:, i]) for i in range(len(pHs[0]))]
        Ts = np.array(Ts)
        median_T = [statistics.median(Ts[:, i]) for i in range(len(Ts[0]))]
        peni_products.append(penicillin_yields)
        # Save data
        if save_res == 1:
            save_csv(run_id, avg_pHs, avg_Ts, penicillin_yields, median_pH, median_T, Xref)

    print(f"=== peni_products: {peni_products}")
    # Plot the last res
    if plot_res == 1:
        show_params(Xref)
