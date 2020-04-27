from BatchRunFlags import BatchRunFlags
import numpy as np
import time
from indpensim_run import indpensim_run
import statistics
from show_params import show_params
from save_csv import save_csv

if __name__ == "__main__":
    total_runs = 1
    num_of_batches = 1
    plot_res = True
    save_res = True

    batch_run_flags = BatchRunFlags(num_of_batches)
    peni_products = []

    for run_id in range(total_runs):
        print(f"===== run_id: {run_id}")
        Recipe_Fs_sp_paper = [8, 15, 30, 75, 150, 30, 37, 43, 47, 51, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80]

        if run_id == 0:
            Recipe_Fs_sp = Recipe_Fs_sp_paper
        elif run_id == 1:
            Recipe_Fs_sp = [8, 15, 30, 75, 150, 30, 37, 43, 47, 55, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80]
        elif run_id == 2:
            Recipe_Fs_sp = [8, 15, 30, 75, 150, 30, 37, 43, 47, 44, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80]
        elif 3 <= run_id < 6:
            Recipe_Fs_sp = [ele + np.random.randint(-1, 1) for ele in Recipe_Fs_sp_paper]
        else:
            Recipe_Fs_sp = [ele + np.random.randint(-5, 5) for ele in Recipe_Fs_sp_paper]

        print(f"=== Recipe_Fs_sp: {Recipe_Fs_sp}")

        # For backend APIs
        penicillin_yields = []
        avg_pHs = []
        avg_Ts = []
        pHs = []
        Ts = []
        median_pH = []
        median_T = []

        for Batch_no in range(1, num_of_batches + 1):
            print(f"==== Batch_no: {Batch_no}")
            t = time.time()
            Xref, h = indpensim_run(Batch_no, batch_run_flags)
            print(f"=== cost: {time.time() - t} s")

            penicillin_harvested_during_batch = sum([a * b for a, b in zip(Xref.Fremoved.y, Xref.P.y)]) * h
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
        median_pH = [statistics.median(pHs[:, i]) for i in range(0, len(pHs[0]))]

        Ts = np.array(Ts)
        median_T = [statistics.median(Ts[:, i]) for i in range(0, len(Ts[0]))]

        peni_products.append(penicillin_yields)

        # Save data
        if save_res:
            save_csv(run_id, avg_pHs, avg_Ts, penicillin_yields, median_pH, median_T)

    print(f"=== peni_products: {peni_products}")
    print(f"=== Viscosity_offline: {Xref.Viscosity_offline.y}")
    # print(f"=== mean: {np.mean(peni_products)}")
    # print(f"=== std: {np.std(peni_products,ddof=1)}")

    # Plot the last res
    if plot_res:
        show_params(Xref)

