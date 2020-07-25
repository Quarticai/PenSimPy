from copy import deepcopy
import numpy as np
from ax.core.observation import ObservationFeatures, observations_from_data
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.core.objective import Objective
from ax.runners.synthetic import SyntheticRunner
from ax.core.parameter import RangeParameter, ParameterType
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.modelbridge.factory import get_sobol, get_GPEI, get_MTGP
from ax.core.generator_run import GeneratorRun
from ax.plot.diagnostic import interact_batch_comparison
from ax.plot.trace import optimization_trace_all_methods
from ax.utils.notebook.plotting import init_notebook_plotting, render
import time
from pensimpy.pensim_classes.Recipe import Recipe
from pensimpy.env_setup.peni_env_setup import PenSimEnv
from ax.metrics.noisy_function import NoisyFunctionMetric


class OnlinePensim(NoisyFunctionMetric):
    def f(self, x):
        env = PenSimEnv(random_seed_ref=247)
        done = False
        observation, batch_data = env.reset()
        recipe = Recipe(x.tolist())

        time_stamp, batch_yield, yield_pre = 0, 0, 0
        while not done:
            time_stamp += 1
            Fs, Foil, Fg, pressure, Fremoved, Fw, Fpaa = recipe.run(time_stamp)
            observation, batch_data, reward, done = env.step(time_stamp,
                                                             batch_data,
                                                             Fs, Foil, Fg, pressure, Fremoved, Fw, Fpaa)
            batch_yield += reward

        return -batch_yield


class OfflinePensim(NoisyFunctionMetric):
    def f(self, x):
        # Random_seed_ref from 0 to 1000, 274 excluded
        seed = np.random.randint(1000)
        while seed == 247:
            seed = np.random.randint(1000)

        env = PenSimEnv(random_seed_ref=seed)
        done = False
        observation, batch_data = env.reset()
        recipe = Recipe(x.tolist())

        time_stamp, batch_yield, yield_pre = 0, 0, 0
        while not done:
            time_stamp += 1
            Fs, Foil, Fg, pressure, Fremoved, Fw, Fpaa = recipe.run(time_stamp)
            observation, batch_data, reward, done = env.step(time_stamp,
                                                             batch_data,
                                                             Fs, Foil, Fg, pressure, Fremoved, Fw, Fpaa)
            batch_yield += reward

        return -batch_yield


def get_experiment(include_true_metric=False):
    noise_sd = 0  # Observations will have this much Normal noise added to them
    x = [8, 15, 30, 75, 150, 30, 37, 43, 47, 51, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80,
         22, 30, 35, 34, 33, 32, 31, 30, 29, 23,
         30, 42, 55, 60, 75, 65, 60]

    # 1. Create simple search space for [0,1]^d, d=38
    param_names = [f"x{i}" for i in range(len(x))]
    parameters = [
        RangeParameter(
            name=param_names[key], parameter_type=ParameterType.INT, lower=int(val * 0.9), upper=int(val * 1.1)
        )
        for key, val in enumerate(x)
    ]
    search_space = SearchSpace(parameters=parameters)

    # 2. Specify optimization config
    online_objective = OnlinePensim("objective", param_names=param_names, noise_sd=noise_sd)
    opt_config = OptimizationConfig(
        objective=Objective(online_objective, minimize=True),
    )

    # 3. Init experiment
    exp = MultiTypeExperiment(
        name="mt_exp",
        search_space=search_space,
        default_trial_type="online",
        default_runner=SyntheticRunner(),
        optimization_config=opt_config,
    )

    # 4. Establish offline trial_type, and how those trials are deployed
    exp.add_trial_type("offline", SyntheticRunner())

    # 5. Add offline metrics that provide biased estimates of the online metrics
    offline_objective = OfflinePensim("offline_objective", param_names=param_names, noise_sd=noise_sd)
    exp.add_tracking_metric(metric=offline_objective, trial_type="offline", canonical_name="objective")

    # # Add a noiseless equivalent for each metric, for tracking the true value of each observation
    # # for the purposes of benchmarking.
    # if include_true_metric:
    # exp.add_tracking_metric(OnlinePensim("objective_noiseless", param_names=param_names, noise_sd=0.0), "online")
    return exp


# Settings for the optimization benchmark.

# This should be changed to 50 to reproduce the results from the paper.
n_reps = 1  # Number of repeated experiments, each with independent observation noise

n_init_online = 4  # Size of the quasirandom initialization run online
n_init_offline = 10  # Size of the quasirandom initialization run offline
n_opt_online = 4  # Batch size for BO selected points to be run online
n_opt_offline = 50  # Batch size for BO selected to be run offline
n_batches = 10  # Number of optimized BO batches


# n_init_online = 4  # Size of the quasirandom initialization run online
# n_init_offline = 4  # Size of the quasirandom initialization run offline
# n_opt_online = 1  # Batch size for BO selected points to be run online
# n_opt_offline = 10  # Batch size for BO selected to be run offline
# n_batches = 100  # Number of optimized BO batches


# This function runs a Bayesian optimization loop, making online observations only.
def run_online_only_bo():
    t1 = time.time()
    ### Do BO with online only
    ## Quasi-random initialization
    exp_online = get_experiment()
    m = get_sobol(exp_online.search_space, scramble=False)
    gr = m.gen(n=n_init_online)
    exp_online.new_batch_trial(trial_type="online", generator_run=gr).run()
    ## Do BO
    for b in range(n_batches):
        print('Online-only batch', b, time.time() - t1)
        # Fit the GP
        m = get_GPEI(
            experiment=exp_online,
            data=exp_online.fetch_data(),
            search_space=exp_online.search_space,
        )
        # Generate the new batch
        gr = m.gen(
            n=n_opt_online,
            search_space=exp_online.search_space,
            optimization_config=exp_online.optimization_config,
        )
        exp_online.new_batch_trial(trial_type="online", generator_run=gr).run()
    ## Extract true objective and constraint at each iteration
    df = exp_online.fetch_data().df
    return df['mean'].values


# Online batches are constructed by selecting the maximum utility points from the offline
# batch, after updating the model with the offline results. This function selects the max utility points according
# to the MTGP predictions.
def max_utility_from_GP(n, m, experiment, search_space, gr):
    obsf = []
    for arm in gr.arms:
        params = deepcopy(arm.parameters)
        params['trial_type'] = 'online'
        obsf.append(ObservationFeatures(parameters=params))
    # Make predictions
    f, cov = m.predict(obsf)
    # Compute expected utility
    u = -np.array(f['objective'])
    best_arm_indx = np.flip(np.argsort(u))[:n]
    gr_new = GeneratorRun(
        arms=[
            gr.arms[i] for i in best_arm_indx
        ],
        weights=[1.] * n,
    )
    return gr_new


# This function runs a multi-task Bayesian optimization loop, as outlined in Algorithm 1 and above.
def run_mtbo():
    t1 = time.time()
    online_trials = []
    ## 1. Quasi-random initialization, online and offline
    exp_multitask = get_experiment()
    # Online points
    m = get_sobol(exp_multitask.search_space, scramble=False)
    gr = m.gen(
        n=n_init_online,
    )
    tr = exp_multitask.new_batch_trial(trial_type="online", generator_run=gr)
    tr.run()
    online_trials.append(tr.index)
    # Offline points
    m = get_sobol(exp_multitask.search_space, scramble=False)
    gr = m.gen(
        n=n_init_offline,
    )
    exp_multitask.new_batch_trial(trial_type="offline", generator_run=gr).run()
    ## Do BO
    for b in range(n_batches):
        print('Multi-task batch', b, time.time() - t1)
        # (2 / 7). Fit the MTGP
        m = get_MTGP(
            experiment=exp_multitask,
            data=exp_multitask.fetch_data(),
            search_space=exp_multitask.search_space,
        )

        # 3. Finding the best points for the online task
        gr = m.gen(
            n=n_opt_offline,
            optimization_config=exp_multitask.optimization_config,
            fixed_features=ObservationFeatures(parameters={'trial_type': 'online'}),
        )

        # 4. But launch them offline
        exp_multitask.new_batch_trial(trial_type="offline", generator_run=gr).run()

        # 5. Update the model
        m = get_MTGP(
            experiment=exp_multitask,
            data=exp_multitask.fetch_data(),
            search_space=exp_multitask.search_space,
        )

        # 6. Select max-utility points from the offline batch to generate an online batch
        gr = max_utility_from_GP(
            n=n_opt_online,
            m=m,
            experiment=exp_multitask,
            search_space=exp_multitask.search_space,
            gr=gr,
        )
        tr = exp_multitask.new_batch_trial(trial_type="online", generator_run=gr)
        tr.run()
        online_trials.append(tr.index)
    # Extract true objective at each online iteration for creating benchmark plot
    obj = []
    for tr in online_trials:
        df_t = exp_multitask.trials[tr].fetch_data().df
        obj.extend(df_t['mean'].values)
    return np.array(obj)


runners = {
    'GP, online only': run_online_only_bo,
    'MTGP': run_mtbo,
}

iteration_objectives = {k: [] for k in runners}
for rep in range(n_reps):
    print('Running rep', rep)
    for k, r in runners.items():
        obj = r()
        print(f'=== obj: {obj}')
        iteration_objectives[k].append(obj)

for k, v in iteration_objectives.items():
    iteration_objectives[k] = np.array(v)

print(f"=== iteration_objectives: {iteration_objectives}")

best_objectives = {}
for m, obj in iteration_objectives.items():
    x = obj.copy()
    # best_objectives[m] = np.array([np.minimum.accumulate(obj_i) for obj_i in x])
    best_objectives[m] = x

print(f"=== best_objectives: {best_objectives}")
render(
    optimization_trace_all_methods({k: best_objectives[k] for k in runners})
)
