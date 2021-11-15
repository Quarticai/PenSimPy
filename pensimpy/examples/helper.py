import math
from datetime import datetime
from pensimpy.constants import STEP_IN_MINUTES, MINUTES_PER_HOUR
from pensimpy.peni_env_setup import PenSimEnv
from pensimpy.utils import get_dataframe
import numpy as np
import random
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from hilo.core.recipe import Recipe, FillingMethod
from hilo.core.recipe_combo import RecipeCombo
from pensimpy.data.constants import FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA
from pensimpy.data.constants import FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, \
    FG_DEFAULT_PROFILE, PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE, \
    WATER_DEFAULT_PROFILE, PAA_DEFAULT_PROFILE

BASE_DF = pd.read_csv('base_df.csv')
NUM_POINTS_PER_BATCH = 1150
BATCH_DURATION_IN_SEC = 230 * 60 * 60
MAX_FS = max(FS_DEFAULT_PROFILE, key=lambda x: x['value'])['value']
DATE_TIME = 'Datetime'
GRANULARITY = '0.2H'


class FailureReason:
    DOWNTIME = 1
    OVERFLOW = 2
    NOISE = 3


def set_labels():
    failure_ranges = [
        {
            "value": FailureReason.DOWNTIME,
            "start": random.randint(1, 950),
            "size": random.randint(10, 20),
        },
        {
            "value": FailureReason.OVERFLOW,
            "start": random.randint(1, 950),
            "size": random.randint(10, 20)
        },
        {
            "value": FailureReason.NOISE,
            "start": random.randint(1, 950),
            "size": random.randint(10, 20)
        }
    ]

    labels = [0] * NUM_POINTS_PER_BATCH
    for failure_range in failure_ranges:
        start = failure_range['start'] - 1
        size = failure_range['size']
        end = start + size - 1
        labels[start: end + 1] = [failure_range['value']] * size

    return labels


class PenSimEnvGen(PenSimEnv):
    def get_batches_with(self, labels, random_seed=0):
        """
        Generate batch data in pandas dataframes.
        """
        self.random_seed_ref = random_seed

        done = False
        observation, batch_data = self.reset()
        k_timestep, batch_yield, yield_pre = 0, 0, 0

        self.yield_pre = 0
        peni_yield = []
        f1_scores = []
        accuracies = []
        while not done:
            k_timestep += 1
            # Get action from recipe agent based on time
            values_dict = self.recipe_combo.get_values_dict_at(time=k_timestep * STEP_IN_MINUTES / MINUTES_PER_HOUR)
            Fs, Foil, Fg, pressure, discharge, Fw, Fpaa = values_dict['Fs'], values_dict['Foil'], values_dict['Fg'], \
                                                          values_dict['pressure'], values_dict['discharge'], \
                                                          values_dict['Fw'], values_dict['Fpaa']

            if labels[k_timestep - 1] == 1:
                Fs = 0
            elif labels[k_timestep - 1] == 2:
                Fs = MAX_FS * 2
            elif labels[k_timestep - 1] == 3:
                Fs = np.random.normal(Fs, 0.2 * Fs)

            # Run and get the reward
            # observation is a class which contains all the variables, e.g. observation.Fs.y[k], observation.Fs.t[k]
            # are the Fs value and corresponding time at k
            observation, batch_data, reward, done = self.step(k_timestep,
                                                              batch_data,
                                                              Fs, Foil, Fg, pressure, discharge, Fw, Fpaa)
            batch_yield += reward
            peni_yield.append(reward)
            f1_scores.append(f1_score(labels[:k_timestep], [0] * k_timestep, average='macro'))
            accuracies.append(accuracy_score(labels[:k_timestep], [0] * k_timestep))

        df, _ = get_dataframe(batch_data, False)
        df['peni_yield'] = peni_yield
        df['label'] = labels
        df['f1_score'] = f1_scores
        df['accuracy'] = accuracies
        return df, batch_yield


RECIPE_DICT = {
    FS: Recipe(FS_DEFAULT_PROFILE, FS),
    FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),
    FG: Recipe(FG_DEFAULT_PROFILE, FG),
    PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),
    DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
    WATER: Recipe(WATER_DEFAULT_PROFILE, WATER),
    PAA: Recipe(PAA_DEFAULT_PROFILE, PAA)
}

recipe_combo = RecipeCombo(recipe_dict=RECIPE_DICT, filling_method=FillingMethod.BACKWARD)
env = PenSimEnvGen(recipe_combo=recipe_combo)


def get_num_batches(
    start_dt: str,
    end_dt: str,
    duration: int = BATCH_DURATION_IN_SEC
) -> int:
    start_dt = datetime.strptime(start_dt, '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.strptime(end_dt, '%Y-%m-%d %H:%M:%S')
    diff_in_sec = (end_dt - start_dt).total_seconds()
    return math.ceil(diff_in_sec / duration)
