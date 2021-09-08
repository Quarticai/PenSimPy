from hilo.core.recipe import Recipe, FillingMethod
from hilo.core.recipe_combo import RecipeCombo
import matplotlib.pyplot as plt
from pensimpy.constants import STEP_IN_MINUTES, MINUTES_PER_HOUR
from pensimpy.peni_env_setup import PenSimEnv
from pensimpy.data.constants import FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA
from pensimpy.data.constants import FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, \
    FG_DEFAULT_PROFILE, PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE, \
    WATER_DEFAULT_PROFILE, PAA_DEFAULT_PROFILE
from pensimpy.utils import get_dataframe
import numpy as np
import random
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

BASE_DF = pd.read_csv('base_df.csv')
BASE_TEMP = BASE_DF['Temperature'].values.tolist()


class FailureReason:
    DOWNTIME = 1
    OVERFLOW = 2
    NOISE = 3


MAX_FS = max(FS_DEFAULT_PROFILE, key=lambda x: x['value'])['value']


def get_labels():
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

    labels = [0] * 1150
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

recipe_combo = RecipeCombo(
    recipe_dict=RECIPE_DICT,
    filling_method=FillingMethod.BACKWARD
)
env = PenSimEnvGen(recipe_combo=recipe_combo)

num_good_batches = 1
dfs = []
for _ in range(num_good_batches):
    df, peni_yield = env.get_batches_with(labels=get_labels(), random_seed=1)
    # df.to_csv('base_df.csv')
    print(f"=== peni_yield: {peni_yield}")
    dfs.append(df)

df_total = pd.concat(dfs)
print(df_total)
# df_total.to_csv('good_dfs.csv')

plt.figure(1)
plt.subplot(4, 2, 1)
plt.title('Sugar feed rate')
plt.plot(df['Sugar feed rate'])
plt.subplot(4, 2, 2)
plt.title('label')
plt.plot(df['label'])
plt.subplot(4, 2, 3)
plt.title('Base Temperature')
plt.plot(BASE_DF['Temperature'])
plt.subplot(4, 2, 4)
plt.title('Temperature')
plt.plot(df['Temperature'])
plt.subplot(4, 2, 5)
plt.title('f1_score')
plt.plot(df['f1_score'])
plt.subplot(4, 2, 6)
plt.title('accuracy')
plt.plot(df['accuracy'])
plt.subplot(4, 2, 7)
plt.title('base_dist')
plt.hist(BASE_DF['Temperature'], bins=100)
plt.subplot(4, 2, 8)
plt.title('dist')
plt.hist(df['Temperature'], bins=100)
plt.tight_layout()
plt.show()
