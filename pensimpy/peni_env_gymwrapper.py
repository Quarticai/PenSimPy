import math
import os
import csv
import codecs
import sys
import random
import numpy as np
from gym import spaces, Env
from pensimpy.peni_env_setup import PenSimEnv
from hilo.core.recipe import Recipe, FillingMethod
from hilo.core.recipe_combo import RecipeCombo
from pensimpy.data.constants import FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA
from pensimpy.data.constants import FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE, \
    PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE, WATER_DEFAULT_PROFILE, PAA_DEFAULT_PROFILE

csv.field_size_limit(sys.maxsize)
random.seed(0)
MINUTES_PER_HOUR = 60
BATCH_LENGTH_IN_MINUTES = 230 * MINUTES_PER_HOUR
BATCH_LENGTH_IN_HOURS = 230
STEP_IN_MINUTES = 12
STEP_IN_HOURS = STEP_IN_MINUTES / MINUTES_PER_HOUR
NUM_STEPS = int(BATCH_LENGTH_IN_MINUTES / STEP_IN_MINUTES)
WAVENUMBER_LENGTH = 2200


def get_observation_data_reformed(observation, t):
    """
    Get observation data at t.

    vars are Temperature,Acid flow rate,Base flow rate,Cooling water,Heating water,Vessel Weight,Dissolved oxygen concentration 
    respectively in csv terms, but used abbreviation here to stay consistent with peni_env_setup
    """
    vars = ['T', 'Fa', 'Fb', 'Fc', 'Fh', 'Wt', 'DO2']
    pH = observation.pH.y[t]
    pH = -math.log(pH) / math.log(10) if pH != 0 else pH
    return [t * STEP_IN_MINUTES / MINUTES_PER_HOUR, pH] + [eval(f"observation.{var}.y[t]", {'observation': observation, 't': t}) for var in vars]


def parent_dir_and_name(file_path):
    """
    >>> file_path="a/b.c"
    >>> parent_dir_and_name(file_path)
    ('/root/.../a', 'b.c')
    :param file_path:
    :return:
    """
    return os.path.split(os.path.abspath(file_path))


def get_things_in_loc(in_path, just_files=True):
    """
    in_path can be file path or dir path.
    This function return a list of file paths
    in in_path if in_path is a dir, or within the 
    parent path of in_path if it is not a dir.
    just_files=False will let the function go recursively
    into the subdirs.
    """
    # TODO: check for file
    if not os.path.exists(in_path):
        print(str(in_path) + " does not exists!")
        return
    re_list = []
    if not os.path.isdir(in_path):
        in_path = parent_dir_and_name(in_path)[0]

    for name in os.listdir(in_path):
        name_path = os.path.abspath(os.path.join(in_path, name))
        if os.path.isfile(name_path):
            re_list.append(name_path)
        elif not just_files:
            if os.path.isdir(name_path):
                re_list += get_things_in_loc(name_path, just_files)
    return re_list


def normalize_spaces(space, max_space=None, min_space=None):
    """
    normalize each column of observation/action(e.g. Sugar feed rate) to be in [-1,1] such that it looks like a Box
    and space can be the whole original space (X by D) or just one row in the original space (D,)
    :param space: numpy array
    """
    assert not isinstance(space, list)
    if max_space is None:
        max_space = space.max(axis=0)
    if min_space is None:
        min_space = space.min(axis=0)
    gap = max_space - min_space
    full_sum = max_space + min_space
    return (2 * space - full_sum) / gap, max_space, min_space


def denormalize_spaces(space_normalized, max_space=None, min_space=None):
    """
    same as above, and space_normalized can be the whole normalized original space or just one row in the normalized space
    """
    assert not isinstance(space_normalized, list)
    if max_space is None:
        max_space = space_normalized.max(axis=0)
    if min_space is None:
        min_space = space_normalized.min(axis=0)
    gap = max_space - min_space
    full_sum = max_space + min_space
    return (space_normalized * gap + full_sum) / 2, max_space, min_space


class PenSimEnvGym(PenSimEnv, Env):
    def __init__(self, recipe_combo, fast=True, state_dim=9, action_dim=6, normalize=True):
        super(PenSimEnvGym, self).__init__(recipe_combo, fast=fast)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.state_dim,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))
        self._max_episode_steps = NUM_STEPS
        # ---- set by dataset or use predefined as you wish if applicable ----
        self.normalize = normalize
        self.max_observations = 1
        self.min_observations = -1
        self.max_actions = np.array([4100.0, 151.0, 36.0, 76.0, 1.2, 510.0])
        self.min_actions = np.array([0.0, 7.0, 21.0, 29.0, 0.5, 0.0])
        # there are 6 items: Discharge rate,Sugar feed rate,Soil bean feed rate,Aeration rate,Back pressure,Water injection/dilution
        # ---- set by dataset or use predefined as you wish if applicable ----

    def reset(self):
        _, x = super().reset()
        self.x = x
        self.k = 0
        observation = get_observation_data_reformed(x, 0)
        observation = np.array(observation, dtype=np.float32)
        if self.normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)

        return observation
    
    def step(self, action):
        """
        actions in action (list) are in the order [discharge, Fs, Foil,Fg, pressure, Fw]
        """
        action = np.array(action, dtype=np.float32)
        if self.normalize:
            action, _, _ = denormalize_spaces(action, self.max_actions, self.min_actions)
        self.k += 1 
        values_dict = self.recipe_combo.get_values_dict_at(self.k * STEP_IN_MINUTES)
        # served as a batch buffer below
        pensimpy_observation, x, yield_per_run, done = super().step(self.k, self.x, action[1], action[2], action[3], action[4], action[0], action[5], values_dict['Fpaa'])
        self.x = x
        new_observation = get_observation_data_reformed(x, self.k - 1)
        new_observation = np.array(new_observation, dtype=np.float32)
        if self.normalize:
            new_observation, _, _ = normalize_spaces(new_observation, self.max_observations, self.min_observations)

        return new_observation, yield_per_run, done, {}
        # state, reward, done, info in gym env term


class PeniControlData:
    """
    dataset class helper, mainly aims to mimic d4rl's qlearning_dataset format (which returns a dictionary).
    produced from PenSimPy generated csvs.
    """
    def __init__(self, load_just_a_file='', dataset_folder='examples/example_batches', delimiter=',', state_dim=9, action_dim=6) -> None:
        """
        :param dataset_folder: where all dataset csv files are living in
        """
        self.dataset_folder = dataset_folder
        self.delimiter = delimiter
        self.state_dim = state_dim
        self.action_dim = action_dim
        if load_just_a_file!= '':
            file_list = [load_just_a_file]
        else:
            file_list = get_things_in_loc(dataset_folder, just_files=True)
        self.file_list = file_list

    def load_file_list_to_dict(self, file_list, shuffle=True):
        file_list = file_list.copy()
        random.shuffle(file_list)
        dataset= {}
        observations = []
        actions = []
        next_observations = []
        rewards = []
        terminals = []
        for file_path in file_list:
            tmp_observations = []
            tmp_actions = []
            tmp_next_observations = []
            tmp_rewards = []
            tmp_terminals = []
            with codecs.open(file_path, 'r', encoding='utf-8') as fp:
                csv_reader = csv.reader(fp, delimiter=self.delimiter)
                next(csv_reader) 
                # get rid of the first line containing only titles
                for row in csv_reader:
                    observation = [row[0]] + row[7:-1] 
                    # there are 9 items: Time Step, pH,Temperature,Acid flow rate,Base flow rate,Cooling water,Heating water,Vessel Weight,Dissolved oxygen concentration
                    assert len(observation) == self.state_dim
                    action = [row[1], row[2], row[3], row[4], row[5], row[6]] 
                    # there are 6 items: Discharge rate,Sugar feed rate,Soil bean feed rate,Aeration rate,Back pressure,Water injection/dilution
                    assert len(action) == self.action_dim
                    reward = row[-1]
                    terminal = False
                    tmp_observations.append(observation)
                    tmp_actions.append(action)
                    tmp_rewards.append(reward)
                    tmp_terminals.append(terminal)
            tmp_terminals[-1] = True
            tmp_next_observations = tmp_observations[1:] + [tmp_observations[-1]]
            observations += tmp_observations
            actions += tmp_actions
            next_observations +=tmp_next_observations
            rewards += tmp_rewards
            terminals += tmp_terminals
        dataset['observations'] = np.array(observations, dtype=np.float32)
        dataset['actions'] = np.array(actions, dtype=np.float32)
        dataset['next_observations'] = np.array(next_observations, dtype=np.float32)
        dataset['rewards'] = np.array(rewards, dtype=np.float32)
        dataset['terminals'] = np.array(terminals, dtype=bool)
        self.max_observations = dataset['observations'].max(axis=0)
        self.min_observations = dataset['observations'].min(axis=0)
        dataset['observations'], _, _ = normalize_spaces(dataset['observations'], self.max_observations, self.min_observations)
        dataset['next_observations'], _, _ = normalize_spaces(dataset['next_observations'], self.max_observations, self.min_observations)
        self.max_actions = dataset['actions'].max(axis=0)
        self.min_actions = dataset['actions'].min(axis=0)
        dataset['actions'], _, _ = normalize_spaces(dataset['actions'], self.max_actions, self.min_actions) # passed in a normalized version.
        # self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))
        return dataset

    def get_dataset(self):
        return self.load_file_list_to_dict(self.file_list)

# a simple example
if __name__ == '__main__':

    recipe_dict = {FS: Recipe(FS_DEFAULT_PROFILE, FS),
                FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),
                FG: Recipe(FG_DEFAULT_PROFILE, FG),
                PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),
                DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
                WATER: Recipe(WATER_DEFAULT_PROFILE, WATER),
                PAA: Recipe(PAA_DEFAULT_PROFILE, PAA)}

    recipe_combo = RecipeCombo(recipe_dict=recipe_dict, filling_method=FillingMethod.BACKWARD)
    env = PenSimEnvGym(recipe_combo=recipe_combo)

    state = env.reset()
    load_just_a_file=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples/example_batches/random_batch_0.csv')
    dataset_obj = PeniControlData(load_just_a_file=load_just_a_file)
    if dataset_obj.file_list:
        print('Penicillin_Control_Challenge data correctly initialized.')
    else:
        raise ValueError("Penicillin_Control_Challenge data initialization failed.")
    dataset = dataset_obj.get_dataset()
    # ---- need to be set by dataset for normalization ----
    env.max_observations = dataset_obj.max_observations
    env.min_observations = dataset_obj.min_observations
    env.normalize = True
    # now, let's try to run for one epoch, use the actions cloned from a csv file. Note that the environment setup might be different.
    total_reward = 0.0
    for step in range(NUM_STEPS):
        state, reward, done, info = env.step(dataset['actions'][step].tolist())
        total_reward += reward
    print("your total reward is (by default, should be 2965.6227):", total_reward)
