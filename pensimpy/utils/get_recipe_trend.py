import numpy as np
import itertools


def get_recipe_trend(recipe_list):
    """
    Get the recipe trend data
    :param recipe:
    :param recipe_sp:
    :return:
    """
    recipe = [int(ele / 0.2) for ele in recipe_list[0]]
    recipe = [recipe[0]] + np.diff(recipe).tolist()
    recipe_sp = [[ele] for ele in recipe_list[1]]
    res_default = [x * y for x, y in zip(recipe, recipe_sp)]
    return list(itertools.chain(*res_default))[0:1150]
