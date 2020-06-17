from pensimpy.helper.get_recipe_trend import get_recipe_trend
import numpy as np


class Recipe:
    def __init__(self):
        # recipes
        Fs = [15, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 800, 1750]
        Fs_sp = [8, 15, 30, 75, 150, 30, 37, 43, 47, 51, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80]
        Fs_sp = [ele * (1 + np.random.randint(-5, 5) / 100) for ele in Fs_sp]
        self.Fs_trend = get_recipe_trend(Fs, Fs_sp)

        Foil = [20, 80, 280, 300, 320, 340, 360, 380, 400, 1750]
        Foil_sp = [22, 30, 35, 34, 33, 32, 31, 30, 29, 23]
        Foil_sp = [ele * (1 + np.random.randint(-5, 5) / 100) for ele in Foil_sp]
        self.Foil_trend = get_recipe_trend(Foil, Foil_sp)

        Fg = [40, 100, 200, 450, 1000, 1250, 1750]
        Fg_sp = [30, 42, 55, 60, 75, 65, 60]
        Fg_sp = [ele * (1 + np.random.randint(-5, 5) / 100) for ele in Fg_sp]
        self.Fg_trend = get_recipe_trend(Fg, Fg_sp)

        pres = [62, 125, 150, 200, 500, 750, 1000, 1750]
        pres_sp = [0.6, 0.7, 0.8, 0.9, 1.1, 1, 0.9, 0.9]
        pres_sp = [ele * (1 + np.random.randint(-5, 5) / 100) for ele in pres_sp]
        self.pres_trend = get_recipe_trend(pres, pres_sp)

        discharge = [500, 510, 650, 660, 750, 760, 850, 860, 950, 960, 1050, 1060, 1150, 1160, 1250, 1260, 1350, 1360,
                     1750]
        discharge_sp = [0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 0]
        discharge_sp = [ele * (1 + np.random.randint(-5, 5) / 100) for ele in discharge_sp]
        self.discharge_trend = get_recipe_trend(discharge, discharge_sp)

        water = [250, 375, 750, 800, 850, 1000, 1250, 1350, 1750]
        water_sp = [0, 500, 100, 0, 400, 150, 250, 0, 100]
        water_sp = [ele * (1 + np.random.randint(-5, 5) / 100) for ele in water_sp]
        self.water_trend = get_recipe_trend(water, water_sp)

        PAA = [25, 200, 1000, 1500, 1750]
        PAA_sp = [5, 0, 10, 4, 0]
        PAA_sp = [ele * (1 + np.random.randint(-5, 5) / 100) for ele in PAA_sp]
        self.PAA_trend = get_recipe_trend(PAA, PAA_sp)
