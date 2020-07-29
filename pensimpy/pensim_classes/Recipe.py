from pensimpy.helper.get_recipe_trend import get_recipe_trend


class Recipe:
    def __init__(self, x):
        if isinstance(x, list):
            Fs_sp, Foil_sp, Fg_sp, pres_sp, discharge_sp, water_sp = self.split(x)
            Fs = [15, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 800, 1750]
            Foil = [20, 80, 280, 300, 320, 340, 360, 380, 400, 1750]
            Fg = [40, 100, 200, 450, 1000, 1250, 1750]
            pres = [62, 125, 150, 200, 500, 750, 1000, 1750]
            discharge = [500, 510, 650, 660, 750, 760, 850, 860, 950, 960, 1050, 1060, 1150, 1160, 1250, 1260, 1350,
                         1360, 1750]
            water = [250, 375, 750, 800, 850, 1000, 1250, 1350, 1750]
            self.discharge_trend = get_recipe_trend(discharge, discharge_sp)
            self.pres_trend = get_recipe_trend(pres, pres_sp)
            self.Fg_trend = get_recipe_trend(Fg, Fg_sp)
            self.Foil_trend = get_recipe_trend(Foil, Foil_sp)
            self.Fs_trend = get_recipe_trend(Fs, Fs_sp)
            self.water_trend = get_recipe_trend(water, water_sp)
        else:
            self.discharge_trend = get_recipe_trend(list(zip(*x['discharge']))[0], list(zip(*x['discharge']))[1])
            self.pres_trend = get_recipe_trend(list(zip(*x['pres']))[0], list(zip(*x['pres']))[1])
            self.Fg_trend = get_recipe_trend(list(zip(*x['Fg']))[0], list(zip(*x['Fg']))[1])
            self.Foil_trend = get_recipe_trend(list(zip(*x['Foil']))[0], list(zip(*x['Foil']))[1])
            self.Fs_trend = get_recipe_trend(list(zip(*x['Fs']))[0], list(zip(*x['Fs']))[1])
            self.water_trend = get_recipe_trend(list(zip(*x['water']))[0], list(zip(*x['water']))[1])

        PAA = [25, 200, 1000, 1500, 1750]
        PAA_sp = [5, 0, 10, 4, 0]
        self.PAA_trend = get_recipe_trend(PAA, PAA_sp)

    def run(self, t):
        t -= 1
        return self.Fs_trend[t], self.Foil_trend[t], self.Fg_trend[t], self.pres_trend[t], self.discharge_trend[t], \
               self.water_trend[t], self.PAA_trend[t]

    def split(self, x):
        Fs_len, Foil_len, Fg_len, pres_len, discharge_len, water_len = 21, 10, 7, 8, 20, 9
        recipe_Fs_sp = x[:Fs_len]
        recipe_Foil_sp = x[Fs_len:
                           Fs_len + Foil_len]
        recipe_Fg_sp = x[Fs_len + Foil_len:
                         Fs_len + Foil_len + Fg_len]
        recipe_pres_sp = x[Fs_len + Foil_len + Fg_len:
                           Fs_len + Foil_len + Fg_len + pres_len]
        recipe_discharge_sp = x[Fs_len + Foil_len + Fg_len + pres_len:
                                Fs_len + Foil_len + Fg_len + pres_len + discharge_len]
        recipe_water_sp = x[Fs_len + Foil_len + Fg_len + pres_len + discharge_len:
                            Fs_len + Foil_len + Fg_len + pres_len + discharge_len + water_len]

        return recipe_Fs_sp, recipe_Foil_sp, recipe_Fg_sp, recipe_pres_sp, recipe_discharge_sp, recipe_water_sp
