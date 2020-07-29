from pensimpy.helper.get_recipe_trend import get_recipe_trend


class Recipe:
    def __init__(self, x):
        self.Fs_trend = get_recipe_trend(self.process_recipe(x['Fs']))
        self.Foil_trend = get_recipe_trend(self.process_recipe(x['Foil']))
        self.Fg_trend = get_recipe_trend(self.process_recipe(x['Fg']))
        self.pres_trend = get_recipe_trend(self.process_recipe(x['pres']))
        self.discharge_trend = get_recipe_trend(self.process_recipe(x['discharge']))
        self.water_trend = get_recipe_trend(self.process_recipe(x['water']))

        PAA = [5.0, 40.0, 200.0, 230.0]
        PAA_sp = [5, 0, 10, 4]
        self.PAA_trend = get_recipe_trend([PAA, PAA_sp])

    def run(self, t):
        t -= 1
        return self.Fs_trend[t], self.Foil_trend[t], self.Fg_trend[t], self.pres_trend[t], self.discharge_trend[t], \
               self.water_trend[t], self.PAA_trend[t]

    def process_recipe(self, x):
        x = list(dict(sorted(x, key=lambda ele: ele[0])).items())
        return [list(zip(*x))[0], list(zip(*x))[1]]
