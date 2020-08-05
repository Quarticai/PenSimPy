from pensimpy.utils.utils import get_recipe_trend


class Recipe:
    SETPOINTS = {
        'Fs': [(3.0, 8), (12.0, 15), (16.0, 30), (20.0, 75), (24.0, 150), (28.0, 30), (32.0, 37), (36.0, 43), (40.0, 47), (44.0, 51), (48.0, 57), (52.0, 61), (56.0, 65), (60.0, 72), (64.0, 76), (68.0, 80), (72.0, 84), (76.0, 90), (80.0, 116), (160.0, 90), (230.0, 80)],
        'Foil': [(4.0, 22), (16.0, 30), (56.0, 35), (60.0, 34), (64.0, 33), (68.0, 32), (72.0, 31), (76.0, 30), (80.0, 29), (230.0, 23)],
        'Fg': [(8.0, 30), (20.0, 42), (40.0, 55), (90.0, 60), (200.0, 75), (230.0, 65)],
        'pres': [(12.4, 0.6), (25.0, 0.7), (30.0, 0.8), (40.0, 0.9), (100.0, 1.1), (150.0, 1), (200.0, 0.9), (230.0, 0.9)],
        'discharge': [(100.0, 0), (102.0, 4000), (130.0, 0), (132.0, 4000), (150.0, 0), (152.0, 4000), (170.0, 0), (172.0, 4000), (190.0, 0), (192.0, 4000), (210.0, 0), (212.0, 4000), (230.0, 0)],
        'water': [(50.0, 0), (75.0, 500), (150.0, 100), (160.0, 0), (170.0, 400), (200.0, 150), (230.0, 250)]}

    def __init__(self, setpoints):
        if setpoints is not None:
            for k, v in setpoints.items():
                if k not in {'Fs', 'Foil', 'Fg', 'pres', 'discharge', 'water'}:
                    raise ValueError(f"{k} id not a correct key. "
                                     f"Valid options are Fs, Foil, Fg, pres, discharge and water")
                if max(v, key=lambda ele: ele[0])[0] > 230 or min(v, key=lambda ele: ele[0])[0] < 0:
                    raise ValueError("Recipe time exceeds range, should be greater than 0 and less than 230 [H]")
                self.SETPOINTS[k] += v

        self.Fs_trend = get_recipe_trend(self.process_recipe(self.SETPOINTS['Fs']))
        self.Foil_trend = get_recipe_trend(self.process_recipe(self.SETPOINTS['Foil']))
        self.Fg_trend = get_recipe_trend(self.process_recipe(self.SETPOINTS['Fg']))
        self.pres_trend = get_recipe_trend(self.process_recipe(self.SETPOINTS['pres']))
        self.discharge_trend = get_recipe_trend(self.process_recipe(self.SETPOINTS['discharge']))
        self.water_trend = get_recipe_trend(self.process_recipe(self.SETPOINTS['water']))

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
