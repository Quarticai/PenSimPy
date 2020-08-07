from collections import OrderedDict
from itertools import chain
from pensimpy.constants import MINUTES_PER_HOUR


class Setpoint:
    """
    Class for defining the setpoint which contributes process variables.
    """
    UOT_HOUR = "hour"
    UOT_MINUTE = "minute"

    def __init__(self, time_until, value, unit_of_time=UOT_HOUR):
        assert unit_of_time in {self.UOT_HOUR, self.UOT_MINUTE}, "Unknown unit of time"
        if unit_of_time == self.UOT_HOUR:
            assert (time_until * MINUTES_PER_HOUR) % 1 == 0, (f"{time_until} {self.UOT_HOUR}s is invalid. "
                                                                   f"The given time must be a multiple of 1 minute"
                                                                   " after being converted from hours")
        self._time_until = time_until
        self._unit_of_time = unit_of_time  # should never be modified once been set
        self._value = value

    def get_time_until_in_minutes(self):
        if self.unit_of_time == self.UOT_MINUTE:
            return self._time_until
        else:
            return int(self._time_until * MINUTES_PER_HOUR)

    @property
    def time(self):
        return self._time_until

    @property
    def value(self):
        return self._value

    @property
    def unit_of_time(self):
        return self._unit_of_time

    def is_after(self, another):
        return self.get_time_until_in_minutes() > another.get_time_until_in_minutes()


class ProcessVariable:
    """
    Process variables feed the simulation and realize the Sequential Batch Control. And it support adding new setpoints
    and modifying current setpoints.
    """
    class Decorator:
        @staticmethod
        def populate_setpoint_value_lookup(func):
            def wrapper(*args, **kwargs):
                func(*args, **kwargs)
                values = ProcessVariable.flatten(args[0]._setpoints_lookup)
                args[0]._setpoint_value_lookup = dict(zip(range(1, len(values) + 1), values))
            return wrapper

    def __init__(self, name):
        self.name = name
        self._setpoints_lookup = OrderedDict()
        self._setpoint_value_lookup = {}
        self._last_added_setpoint = None
        self._setpoint_time_min = None
        self._setpoint_time_max = None

    @Decorator.populate_setpoint_value_lookup
    def add_setpoint(self, setpoint):
        time_key = setpoint.get_time_until_in_minutes()

        if len(self._setpoints_lookup) == 0:
            self._setpoints_lookup[time_key] = setpoint
        else:
            if time_key in self._setpoints_lookup:
                raise ValueError("A setpoint with the same time has been added. The time of setpoint you're "
                                 "adding must be "
                                 "greater the time of the last one you've added. To override a existing setpoint, "
                                 "please use update_setpoint() instead")

            if setpoint.is_after(self._last_added_setpoint):
                self._setpoints_lookup[time_key] = setpoint
            else:
                ValueError(
                    "The time of setpoint you're adding must be greater the time of the last one you've added.")

        self._last_added_setpoint = setpoint

        if self._setpoint_time_min is None:
            self._setpoint_time_min = time_key
            self._setpoint_time_max = time_key
        else:
            self._setpoint_time_max = time_key

    @Decorator.populate_setpoint_value_lookup
    def update_setpoint(self, setpoint):
        time_key = setpoint.get_time_until_in_minutes()

        if time_key in self._setpoints_lookup:
            self._setpoints_lookup[time_key] = setpoint
        else:
            raise KeyError("Time key not found")

    @Decorator.populate_setpoint_value_lookup
    def add_setpoints(self, *setpoints):
        for sp in setpoints:
            self.add_setpoint(sp)

    def update_setpoints(self, *setpoints):
        for sp in setpoints:
            self.update_setpoint(sp)

    @staticmethod
    def flatten(setpoints_lookup):
        setpoints = list(setpoints_lookup.values())
        time_deltas = [sp.get_time_until_in_minutes() for sp in setpoints[:1]]
        time_deltas.extend([sp.get_time_until_in_minutes() - prev_sp.get_time_until_in_minutes()
                            for sp, prev_sp in zip(setpoints[1:], setpoints[:-1])])

        return list(chain.from_iterable([[sp.value] * delta for sp, delta in zip(setpoints, time_deltas)]))

    def __len__(self):
        return len(self._setpoints_lookup)

    def get_setpoint_value_at(self, t_minute, forward_fill=True):
        if t_minute > self._setpoint_time_max and forward_fill is True:
            return self._setpoint_value_lookup[self._setpoint_time_max]
        try:
            value = self._setpoint_value_lookup[t_minute]
            return value
        except KeyError as e:
            raise RuntimeError(f'Unable to find value at {t_minute} minutes for process '
                               f'variable "{self.name}"') from e


class ManualProcessVariable(ProcessVariable):
    pass


class PIDControlledProcessVariable(ProcessVariable):
    pass


class Recipe:
    """
    Recipe class for getting the default recipes and manually updating setpoints in recipe.
    """
    FS = "Fs"
    FOIL = "Foil"
    FG = "Fg"
    PRES = "pres"
    DISCHARGE = "discharge"
    WATER = "water"
    PAA = "paa"

    DEFAULT_OUTPUT_ORDER = [FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA]

    def __init__(self, **kwargs):
        self._process_variables_lookup = OrderedDict([(name, kwargs[name])
                                                      for name in self.DEFAULT_OUTPUT_ORDER])

    @classmethod
    def get_default(cls):
        fs = ManualProcessVariable(name="fs")
        fs.add_setpoints(
            Setpoint(time_until=3, value=8),
            Setpoint(time_until=12, value=15),
            Setpoint(time_until=16, value=30),
            Setpoint(time_until=20, value=75),
            Setpoint(time_until=24, value=150),
            Setpoint(time_until=28, value=30),
            Setpoint(time_until=32, value=37),
            Setpoint(time_until=36, value=43),
            Setpoint(time_until=40, value=47),
            Setpoint(time_until=44, value=51),
            Setpoint(time_until=48, value=57),
            Setpoint(time_until=52, value=61),
            Setpoint(time_until=56, value=65),
            Setpoint(time_until=60, value=72),
            Setpoint(time_until=64, value=76),
            Setpoint(time_until=68, value=80),
            Setpoint(time_until=72, value=84),
            Setpoint(time_until=76, value=90),
            Setpoint(time_until=80, value=116),
            Setpoint(time_until=160, value=90),
            Setpoint(time_until=230, value=80),
        )

        foil = ManualProcessVariable(name="foil")
        foil.add_setpoints(
            Setpoint(time_until=4, value=22),
            Setpoint(time_until=16, value=30),
            Setpoint(time_until=56, value=35),
            Setpoint(time_until=60, value=34),
            Setpoint(time_until=64, value=33),
            Setpoint(time_until=68, value=32),
            Setpoint(time_until=72, value=31),
            Setpoint(time_until=76, value=30),
            Setpoint(time_until=80, value=29),
            Setpoint(time_until=230, value=23),
        )

        fg = ManualProcessVariable(name="fg")
        fg.add_setpoints(
            Setpoint(time_until=8, value=30),
            Setpoint(time_until=20, value=42),
            Setpoint(time_until=40, value=55),
            Setpoint(time_until=90, value=60),
            Setpoint(time_until=200, value=75),
            Setpoint(time_until=230, value=65)
        )

        pres = ManualProcessVariable(name="pres")
        pres.add_setpoints(
            Setpoint(time_until=12.4, value=0.6),
            Setpoint(time_until=25, value=0.7),
            Setpoint(time_until=30, value=0.8),
            Setpoint(time_until=40, value=0.9),
            Setpoint(time_until=100, value=1.1),
            Setpoint(time_until=150, value=1),
            Setpoint(time_until=200, value=0.9),
            Setpoint(time_until=230, value=0.9)
        )

        discharge = ManualProcessVariable(name="discharge")
        discharge.add_setpoints(
            Setpoint(time_until=100, value=0),
            Setpoint(time_until=102, value=4000),
            Setpoint(time_until=130, value=0),
            Setpoint(time_until=132, value=4000),
            Setpoint(time_until=150, value=0),
            Setpoint(time_until=152, value=4000),
            Setpoint(time_until=170, value=0),
            Setpoint(time_until=172, value=4000),
            Setpoint(time_until=190, value=0),
            Setpoint(time_until=192, value=4000),
            Setpoint(time_until=210, value=0),
            Setpoint(time_until=212, value=4000),
            Setpoint(time_until=230, value=0)
        )

        water = ManualProcessVariable(name="water")
        water.add_setpoints(
            Setpoint(time_until=50, value=0),
            Setpoint(time_until=75, value=500),
            Setpoint(time_until=150, value=100),
            Setpoint(time_until=160, value=0),
            Setpoint(time_until=170, value=400),
            Setpoint(time_until=200, value=150),
            Setpoint(time_until=230, value=250)
        )

        paa = PIDControlledProcessVariable(name="paa")
        paa.add_setpoints(
            Setpoint(time_until=5, value=5),
            Setpoint(time_until=40, value=0),
            Setpoint(time_until=200, value=10),
            Setpoint(time_until=230, value=4),
        )

        return Recipe(Fs=fs, Foil=foil, Fg=fg, pres=pres, discharge=discharge, water=water, paa=paa)

    def get_values_at(self, t_minute, forward_fill=True, output_order=None):
        output_order = output_order or self.DEFAULT_OUTPUT_ORDER
        return [self._process_variables_lookup[process_variable_name].get_setpoint_value_at(t_minute, forward_fill)
                for process_variable_name in output_order]

    def update_process_variable_setpoints(self, process_variable_name, *setpoints):
        if process_variable_name in self._process_variables_lookup:
            self._process_variables_lookup[process_variable_name].update_setpoints(*setpoints)
        else:
            raise KeyError("Process variable name not found")
