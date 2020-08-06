class Channel:
    """
    Class attributes for each var in batch data.
    """

    def __init__(self, **kwargs):
        self.name = kwargs.get('name')
        self.y_unit = kwargs.get('y_unit')
        self.t_unit = kwargs.get('t_unit')
        if kwargs.get('time') is not None:
            self.t = kwargs['time'].T.tolist()[0]

        if kwargs.get('value') is not None:
            self.y = kwargs['value'].T.tolist()[0]

        if kwargs.get('Wavenumber') is not None:
            self.Wavenumber = kwargs['Wavenumber'].T.tolist()[0]

        if kwargs.get('Intensity') is not None:
            self.Intensity = kwargs['Intensity']