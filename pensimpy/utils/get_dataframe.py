import pandas as pd


def get_dataframe(batch_data, include_raman):
    df = pd.DataFrame(data={"Volume": batch_data.V.y,
                            "Penicillin Concentration": batch_data.P.y,
                            "Discharge rate": batch_data.Fremoved.y,
                            "Sugar feed rate": batch_data.Fs.y,
                            "Soil bean feed rate": batch_data.Foil.y,
                            "Aeration rate": batch_data.Fg.y,
                            "Back pressure": batch_data.pressure.y,
                            "Water injection/dilution": batch_data.Fw.y,
                            "Phenylacetic acid flow-rate": batch_data.Fpaa.y,
                            "pH": batch_data.pH.y,
                            "Temperature": batch_data.T.y,
                            "Acid flow rate": batch_data.Fa.y,
                            "Base flow rate": batch_data.Fb.y,
                            "Cooling water": batch_data.Fc.y,
                            "Heating water": batch_data.Fh.y,
                            "Vessel Weight": batch_data.Wt.y,
                            "Dissolved oxygen concentration": batch_data.DO2.y,
                            "Oxygen in percent in off-gas": batch_data.O2.y, })
    df = df.set_index([[t * 0.2 for t in range(1, 1151)]])

    df_raman = pd.DataFrame()
    if include_raman:
        wavenumber = batch_data.Raman_Spec.Wavenumber
        df_raman = pd.DataFrame(batch_data.Raman_Spec.Intensity, columns=wavenumber)
        df_raman = df_raman[df_raman.columns[::-1]]
        df_raman = df_raman.set_index([[t * 0.2 for t in range(1, 1151)]])
        return df, df_raman

    return df, df_raman
