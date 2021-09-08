import pandas as pd
import matplotlib.pyplot as plt

good_dfs = pd.read_csv('good_dfs.csv', index_col=None)
print(good_dfs)
bad_dfs = pd.read_csv('bad_df.csv', index_col=None)
print(bad_dfs)
df = pd.concat([good_dfs, bad_dfs])
print(df)
df = df.reset_index()
cols = ['Volume', 'Penicillin Concentration', 'Discharge rate', 'Sugar feed rate', 'Soil bean feed rate',
        'Aeration rate', 'Back pressure', 'Water injection/dilution', 'Phenylacetic acid flow-rate', 'pH',
        'Temperature', 'Acid flow rate', 'Base flow rate', 'Cooling water', 'Heating water', 'Vessel Weight',
        'Dissolved oxygen concentration', 'Oxygen in percent in off-gas', 'peni_yield', 'label', 'f1_score', 'accuracy']
df = df[cols]
df['Datetime'] = pd.date_range(start='1/1/2021', periods=len(df), freq='12min')
print(df)
df.to_csv('demo_df.csv', index=None)

plt.figure(1)
plt.subplot(2, 1, 1)
plt.title('f1_score')
plt.plot(df['f1_score'])
plt.subplot(2, 1, 2)
plt.title('accuracy')
plt.plot(df['accuracy'])
plt.tight_layout()
plt.show()


