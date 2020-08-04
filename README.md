![alt text](figures/logo_light.png "Logo Title Text 1")

**Status:** Maintenance (expect bug fixes and updates)
# PenSimPy ![](https://img.shields.io/badge/python-3.6.8-orange) ![](https://img.shields.io/badge/c%2B%2B-11-blue)
PenSimPy is a python version of [IndPenSim](http://www.industrialpenicillinsimulation.com/), which simulates the industrial penicillin
yield process. Additionally, the PenSimPy is based on c++ to solve the ODE functions so as to achieve a faster performance. 
Basic batch data and Raman simulation data can be generated in Pandas dataframes with different random seeds. A conventionally used 
Sequential Batch Control strategy is realized and all the recipe's feed can be manually updated as setpoints by the user. Further, 
we incorporate the Reinforcement Learning to improve the penicillin gain and present this as example code.
Also, a web application based on PenSimPy can be found [here](http://quartic.ddns.net:8000/).

#### How to install
```
pip install pensimpy
```
#### How to use
Sequential batch control example can be found here:
```
from pensimpy import PenSimEnv

# Provide recipes for Fs, Foil, Fg, pres, discharge, water
# Recipe time range: 0 < t <= 230
setpoints_dict = {'water': [(100, 11)]}

env = PenSimEnv()
df, df_raman = env.get_batches(random_seed=1, setpoints=setpoints_dict, include_raman=False)
```

