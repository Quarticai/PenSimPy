![alt text](figures/logo_light.png "Logo Title Text 1")

**Status:** Maintenance (expect bug fixes and updates)
# PenSimPy ![](https://img.shields.io/badge/python-3.6.8-orange) ![](https://img.shields.io/badge/c%2B%2B-11-blue)
PenSimPy is a python version of [IndPenSim](http://www.industrialpenicillinsimulation.com/), which simulates the industrial penicillin
yield process. Additionally, the PenSimPy is based on c++ to solve the ODE functions so as to achieve a faster performance. 
Basic batch data and Raman simulation data can be generated in Pandas dataframes with different random seeds. A conventionally used 
Sequential Batch Control strategy is realized and all the recipe's feed can be manually updated as setpoints by the user. Further, 
we incorporate the Reinforcement Learning to improve the penicillin gain and present this as example code.
Also, a web application based on PenSimPy can be found [here](http://quartic.ddns.net:8000/).

Installation
============
```
pip install pensimpy
```
Examples
============
See the `examples` directory
- run `examples/batch_generation_example.py <https://github.com/Quarticai/PenSimPy/blob/refactor_opensource/pensimpy/examples/batch_generation_example.py>` to realize 
Sequential Batch Control with modified recipe feed

