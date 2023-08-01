# Time Series Data Cleaning under expressive constraints on both row and column dependencies.

## Set up

```python
# requirements: 
python >= 3.9
numpy
pandas
matplotlib
time
pickle
scipy
statsmodels
sklearn
geatpy (https://github.com/geatpy-dev/geatpy)
```

## Implementation

​	We implement the discovery of quality constraints in $DQDiscovery$. Specifically, we implement the discovery of the quality constraints expressing complex between sequences, i.e. conditional regression rules ($CRR$) in $LearnCRR$ in $DQDiscovery$. The discovery of the quality constraints expressing temporal dependencies, e.g. speed constraints($SC$), acceleration constraints($AC$) and variance constraints($VC$) in $LearnTD$.

​	Based on what's mentioned above, we implement our proposed method  $Clean4MTS$ and its variants, and benchmark time series data cleaning methods in $TSRepair$. 

