# Brain dynamics and temporal trajectories during task and naturalistic processing

The work includes use of reservoir computing to capture dynamics in fMRI data. A preprint of this work can be found [here](https://www.biorxiv.org/content/early/2018/07/30/380402).

## Demo code

Code runs on a toy data set included in 
```
data/toy/data_Xy.mat
```

### Generate reservoir outputs

```
matlab code/matlab/toy/save_data/run_esn.m
```
### Train classifiers

For activation data, use
```
python3 code/python/toy/lr_Xy.py
```
For reservoir data, use
```
python3 code/python/toy/lr_ESN.py
```
Visualize results using
```
matlab code/matlab/toy/compare_ESN_avg_max.m
```


### Low dimensional representations

Obtain weights using
```
python3 code/python/toy/wts_PCA.py
```
Test low dimensional models using
```
python3 code/python/toy/lr_nDIM_ESN.py
```

### Importance values for brain regions

```
matlab code/matlab/toy/brain_importance.m
```

## Acknowledgments

* Reservoir computing code is from the [Modeling Intelligent Dynamical Systems research group](http://minds.jacobs-university.de/research/esnresearch/)