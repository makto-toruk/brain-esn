TOM_meta

data: 
- 100 subjects
- ~ 5 blocks per subject 
- task conditions: 'social' and 'rand'
- location: /data/TOM_meta/data_Xy.mat

STEP 1: generate reservoir data from activation data
- run /code/bash/TOM_meta/mat_generate_esn.slurm
- in folder, 'sbatch mat_generate_esn.slurm'

STEP 2: train classifiers for activation and reservoir data
- run /code/bash/TOM_meta/py_lr.slurm
- in folder, 'sbatch py_lr.slurm'

STEP 3: generate figure for results
- run /code/bash/TOM_meta/mat_esn_compare.slurm

