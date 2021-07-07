# dm_math_solvers

This repo contains the code used to produce the experimental results for: <insert paper link here>.

To use the the RL environment (math_prog_synth_env) for your own research please see the following repo for code and setup instructions: https://github.com/JohnnyYeeee/math_prog_synth_env

## Setup

- Download data (wget https://storage.cloud.google.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz)
- Untar it and put it in the root directory of the repo

## Training

- Note that in this repo environment and training hyperparameters are controlled by hparams.cfg
- Initialize the replay buffer: 

```python
python fill_trajectory_cache.py
```

- Train model:

```python
python train.py
```

- Evaluate trained model on test problems:

```python
python evaluate.py
```