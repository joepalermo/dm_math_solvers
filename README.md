# dm_math_solvers

This repo contains the code used to produce the experimental results for: <insert paper link here>.

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
evaluate.py
```