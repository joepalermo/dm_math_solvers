Seeded with code from a prior project by Helen Ngo, Joseph Palermo and Michael Jia, with support from Rayhane Mama (https://github.com/mathemakitten/transformers-mathematics).

Download data
wget https://storage.cloud.google.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz

Docker commands
nvidia-docker run -it -p 6006:6006 -v /home/jpalermo:/home/ tensorflow/tensorflow:latest-gpu /bin/bash
nvidia-docker run -it -p 6006:6006 -v /home/jpalermo:/home/ math /bin/bash
docker run -gpus all -it -p 6006:6006 -v /home/jpalermo:/home/ math /bin/bash

tensorboard --logdir=/home/dm_math_solvers/experiment_results --bind_all
tensorboard --logdir=logs/gradient_tape --bind_all

____

Tasks

- Gather corpus of observations, then run BPE to get encoder/decoder mappings of fixed size
    - Implement encoder function which both encodes observations and also 0-pads to a max observation size  
- Model and optimization
    1. Implement a VPG training loop and connect it to vanilla Transformer model
        - If this fails to do well, then here's a list of things to try:
            - variants on VPG (see spinningup)
            - gated transformer XL
            - PPO
- Strategy to alternate between training and running the policy
    1. Gather train data
    2. Train the policy on the most recent data
    3. Eval the policy
    4. (optional) Determine curriculum for the next round based on eval performance across modules
- Evaluate on test set

Optional future work:
    - Extend the dataset by composing all problem types more fully, then systematically study generalization of different sorts
    - Use LM on top of outputs from program synthesis to improve usability/robustness
        - When using a LM on top of program synthesis one should extract all intermediate results
            - Does this require a different training objective?
    