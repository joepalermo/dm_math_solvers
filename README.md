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

- Formalize a few problems from each module
    - [x] buildup a library of operators
    - [x] Turn that into RL env (Gym)
    - [-] Define curriculum
        - [x] First cut solution: start with uncomposed, then composed, but split composed by # sentences per problem
        - Choose strategy:
            - automatically (treat as a k-armed bandit)
            - manually? (define difficulty ordering on the modules, define master threshold, define simple strategy for mixing in mastered problems to prevent forgetting)
- write annotation script (to add the formal elements into the problem description)
    - Use the script to generated a separate processed dataset which is used to construct RL env- Model and optimization
- Decide on problem framing, NN architecture, and training procedure
    - Decide on input:
        - raw text with annotations + intermediate (incomplete) graph as text
        - raw text with annotations + intermediate (incomplete) graph as graph
    - output: operator probability vector (i.e. policy)
    - implementation: vanilla transformer or gated transformer XL (more stable for RL?)
    - optimization: PPO?
- Strategy to alternate between training and running the policy
    1. Run the policy with search (e.g. BFS) and save a subset of actions/programs in a replay buffer
    2. Train the policy from the replay buffer
- Evaluate on test set

Optional future work:
    - Extend the dataset
    - Systematically study generalization of different sorts
    - Use LM on top of outputs from program synthesis to improve usability/robustness
        - to use LM on top of NN will need to run program for various outputs, might need to make NN good at ignoring irrelevant info (can regularize by introducing garbage into problem statements)
    