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

- [x] Formalize a few problems from each module
    - [x] buildup a library of operators
    - use typed operators to reduce the combinatorial explosion?
- [x] Create RL env (Gym)
- [-] Define curriculum
    - [x] Difficulty ordering: start with uncomposed, then composed with 2 sentences, composed with 3 sentences, etc...
    - Choose strategy:
        - automatically (treat as a k-armed bandit)
        - [-] manually? (define master threshold and define simple strategy for mixing in mastered problems to prevent forgetting)
- [x] write annotation script (to add the formal elements into the problem description)
- Model and optimization
    - Decide on how to represent incomplete compute graphs:
        - text
        - graph
    - output: action probability vector (i.e. policy)
    - implementation: vanilla transformer or gated transformer XL (more stable for RL?)
    - optimization: PPO?
- Strategy to alternate between training and running the policy
    1. Gather train data: Run the policy and save a subset of actions/programs in a replay buffer
        a. with deterministic search (e.g. BFS)
        b. by random sampling from the policy vectors
    2. Train the policy from the replay buffer
    3. Eval the policy
    4. (optional) Determine curriculum for the next round based on eval performance across modules
- Evaluate on test set

Optional future work:
    - Extend the dataset by composing all problem types more fully, then systematically study generalization of different sorts
    - Use LM on top of outputs from program synthesis to improve usability/robustness
        - When using a LM on top of program synthesis one should extract all intermediate results
            - Does this require a different training objective?
    