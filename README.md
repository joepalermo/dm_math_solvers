Download data
wget https://storage.cloud.google.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz
____

What does end-state look like?

- Design a procedure for testing new RL algorithms with simple environments

- Design interfaces to the database of rewarded graphs so that batches can be constructed from it
    - 1 => flatten it into list of steps that can be shuffled and used for training
    - 2 => construct a mapping from module-difficulty to a list of steps

- How to use multi-GPUs for sampling
    - 2 alternatives:
        - sampling in parallel (one set of environments associated to each GPU)
        - sample sequentially but with huge batch size (sharing all GPUs for one big batch) 

- Curriculum learning: How to change the data distribution over time from simple to complex
    - If performance of (module, difficulty) pair goes over a threshold then switch from 'learning' mode to 'maintenance' mode
    - 'learning' mode implies a larger fraction of each batch would comprise that (module, difficulty) pair
    - 'maintenance' mode  implies a smaller fraction
    - Teacher-student curriculum learning?
    
- Procedures for adding composed operators back into the action space

- Adaptive modification of the exploration/exploitation tradeoff
    - how does this change?

- Optional future work:
    - Can we generalize the operators into a general purpose functional programming language which if useful for other problems too?
    - Extend the dataset by composing all problem types more fully, then systematically study generalization of different sorts
    - Use LM on top of outputs from program synthesis to improve usability/robustness
        - When using a LM on top of program synthesis one should extract all intermediate results
            - Does this require a different training objective?
    