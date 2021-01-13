Download data
wget https://storage.cloud.google.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz
____

Tasks

Clear Tasks
- Breakdown validation performance by module_name and difficulty
    
Clear Benefit to Performance
- dropout only during training?
- Add padding mask
- Make the graphs more compact?
    - Reducing the length of the description

Experiments
- Scale up to whole dataset
    - Any module type that doesn't converge easily, train them in isolation to figure out why 
    
Open Ended Improvements
- Add curriculum learning
- Hyperparam search
- Optimizing exploration
- Use advantage functions?
    - Actor/Critic?
    - PPO?
- Provide access to a test set within the environment

Less Important for now
- Multi-gpu
- Parallel environment steps? (only if this becomes a bottleneck)

Optional future work:
    - Extend the dataset by composing all problem types more fully, then systematically study generalization of different sorts
    - Use LM on top of outputs from program synthesis to improve usability/robustness
        - When using a LM on top of program synthesis one should extract all intermediate results
            - Does this require a different training objective?
    