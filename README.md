Download data
wget https://storage.cloud.google.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz
____

Tasks

- Put this on the cluster
    - Multi-gpu
    - Parallel environment steps? (only if this becomes a bottleneck)
- Model/Optimization experiments
    - Hyperparam search
    - Use advantage functions?
    - Actor/Critic?
    - PPO?
- Optimizing exploration
- Scale up to whole dataset
- Add curriculum learning
- Make the graphs more compact (to reduce sequence length)?
- dropout only during training?
- Provide access to a test set within the environment


Optional future work:
    - Extend the dataset by composing all problem types more fully, then systematically study generalization of different sorts
    - Use LM on top of outputs from program synthesis to improve usability/robustness
        - When using a LM on top of program synthesis one should extract all intermediate results
            - Does this require a different training objective?
    