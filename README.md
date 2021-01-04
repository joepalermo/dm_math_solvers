Download data
wget https://storage.cloud.google.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz
____

Tasks


- Make the graphs more compact (to reduce sequence length)?
- Setup learning script
    - Build train/val sets 
        - Load serialized rewarded trajectories
        - Convert them into train examples
        - Merge them with a prescribed ratio of correct to incorrect
        - split train/val 
    - Sample batches
- dropout only during training?
- Provide access to a test set within the environment  


Optional future work:
    - Extend the dataset by composing all problem types more fully, then systematically study generalization of different sorts
    - Use LM on top of outputs from program synthesis to improve usability/robustness
        - When using a LM on top of program synthesis one should extract all intermediate results
            - Does this require a different training objective?
    