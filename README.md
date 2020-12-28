Download data
wget https://storage.cloud.google.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz
____

Tasks

- Provide access to a test set within the environment  
- Model and optimization
    1. Get Ray RLLIB working well
        - Add an embedding layer (why does SL dissapear in the shape after embedding?)
            - Try a radically simpler model such as a vanilla transformer?
        - Add type-constraint masking
        - Try scaling the environment
    2. Implement a VPG training loop and connect it to vanilla Transformer model
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
    