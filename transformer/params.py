import os
from datetime import datetime as dt

current_time = dt.now().strftime('%Y%m%d_%H_%M-')


class TransformerParams:
    def __init__(self):
        # self.experiment_dir = os.path.join('experiment_results', '20201120_11_38-transformer_testing')
        self.experiment_dir = os.path.join('experiment_results', current_time + 'transformer_testing')
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.tb_logdir = os.path.join(self.experiment_dir, 'tensorboard')
        self.gpu_id = "0"

        self.questions_max_length = 160
        self.answer_max_length = 32
        self.predicted_output_length = self.answer_max_length - 1
        self.vocab_size = 64
        self.padding_token = 0
        self.start_token = 1
        self.end_token = 2

        self.num_examples = 1200000
        self.batch_size = 128
        self.batches_per_inspection = 1000
        self.min_epochs_until_checkpoint = 1
        self.num_epochs = 15
        self.p_val = 0.01

        self.learning_rate = 6e-4
        self.max_context = 160
        self.is_training = True
        self.embedding_dim = 512
        self.d_model = 512
        self.num_heads = 4
        self.dff = 512
        self.ffn_expansion = 4
        self.dropout = 0.1
        self.attention_dropout = 0.05
        self.num_layers = 1  # Todo: split this into encoder and decoder layers