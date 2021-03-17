import math
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Softmax, LSTM
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from hparams import HParams
hparams = HParams.get_hparams_by_name('rl_math')


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class DenseBlock(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(DenseBlock, self).__init__()
        self.dropout = torch.nn.Dropout(hparams.train.dropout)
        self.dense = torch.nn.Linear(dim_in, dim_out)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = self.relu(x)
        return x


class TransformerEncoderModel(torch.nn.Module):
    def __init__(self, num_question_tokens, num_outputs, question_padding_token, action_padding_token, device):
        super().__init__()
        torch.nn.Module.__init__(self)
        self.num_question_tokens = num_question_tokens
        self.num_outputs = num_outputs
        self.question_padding_token = question_padding_token
        self.action_padding_token = action_padding_token
        self.num_action_tokens = num_outputs + 1  # each output has one id, so the +1 is for the padding token
        self.max_grad_norm = hparams.train.max_grad_norm
        self.batch_size = hparams.train.batch_size
        self.epsilon = hparams.train.epsilon
        # sanity check padding tokens
        assert self.action_padding_token == num_question_tokens + num_outputs
        assert self.question_padding_token == num_question_tokens - 1

        # define tunable layers -------------------
        self.token_embedding = torch.nn.Embedding(self.num_question_tokens + self.num_action_tokens, hparams.model.nhid)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=hparams.model.nhid, nhead=hparams.model.nhead), hparams.model.nlayers
        )
        self.dense_block_1 = DenseBlock(hparams.model.nhid, hparams.model.nhid)
        self.dense_2 = torch.nn.Linear(hparams.model.nhid, num_outputs)

        # define non-tunable layers -------------------
        self.pos_encoder = PositionalEncoding(hparams.model.nhid, hparams.train.dropout)
        self.dropout = torch.nn.Dropout(hparams.train.dropout)
        self.relu = torch.nn.ReLU()

        # other
        self.device = device
        self.to(device)

        # set optimization
        self.optimizer = torch.optim.Adam(self.parameters(), lr=hparams.train.lr,
                                          weight_decay=hparams.train.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max',
                                                                    factor=hparams.train.factor,
                                                                    patience=hparams.train.patience, threshold=0.001,
                                                                    threshold_mode='rel', cooldown=0,
                                                                    min_lr=hparams.train.min_lr, eps=1e-08,
                                                                    verbose=False)
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=hparams.train.max_lr,
        #                                                      total_steps=hparams.train.total_steps,
        #                                                      final_div_factor=hparams.train.final_div_factor)

    def forward(self, question_tokens, action_tokens):
        # question_tokens: (BS, max_question_length), action_tokens: (BS, max_num_actions)
        # question model --------------
        # embed the tokens
        combined_tokens = torch.cat([question_tokens, action_tokens], dim=1)
        embedding = self.token_embedding(combined_tokens)
        # pos_encoder and transformer_encoder require shape (seq_len, batch_size, embedding_dim)
        embedding = embedding.permute((1, 0, 2))
        # apply positional encoding
        embedding_with_pos = self.pos_encoder(embedding)
        # create the padding mask
        padding_mask = torch.where(
            torch.logical_or(combined_tokens == self.question_padding_token,
                             combined_tokens == self.action_padding_token), 1, 0).type(torch.BoolTensor).to(self.device)
        # apply the transformer encoder
        # encoding = self.transformer_encoder(embedding_with_pos)
        encoding = self.transformer_encoder(embedding_with_pos, src_key_padding_mask=padding_mask)
        # output model --------------
        output = self.dense_block_1(encoding[0])
        output = self.dense_2(self.dropout(output))
        return output