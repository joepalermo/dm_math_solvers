import math
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Softmax
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
    def __init__(self, ntoken, num_outputs, max_num_nodes, device):
        super().__init__()
        torch.nn.Module.__init__(self)
        self.max_grad_norm = hparams.train.max_grad_norm
        self.batch_size = hparams.train.batch_size
        self.epsilon = hparams.train.epsilon
        # ntoken is vocab_size + 1 and vocab_size is index of padding_token, thus need to decrement ntoken by 1
        self.padding_token = ntoken - 1

        # define tunable layers -------------------
        self.token_embedding = torch.nn.Embedding(ntoken, hparams.model.nhid)
        self.action_embedding = torch.nn.Embedding(num_outputs, hparams.model.action_embedding_size)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=hparams.model.nhid, nhead=hparams.model.nhead), hparams.model.nlayers
        )
        self.dense_block_1 = DenseBlock(max_num_nodes * hparams.model.action_embedding_size, hparams.model.nhid)
        self.dense_block_2 = DenseBlock(2*hparams.model.nhid, hparams.model.nhid)
        self.dense_3 = torch.nn.Linear(hparams.model.nhid, num_outputs)

        # define non-tunable layers -------------------
        self.pos_encoder = PositionalEncoding(hparams.model.nhid, hparams.train.dropout)
        self.dropout = torch.nn.Dropout(hparams.train.dropout)
        self.relu = torch.nn.ReLU()

        # other
        self.device = device
        self.to(device)

        # set optimization
        self.optimizer = torch.optim.SGD(self.parameters(), lr=hparams.train.lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=1)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=hparams.train.factor, patience=hparams.train.patience,
                                                               threshold=0.001, threshold_mode='rel', cooldown=0,

                                                               min_lr=hparams.train.min_lr, eps=1e-08, verbose=False)

    def forward(self, question_tokens, action_tokens):
        # question_tokens: (BS, max_question_length), action_tokens: (BS, max_num_actions)
        # question model --------------
        # embed the tokens
        embedding = self.token_embedding(question_tokens)
        # pos_encoder and transformer_encoder require shape (seq_len, batch_size, embedding_dim)
        embedding = embedding.permute((1, 0, 2))
        # apply positional encoding
        embedding_with_pos = self.pos_encoder(embedding)
        # create the padding mask
        padding_mask = torch.where(question_tokens == self.padding_token, 1, 0).type(torch.BoolTensor).to(self.device)
        # apply the transformer encoder
        # encoding = self.transformer_encoder(embedding_with_pos)
        encoding = self.transformer_encoder(embedding_with_pos, src_key_padding_mask=padding_mask)
        question_encoding = encoding[0]
        # action model --------------
        # (BS, max_num_actions) => (BS, max_num_actions, embedding_dim)
        action_embedding = self.action_embedding(action_tokens)
        # (BS, max_num_actions, embedding_dim) => (BS, max_num_actions * embedding_dim)
        flattened_action_embedding = torch.flatten(action_embedding, start_dim=1)
        action_encoding = self.dense_block_1(flattened_action_embedding)
        # output model --------------
        question_and_actions = torch.cat([question_encoding, action_encoding], dim=1)
        output = self.dense_block_2(question_and_actions)
        output = self.dense_3(self.dropout(output))
        return output