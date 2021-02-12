import math
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Softmax, LSTM
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


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
    def __init__(self, dim_in, dim_out, dropout):
        super(DenseBlock, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.dense = torch.nn.Linear(dim_in, dim_out)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = self.relu(x)
        return x


class Transformer(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()
        torch.nn.Module.__init__(self)
        # data
        self.num_cases = config['num_cases']
        self.action_padding_token = config['action_padding_token']
        self.num_action_tokens = config['num_action_tokens']
        # model
        self.num_layers = config['num_layers']
        self.hidden_size = config['hidden_size']
        self.dropout = config['dropout']
        # optimization
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']

        # define tunable layers -------------------
        self.action_embedding = torch.nn.Embedding(self.num_action_tokens, self.hidden_size)
        self.question_id_embedding = torch.nn.Embedding(self.num_cases, self.hidden_size)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=self.hidden_size, nhead=4), self.num_layers
        )
        self.lstm_block = LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                               num_layers=self.num_layers, batch_first=True, dropout=self.dropout)
        self.dense_block_1 = DenseBlock(2 * self.hidden_size, self.hidden_size, self.dropout)
        self.dense_2 = torch.nn.Linear(self.hidden_size, 1)

        # define non-tunable layers -------------------
        self.pos_encoder = PositionalEncoding(self.hidden_size, self.dropout)
        self.dropout = torch.nn.Dropout(self.dropout)
        self.relu = torch.nn.ReLU()

        # other
        self.device = device
        self.to(device)

        # set optimization
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=hparams.train.lr, weight_decay=hparams.train.weight_decay)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


    def forward(self, case_token, action_tokens):
        # question model --------------
        question_encoding = self.question_id_embedding(case_token)
        # question model --------------
        # embed the tokens
        embedding = self.action_embedding(action_tokens)
        # pos_encoder and transformer_encoder require shape (seq_len, batch_size, embedding_dim)
        embedding = embedding.permute((1, 0, 2))
        # apply positional encoding
        embedding_with_pos = self.pos_encoder(embedding)
        # create the padding mask
        padding_mask = torch.where(action_tokens == self.action_padding_token, 1, 0).type(torch.BoolTensor).to(self.device)
        # apply the transformer encoder
        # encoding = self.transformer_encoder(embedding_with_pos)
        encoding = self.transformer_encoder(embedding_with_pos, src_key_padding_mask=padding_mask)
        action_encoding = encoding[0]
        # output model --------------
        question_and_actions = torch.cat([question_encoding, action_encoding], dim=1)
        output = self.dense_block_1(question_and_actions)
        output = self.dense_2(self.dropout(output))
        return output


class RNN(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()
        torch.nn.Module.__init__(self)
        # data
        self.num_cases = config['num_cases']
        self.action_padding_token = config['action_padding_token']
        self.num_action_tokens = config['num_action_tokens']
        # model
        self.num_layers = config['num_layers']
        self.hidden_size = config['hidden_size']
        self.dropout = config['dropout']
        # optimization
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']

        # define tunable layers -------------------
        self.action_embedding = torch.nn.Embedding(self.num_action_tokens, self.hidden_size)
        self.question_id_embedding = torch.nn.Embedding(self.num_cases, self.hidden_size)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=self.hidden_size, nhead=4), self.num_layers
        )
        self.lstm_block = LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                               num_layers=self.num_layers, batch_first=True, dropout=self.dropout)
        self.dense_block_1 = DenseBlock(2 * self.hidden_size, self.hidden_size, self.dropout)
        self.dense_2 = torch.nn.Linear(self.hidden_size, 1)

        # define non-tunable layers -------------------
        self.pos_encoder = PositionalEncoding(self.hidden_size, self.dropout)
        self.dropout = torch.nn.Dropout(self.dropout)
        self.relu = torch.nn.ReLU()

        # other
        self.device = device
        self.to(device)

        # set optimization
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=hparams.train.lr, weight_decay=hparams.train.weight_decay)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


    def forward(self, case_token, action_tokens):
        # question model --------------
        question_encoding = self.question_id_embedding(case_token)
        # action model --------------
        sequence_lens = action_tokens.shape[1] - torch.sum(action_tokens == self.action_padding_token, axis=1)
        sequence_lens = sequence_lens.detach().cpu()
        # (BS, max_num_actions) => (BS, max_num_actions, embedding_dim)
        action_embedding = self.action_embedding(action_tokens)
        packed_action_embedding = pack_padded_sequence(action_embedding, sequence_lens, batch_first=True,
                                                       enforce_sorted=False)
        # (BS, max_num_actions, embedding_dim) => (BS, max_num_actions, hparams.model.lstm_hidden_size)
        packed_lstm_output, _ = self.lstm_block(packed_action_embedding)
        padded_lstm_output, output_lengths = pad_packed_sequence(packed_lstm_output, batch_first=True)
        action_encoding = padded_lstm_output[torch.arange(len(padded_lstm_output)), output_lengths-1]
        # output model --------------
        question_and_actions = torch.cat([question_encoding, action_encoding], dim=1)
        output = self.dense_block_1(question_and_actions)
        output = self.dense_2(self.dropout(output))
        return output