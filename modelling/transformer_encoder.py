import math
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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


class TransformerEncoderModel(torch.nn.Module):
    def __init__(self, ntoken, nhead, nhid, nlayers, num_outputs, dropout, device, lr, max_grad_norm, batch_size):
        super().__init__()
        torch.nn.Module.__init__(self)
        # ntoken is vocab_size + 1 and vocab_size is index of padding_token, thus need to decrement ntoken by 1
        self.padding_token = ntoken - 1
        # define layers
        self.token_embedding = torch.nn.Embedding(ntoken, nhid)
        self.pos_encoder = PositionalEncoding(nhid, dropout)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=nhid, nhead=nhead), nlayers
        )
        self.policy_output = torch.nn.Linear(nhid, num_outputs)
        # set other things
        self.device = device
        self.to(device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=1)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.2, patience=10,
                                                               threshold=0.001, threshold_mode='rel', cooldown=0,
                                                               min_lr=0, eps=1e-08, verbose=False)
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size

    def forward(self, token_idxs):
        # embed the tokens
        embedding = self.token_embedding(token_idxs)
        # pos_encoder and transformer_encoder require shape (seq_len, batch_size, embedding_dim)
        embedding = embedding.permute((1, 0, 2))
        # apply positional encoding
        embedding_with_pos = self.pos_encoder(embedding)
        # create the padding mask
        padding_mask = torch.where(token_idxs == self.padding_token, 1, 0).type(torch.BoolTensor).to(self.device)
        # apply the transformer encoder
        # encoding = self.transformer_encoder(embedding_with_pos)
        encoding = self.transformer_encoder(embedding_with_pos, src_key_padding_mask=padding_mask)
        sliced_encoding = encoding[0]
        logits = self.policy_output(sliced_encoding)
        return logits