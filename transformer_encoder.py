import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import TensorType
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

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


class TransformerModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        num_outputs,
        model_config,
        name,
        ntoken,
        ninp,
        nhead,
        nhid,
        nlayers,
        dropout=0.5,
    ):
        super().__init__(
            observation_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.ninp = ninp
        self.model_type = "Transformer"
        self.token_embedding = nn.Embedding(ntoken, nhid)
        self.pos_encoder = PositionalEncoding(nhid, dropout)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=nhid, nhead=nhead), nlayers
        )
        # if slicing a single element of the encoded sequence
        self.policy_output = nn.Linear(nhid, 3)
        self.value_output = nn.Linear(nhid, 1)

        # if using all elements of the encoded sequence
        # self.policy_output = nn.Linear(nhid*ninp, 3)
        # self.value_output = nn.Linear(nhid*ninp, 1)

    def forward(self, input_dict, state, seq_lens):
        # extract the observations
        token_idxs = input_dict["obs"].type(torch.LongTensor)
        # embed the tokens
        embedding = self.token_embedding(token_idxs)
        embedding_with_pos = self.pos_encoder(embedding)
        # create the padding mask
        padding_mask = torch.clip(token_idxs, max=1).type(torch.BoolTensor)
        # nn.transformer requires shape (seq_len, batch_size, embedding_dim)
        embedding_with_pos = embedding_with_pos.permute((1, 0, 2))
        # apply the transformer encoder
        encoding = self.transformer_encoder(embedding_with_pos, src_key_padding_mask=padding_mask)
        # produce outputs
        sliced_encoding = encoding[0]
        # flattened_encoding = torch.flatten(encoding, start_dim=1)
        logits = self.policy_output(sliced_encoding)
        # squeeze because values are scalars, not 1D array
        self.value = self.value_output(sliced_encoding).squeeze(-1)
        # print()
        # print('token_idxs', token_idxs.shape)
        # # print(token_idxs)
        # print('embedding', embedding.shape)
        # print('embedding_with_pos', embedding_with_pos.shape)
        # print('sliced_encoding', sliced_encoding.shape)
        # print('encoding', encoding.shape)
        # # print(flattened_encoding.shape)
        # print('logits', logits.shape)
        # print('value', self.value.shape)
        return logits, state

    def value_function(self) -> TensorType:
        return self.value

    def import_from_h5(self, h5_file: str) -> None:
        pass
