import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import TensorType
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=5000):
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
        name
    ):
        super().__init__(
            observation_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # unpack model config
        self.custom_model_config = model_config["custom_model_config"]
        ntoken = self.custom_model_config["ntoken"]
        nhead = self.custom_model_config["nhead"]
        nhid = self.custom_model_config["nhid"]
        nlayers = self.custom_model_config["nlayers"]
        dropout = self.custom_model_config["dropout"]
        num_outputs = self.custom_model_config["num_outputs"]

        # define layers
        self.model_type = "Transformer"
        self.token_embedding = nn.Embedding(ntoken, nhid)
        self.pos_encoder = PositionalEncoding(nhid, dropout)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=nhid, nhead=nhead), nlayers
        )
        self.policy_output = nn.Linear(nhid, num_outputs)
        self.value_output = nn.Linear(nhid, 1)

    def forward(self, input_dict, state, seq_lens):
        # extract the observations
        token_idxs = input_dict["obs"].type(torch.LongTensor)
        # embed the tokens
        embedding = self.token_embedding(token_idxs)
        # pos_encoder and transformer_encoder require shape (seq_len, batch_size, embedding_dim)
        embedding = embedding.permute((1, 0, 2))
        # apply positional encoding
        embedding_with_pos = self.pos_encoder(embedding)
        # create the padding mask
        # padding_mask = torch.where(token_idxs == padding_token, 0, 1).type(torch.BoolTensor)
        # apply the transformer encoder
        encoding = self.transformer_encoder(embedding_with_pos)  # , src_key_padding_mask=padding_mask)
        sliced_encoding = encoding[0]
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
