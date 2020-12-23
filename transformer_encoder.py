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
        self.policy_output = nn.Linear(nhid, 3)
        self.value_output = nn.Linear(nhid, 1)

    def forward(self, input_dict, state, seq_lens):
        token_idxs = input_dict["obs"].type(torch.LongTensor)
        embedding = self.token_embedding(token_idxs) * math.sqrt(self.ninp)
        embedding_with_pos = self.pos_encoder(embedding)
        encoding = self.transformer_encoder(embedding_with_pos)
        logits = self.policy_output(encoding[:, 0, :])
        self.value = self.value_output(encoding[:, 0, :])
        # print(token_idxs.shape)
        # print(self.embedding.shape)
        # print(self.embedding_with_pos.shape)
        # print(self.encoding.shape)
        # print(self.logits.shape)
        return logits, state

    def value_function(self) -> TensorType:
        return self.value

    def import_from_h5(self, h5_file: str) -> None:
        pass
