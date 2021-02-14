import torch
from torch import nn
from hparams import HParams
hparams = HParams.get_hparams_by_name('rl_math')
from collections import OrderedDict

class MLPBlock(nn.Module):
    def __init__(self, device, input_size, num_outputs):
        super(MLPBlock, self).__init__()
        self.device = device
        self.layer_dict = OrderedDict()
        layer_sizes = [input_size] + hparams.mlp.hidden_size + [num_outputs]
        for i in range(hparams.mlp.n_hidden+1):
            self.layer_dict[f"dense_{i}"] = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layer_dict[f"relu_{i}"] = nn.ReLU()
        self.layers = nn.Sequential(self.layer_dict)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        self.batch_size = 32
        self.max_grad_norm = 0.5

    def forward(self, x):
        x = self.layers(x)
        return x
