import torch


class Dataset(torch.utils.data.Dataset):

    def __init__(self, case, sequence, target):
        self.case = case
        self.sequence = sequence
        self.target = target

    def __len__(self):
        return len(self.case)

    def __getitem__(self, idx):
        return self.case[idx], self.sequence[idx], self.target[idx]


