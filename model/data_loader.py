import torch
from torch.utils.data import Dataset

class FraudDataset(Dataset):
    def __init__(self, df):
        self.X = df.drop(columns=["Class"]).values
        self.y = df["Class"].values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )
