import pandas as pd
import torch
from torch.utils.data import DataLoader

from resnet_model import FraudResNet
from data_loader import FraudDataset
from replay_buffer import ReplayBuffer

# Load data
df = pd.read_csv("data/creditcard_2023.csv")

# New batch
df_new = df.iloc[150000:300000]
df_old = df.iloc[:150000]

new_dataset = FraudDataset(df_new)
new_loader = DataLoader(new_dataset, batch_size=1, shuffle=True)

# Load model
model = FraudResNet(input_dim=30)
model.load_state_dict(torch.load("saved_models/fraud_model_v1.pth"))
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn = torch.nn.BCELoss()

# Replay Buffer
buffer = ReplayBuffer()
old_dataset = FraudDataset(df_old)
buffer.add(list(old_dataset))

print("🔁 Incremental Training Started")

for X, y in new_loader:
    replay_samples = buffer.sample(1)
    mixed = [(X, y)] + replay_samples

    for xb, yb in mixed:
        optimizer.zero_grad()

        pred = model(xb.unsqueeze(0)).view(-1)
        yb = yb.view(-1)

        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "saved_models/fraud_model_v2.pth")
print("✅ Saved: fraud_model_v2.pth")
