import pandas as pd
import torch
from torch.utils.data import DataLoader

from resnet_model import FraudResNet
from data_loader import FraudDataset

# Load data
df = pd.read_csv("data/creditcard_2023.csv")

# Initial batch
df_init = df.iloc[:150000]

dataset = FraudDataset(df_init)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

model = FraudResNet(input_dim=30)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.BCELoss()

print("🚀 Initial Training Started")

for epoch in range(5):
    total_loss = 0
    for X, y in loader:
        optimizer.zero_grad()
        preds = model(X).view(-1)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "saved_models/fraud_model_v1.pth")
print("✅ Saved: fraud_model_v1.pth")
