import kagglehub
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import torch
from torch import nn
# from torch.optim import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

def model_trainer(train_df, test_df):
    labels = ['call_iv_23500','call_iv_23600','call_iv_23700','call_iv_23800','call_iv_23900','call_iv_24000','call_iv_24100','call_iv_24200','call_iv_24300','call_iv_24400','call_iv_24500','call_iv_24600','call_iv_24700','call_iv_24800','call_iv_24900','call_iv_25000','call_iv_25100','call_iv_25200','call_iv_25300','call_iv_25400','call_iv_25500','call_iv_25600','call_iv_25700','call_iv_25800','call_iv_25900','call_iv_26000','put_iv_22500','put_iv_22600','put_iv_22700','put_iv_22800','put_iv_22900','put_iv_23000','put_iv_23100','put_iv_23200','put_iv_23300','put_iv_23400','put_iv_23500','put_iv_23600','put_iv_23700','put_iv_23800','put_iv_23900','put_iv_24000','put_iv_24100','put_iv_24200','put_iv_24300','put_iv_24400','put_iv_24500','put_iv_24600','put_iv_24700','put_iv_24800','put_iv_24900','put_iv_25000']
    features = ['underlying', 'X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23','X24','X25','X26','X27','X28','X29','X30','X31','X32','X33','X34','X35','X36','X37','X38','X39','X40','X41']
    # Separate features and labels
    X = train_df[features].values
    y = train_df[labels].values

    print("Features: ", X)
    # Convert to NumPy arrays and then to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Create TensorDataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Example of iterating through the data loader
    # for batch_idx, (features, labels) in enumerate(dataloader):
    #     print(f"Batch {batch_idx}: Features {features}, Labels {labels}")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits