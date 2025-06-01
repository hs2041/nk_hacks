import kagglehub
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import torch
from torch import nn
# from torch.optim import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import helper.data_plotter as data_plotter

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(43, 64) # 1 input, 64 hidden neurons in layer 1
        self.hidden2 = torch.nn.Linear(64, 128) # 64 input, 128 hidden neurons in layer 2
        self.output = torch.nn.Linear(128, 1) # 128 input, 1 output

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

def model_trainer(train_df, test_df):
    labels = ['call_iv_23500']
    # labels = ['call_iv_23500','call_iv_23600','call_iv_23700','call_iv_23800','call_iv_23900','call_iv_24000','call_iv_24100','call_iv_24200','call_iv_24300','call_iv_24400','call_iv_24500','call_iv_24600','call_iv_24700','call_iv_24800','call_iv_24900','call_iv_25000','call_iv_25100','call_iv_25200','call_iv_25300','call_iv_25400','call_iv_25500','call_iv_25600','call_iv_25700','call_iv_25800','call_iv_25900','call_iv_26000','put_iv_22500','put_iv_22600','put_iv_22700','put_iv_22800','put_iv_22900','put_iv_23000','put_iv_23100','put_iv_23200','put_iv_23300','put_iv_23400','put_iv_23500','put_iv_23600','put_iv_23700','put_iv_23800','put_iv_23900','put_iv_24000','put_iv_24100','put_iv_24200','put_iv_24300','put_iv_24400','put_iv_24500','put_iv_24600','put_iv_24700','put_iv_24800','put_iv_24900','put_iv_25000']
    features = ['underlying', 'X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23','X24','X25','X26','X27','X28','X29','X30','X31','X32','X33','X34','X35','X36','X37','X38','X39','X40','X41']
    # Separate features and labels
    X = train_df[features].values
    Y = train_df[labels].values

    # Normalize feature data
    for column in features:
        train_df[column] = train_df[column] /train_df[column].abs().max()

    # print("Features: ", X)
    # data_plotter.plot_train_params(train_df)

    # Convert to NumPy arrays and then to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    # Create TensorDataset
    dataset = TensorDataset(X_tensor, Y_tensor)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

    net = Net()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  # Using Adam optimizer

    for epoch in range(50):
        for X_batch, Y_batch in dataloader:
            # print(X_batch)
            # print(Y_batch)
            running_loss = 0.0
            optimizer.zero_grad()
            outputs = net(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if epoch % 100 == 0:
            print("Epoch {}: Loss = {}".format(epoch, loss.detach().numpy()))



    # Example of iterating through the data loader
    # for batch_idx, (features, labels) in enumerate(dataloader):
    #     print(f"Batch {batch_idx}: Features {features}, Labels {labels}")

