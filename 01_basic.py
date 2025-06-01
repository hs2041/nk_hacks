import kagglehub
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import torch
import helper.data_plotter as data_plotter
import helper.model_trainer as model_trainer
import os
# kagglehub.login()

# nk_iv_prediction_path = kagglehub.competition_download('nk-iv-prediction')
# print(nk_iv_prediction_path)
# print('Data source import complete.')

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = model_trainer.NeuralNetwork().to(device)
print(model)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load the training data
train_df = pd.read_parquet('data/train_data.parquet')

# Load the test data
test_df = pd.read_parquet('data/test_data.parquet')

# print(train_df)

# Sample prediction
# test_df = test_df.replace(np.nan, 0.1)

# sample_submission = pd.read_csv('data/sample_submission.csv')
# submission_cols = sample_submission.columns.tolist()

# submission_df = test_df[submission_cols]
# submission_df.to_csv('output/submission.csv', index = False)

# model_trainer.model_trainer(train_df, test_df)

data_plotter.plot_train_params(train_df)

# Save files in csv format
# train_df.to_csv("plots/train_data.csv", index = False)
# test_df.to_csv("plots/test_data.csv", index = False)