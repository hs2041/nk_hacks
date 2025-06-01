import kagglehub
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

def plot_train_nifty(train_df):
    train_df.plot(x="timestamp", y='underlying', kind='line')
    plt.show()

def plot_train_calls(train_df):
    train_df.plot(x="timestamp", y='call_iv_23500', kind='line')
    train_df.plot(x="timestamp", y='call_iv_23600', kind='line')
    train_df.plot(x="timestamp", y='call_iv_23700', kind='line')
    train_df.plot(x="timestamp", y='call_iv_23800', kind='line')
    train_df.plot(x="timestamp", y='call_iv_23900', kind='line')
    train_df.plot(x="timestamp", y='call_iv_24000', kind='line')
    train_df.plot(x="timestamp", y='call_iv_24100', kind='line')
    train_df.plot(x="timestamp", y='call_iv_24200', kind='line')
    train_df.plot(x="timestamp", y='call_iv_24300', kind='line')
    train_df.plot(x="timestamp", y='call_iv_24400', kind='line')
    train_df.plot(x="timestamp", y='call_iv_24500', kind='line')
    train_df.plot(x="timestamp", y='call_iv_24600', kind='line')
    train_df.plot(x="timestamp", y='call_iv_24700', kind='line')
    train_df.plot(x="timestamp", y='call_iv_24800', kind='line')
    train_df.plot(x="timestamp", y='call_iv_24900', kind='line')
    train_df.plot(x="timestamp", y='call_iv_25000', kind='line')
    train_df.plot(x="timestamp", y='call_iv_25100', kind='line')
    train_df.plot(x="timestamp", y='call_iv_25200', kind='line')
    train_df.plot(x="timestamp", y='call_iv_25300', kind='line')
    train_df.plot(x="timestamp", y='call_iv_25400', kind='line')
    train_df.plot(x="timestamp", y='call_iv_25500', kind='line')
    train_df.plot(x="timestamp", y='call_iv_25600', kind='line')
    train_df.plot(x="timestamp", y='call_iv_25700', kind='line')
    train_df.plot(x="timestamp", y='call_iv_25800', kind='line')
    train_df.plot(x="timestamp", y='call_iv_25900', kind='line')
    train_df.plot(x="timestamp", y='call_iv_26000', kind='line')
    plt.show()

def plot_train_puts(train_df):
    train_df.plot(x="timestamp", y='put_iv_22500', kind='line')
    train_df.plot(x="timestamp", y='put_iv_22600', kind='line')
    train_df.plot(x="timestamp", y='put_iv_22700', kind='line')
    train_df.plot(x="timestamp", y='put_iv_22800', kind='line')
    train_df.plot(x="timestamp", y='put_iv_22900', kind='line')
    train_df.plot(x="timestamp", y='put_iv_23000', kind='line')
    train_df.plot(x="timestamp", y='put_iv_23100', kind='line')
    train_df.plot(x="timestamp", y='put_iv_23200', kind='line')
    train_df.plot(x="timestamp", y='put_iv_23300', kind='line')
    train_df.plot(x="timestamp", y='put_iv_23400', kind='line')
    train_df.plot(x="timestamp", y='put_iv_23500', kind='line')
    train_df.plot(x="timestamp", y='put_iv_23600', kind='line')
    train_df.plot(x="timestamp", y='put_iv_23700', kind='line')
    train_df.plot(x="timestamp", y='put_iv_23800', kind='line')
    train_df.plot(x="timestamp", y='put_iv_23900', kind='line')
    train_df.plot(x="timestamp", y='put_iv_24000', kind='line')
    train_df.plot(x="timestamp", y='put_iv_24100', kind='line')
    train_df.plot(x="timestamp", y='put_iv_24200', kind='line')
    train_df.plot(x="timestamp", y='put_iv_24300', kind='line')
    train_df.plot(x="timestamp", y='put_iv_24400', kind='line')
    train_df.plot(x="timestamp", y='put_iv_24500', kind='line')
    train_df.plot(x="timestamp", y='put_iv_24600', kind='line')
    train_df.plot(x="timestamp", y='put_iv_24700', kind='line')
    train_df.plot(x="timestamp", y='put_iv_24800', kind='line')
    train_df.plot(x="timestamp", y='put_iv_24900', kind='line')
    train_df.plot(x="timestamp", y='put_iv_25000', kind='line')
    plt.show()

def plot_train_params(train_df):
    train_df.plot(x="timestamp", y='X0', kind='line')
    train_df.plot(x="timestamp", y='X1', kind='line')
    train_df.plot(x="timestamp", y='X2', kind='line')
    train_df.plot(x="timestamp", y='X3', kind='line')
    train_df.plot(x="timestamp", y='X4', kind='line')
    train_df.plot(x="timestamp", y='X5', kind='line')
    train_df.plot(x="timestamp", y='X6', kind='line')
    train_df.plot(x="timestamp", y='X7', kind='line')
    train_df.plot(x="timestamp", y='X8', kind='line')
    train_df.plot(x="timestamp", y='X9', kind='line')
    train_df.plot(x="timestamp", y='X10', kind='line')
    train_df.plot(x="timestamp", y='X11', kind='line')
    train_df.plot(x="timestamp", y='X12', kind='line')
    train_df.plot(x="timestamp", y='X13', kind='line')
    train_df.plot(x="timestamp", y='X14', kind='line')
    train_df.plot(x="timestamp", y='X15', kind='line')
    train_df.plot(x="timestamp", y='X16', kind='line')
    train_df.plot(x="timestamp", y='X17', kind='line')
    train_df.plot(x="timestamp", y='X18', kind='line')
    train_df.plot(x="timestamp", y='X19', kind='line')
    train_df.plot(x="timestamp", y='X20', kind='line')
    train_df.plot(x="timestamp", y='X21', kind='line')
    train_df.plot(x="timestamp", y='X22', kind='line')
    train_df.plot(x="timestamp", y='X23', kind='line')
    train_df.plot(x="timestamp", y='X24', kind='line')
    train_df.plot(x="timestamp", y='X25', kind='line')
    train_df.plot(x="timestamp", y='X26', kind='line')
    train_df.plot(x="timestamp", y='X27', kind='line')
    train_df.plot(x="timestamp", y='X28', kind='line')
    train_df.plot(x="timestamp", y='X29', kind='line')
    train_df.plot(x="timestamp", y='X30', kind='line')
    train_df.plot(x="timestamp", y='X31', kind='line')
    train_df.plot(x="timestamp", y='X32', kind='line')
    train_df.plot(x="timestamp", y='X33', kind='line')
    train_df.plot(x="timestamp", y='X34', kind='line')
    train_df.plot(x="timestamp", y='X35', kind='line')
    train_df.plot(x="timestamp", y='X36', kind='line')
    train_df.plot(x="timestamp", y='X37', kind='line')
    train_df.plot(x="timestamp", y='X38', kind='line')
    train_df.plot(x="timestamp", y='X39', kind='line')
    train_df.plot(x="timestamp", y='X40', kind='line')
    train_df.plot(x="timestamp", y='X41', kind='line')
    plt.show()

























