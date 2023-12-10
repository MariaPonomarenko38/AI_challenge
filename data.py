import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

DATA_PATH = './data/Microsoft_Stock.csv'
TRAIN_DATA_PATH = './data/train.npy'
TEST_DATA_PATH = './data/test.npy'

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, n_input):
        self.sequences = sequences
        self.n_timestamps = n_input

    def __len__(self):
        return len(self.sequences) - self.n_timestamps

    def __getitem__(self, idx):
        return (self.sequences[idx:idx+self.n_timestamps], self.sequences[idx+self.n_timestamps])


def get_data():
    df = pd.read_csv(DATA_PATH, index_col = 0, parse_dates = True )
    df.index = df.index.date
    len_train = len(df) * 0.7   
    len_test = len(df) * 0.3
    train_df = df.iloc[ :int(len_train+len_test)+1 ,:]
    test_df = df.iloc[int(len_train+len_test)+1: , :]

    scaler = MinMaxScaler()
    open_train = train_df[['Open']]  
    scaler.fit(open_train)

    x_train = scaler.transform(open_train)
    x_test = scaler.transform(test_df[['Open']])

    np.save(TRAIN_DATA_PATH, x_train)
    np.save(TEST_DATA_PATH, x_test)


def get_data_loader(number_of_time_stamps):
    x_train = np.load(TRAIN_DATA_PATH)
    x_test = np.load(TEST_DATA_PATH)

    x_train_tensor = torch.Tensor(x_train)
    x_test_tensor = torch.Tensor(x_test)
    
    train_dataset = TimeSeriesDataset(x_train_tensor, number_of_time_stamps)
    test_dataset = TimeSeriesDataset(x_test_tensor, number_of_time_stamps)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return (train_loader, test_loader)