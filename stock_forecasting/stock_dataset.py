# coding=utf-8

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):

    def __init__(self, file_path, target_index='close', sample_size=32, n_rowsto_drop=0, steps_ahead=1):
        self.steps_ahead = steps_ahead
        self.df = pd.read_csv(file_path)
        self.df['date'] = pd.to_datetime(self.df['timestampUTC'], unit='s', errors='coerce')
        # timestampUTC,open,high,low,close,volume,mean,2ndStdUp,3rdStdUp,2ndStdDown,3rdStdDown,macd,signal,histogram,rsi
        self.df = self.df[['date', 'close', 'open', 'high', 'low', 'macd', 'signal', 'mean', 'volume', '2ndStdDown', '3rdStdDown', '2ndStdUp', '3rdStdUp']]
        self.df.set_index('date', inplace=True)
        # self.df.drop(columns=['timestampUTC'], inplace=True)
        self.df.drop([self.df.index[i] for i in range(n_rowsto_drop)], inplace=True, axis='index')
        self.df.insert(0, target_index, self.df.pop(target_index))
        self.sample_size = sample_size
        self.means = self.df.mean(numeric_only=True) #TODO usar algum tipo de "transform"???
        self.stds = self.df.std(numeric_only=True)
        self.standardize()

    def standardize(self):
        self.df = (self.df - self.means) / self.stds

    def get_original_data(self):
        return (self.df * self.stds[0]) + self.means[0]

    def invert_standardize(self, df):
        return (df * self.stds[0]) + self.means[0]

    def __len__(self):
        return self.df.shape[0] - self.sample_size - self.steps_ahead

    def __getitem__(self, idx):
        # train array and test item
        train = torch.from_numpy(self.df[idx:idx+self.sample_size].to_numpy()).type(torch.Tensor)
        test = torch.from_numpy(
            np.array(self.df['close'][idx+self.sample_size:idx+self.sample_size+self.steps_ahead])
        ).type(torch.Tensor)
        return train.cuda(), test.cuda()

