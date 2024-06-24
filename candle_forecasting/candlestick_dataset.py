# coding=utf-8

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CandlestickDataset(Dataset):

    def __init__(self, file_path, sample_size=32, n_rowsto_drop=0, steps_ahead=1):
        self.df = pd.read_csv(file_path)
        self.df['date'] = pd.to_datetime(self.df['timestampUTC'], unit='s', errors='coerce')
        self.df = self.df[['date', 'close', 'open']]
        self.df.set_index('date', inplace=True)
        # self.df.drop(columns=['timestampUTC'], inplace=True)
        self.df.drop([self.df.index[i] for i in range(n_rowsto_drop)], inplace=True, axis='index')
        self.sample_size = sample_size
        self.steps_ahead = steps_ahead
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
        train = torch.from_numpy(self.df[idx:idx+self.sample_size].to_numpy()).type(torch.Tensor)
        test = (torch.from_numpy(np.array(self.df[['open', 'close']][idx+self.sample_size:idx+self.sample_size+self.steps_ahead]))
                .type(torch.Tensor))
        return train.cuda(), test.cuda()

    def get_item_replaced_with_predictions_as_df(self, idx, open_prediction_values, close_prediction_values):
        #TODO calculate macd, signal, mean for the last item
        replace_start_index = self.sample_size-len(open_prediction_values)
        train = self.df[idx:idx+self.sample_size]
        train.loc[train.index[replace_start_index:], 'open'] = open_prediction_values
        train.loc[train.index[replace_start_index:], 'close'] = close_prediction_values
        # train.loc[train.index[replace_start_index:], '2ndStdDown'] = self.df.loc[self.df.index[idx-replace_start_index-1], '2ndStdDown']
        # train.loc[train.index[replace_start_index:], '3rdStdDown'] = self.df.loc[self.df.index[idx-replace_start_index-1], '3rdStdDown']
        # train.loc[train.index[replace_start_index:], '2ndStdUp'] = self.df.loc[self.df.index[idx-replace_start_index-1], '2ndStdUp']
        # train.loc[train.index[replace_start_index:],  '3rdStdUp'] = self.df.loc[self.df.index[idx-replace_start_index-1], '3rdStdUp']
        # train.loc[train.index[replace_start_index:],  'macd'] = self.df.loc[self.df.index[idx-replace_start_index-1], 'macd']
        # train.loc[train.index[replace_start_index:],  'signal'] = self.df.loc[self.df.index[idx-replace_start_index-1], 'signal']
        # train.loc[train.index[replace_start_index:],  'mean'] = self.df.loc[self.df.index[idx-replace_start_index-1], 'mean']
        test = (torch.from_numpy(np.array(self.df[['open', 'close']][idx+self.sample_size:idx+self.sample_size+self.steps_ahead]))
                .type(torch.Tensor))
        return torch.from_numpy(train.to_numpy()).type(torch.Tensor).cuda(), test.cuda()
