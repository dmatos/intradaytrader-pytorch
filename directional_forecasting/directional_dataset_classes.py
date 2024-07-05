# coding=utf-8

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DirectionalClassesDataset(Dataset):

    def __init__(self, file_path, target_index='close', sample_size=32, n_rowsto_drop=0, steps_ahead=1, target_percent=0.01):
        self.steps_ahead = steps_ahead
        self.df = pd.read_csv(file_path)
        self.df['date'] = pd.to_datetime(self.df['timestampUTC'], unit='s', errors='coerce')
        self.df.set_index('date', inplace=True)
        self.df.drop(columns=['timestampUTC'], inplace=True)
        self.df.drop([self.df.index[i] for i in range(n_rowsto_drop)], inplace=True, axis='index')
        self.df.insert(0, target_index, self.df.pop(target_index))
        self.sample_size = sample_size
        self.target_percent = target_percent
        self.means = self.df.mean(numeric_only=True)
        self.stds = self.df.std(numeric_only=True)
        self.labels = self.generate_labels()
        self.labels = torch.from_numpy(self.labels).type(torch.LongTensor).to(device=torch.device("cuda:0"))
        self.standardize()
        self.arr = torch.from_numpy(self.df.to_numpy()).type(torch.Tensor).to(device=torch.device("cuda:0"))

    def standardize(self):
        self.df = (self.df - self.means) / self.stds

    def get_original_data(self):
        return (self.df * self.stds[0]) + self.means[0]

    def invert_standardize(self, df):
        return (df * self.stds[0]) + self.means[0]

    def generate_labels(self):
        """
        It checks if there is any close from T+1 until T+N steps ahead that is X% different from the current close
        :return: list of labels
        """
        labels = []
        for i in range(0, self.df.shape[0]-self.steps_ahead):
            value_to_append = 0
            for j in range(1, self.steps_ahead):
                value = self.df.iloc[i+j]['close']
                if value >= self.df.iloc[i]['close'] * (1.0 + self.target_percent):
                    # print("labels.append(1) at : ", i)
                    value_to_append = 1
                    break
                elif value <= self.df.iloc[i]['close'] * (1.0 - self.target_percent):
                    value_to_append = 2
                    break
            labels.append(value_to_append)
        return np.array(labels)  # numpy because of proc fork() and memory issues in python

    def __len__(self):
        return self.df.shape[0] - self.sample_size - self.steps_ahead

    def __getitem__(self, idx):
        # train array and test item
        train = self.arr[idx:idx+self.sample_size]
        test = self.labels[idx]
        return train, test

