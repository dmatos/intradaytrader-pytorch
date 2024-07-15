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
        self.sample_size = sample_size
        self.target_percent = target_percent
        self.means = self.df.mean(numeric_only=True)
        self.stds = self.df.std(numeric_only=True)
        self.labels, self.indexes = self.generate_labels_and_indexes()
        self.labels = torch.from_numpy(self.labels).type(torch.LongTensor).to(device=torch.device("cuda:0"), non_blocking=True)
        self.standardize()
        self.arr = self.df.to_numpy()
        self.len = len(self.indexes)

    def standardize(self):
        self.df = (self.df - self.means) / self.stds

    def get_original_data(self):
        return (self.df * self.stds[0]) + self.means[0]

    def invert_standardize(self, df):
        return (df * self.stds[0]) + self.means[0]

    def generate_labels_and_indexes(self):
        """
        It checks if there is any close from T+1 until T+N steps ahead that is X% different from the current close
        :return: list of labels, list of indexes to start and finish sample
        """
        labels = []
        indexes = []
        date_format = '%Y-%m-%d %H:%M:%S'
        for i in range(self.sample_size, self.df.shape[0]-self.steps_ahead):
            sample_end_day = pd.to_datetime(self.df.index[i], format=date_format).day
            labels_end_day = pd.to_datetime(self.df.index[i+self.steps_ahead], format=date_format).day
            if sample_end_day != labels_end_day:
                continue
            indexes.append([i-self.sample_size, i])
            value_to_append = 0
            value = self.df.iloc[i+self.steps_ahead]['close']
            if value >= self.df.iloc[i]['close'] * (1.0 + self.target_percent):
                # print("labels.append(1) at : ", i)
                value_to_append = 1
            elif value <= self.df.iloc[i]['close'] * (1.0 - self.target_percent):
                value_to_append = 2
            labels.append(value_to_append)
        return np.array(labels), np.array(indexes)  # numpy because of proc fork() and memory issues in python

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        train = self.arr[self.indexes[idx][0]:self.indexes[idx][1]]
        train = torch.from_numpy(train).type(dtype=torch.Tensor).to(device=torch.device("cuda:0"), non_blocking=True)
        test = self.labels[idx]
        return train, test

