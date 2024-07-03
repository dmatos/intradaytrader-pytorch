# coding=utf-8

import torch
from torch import nn

class IntradayTraderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2, output_size=1):
        super(IntradayTraderLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Size of the hidden state in the LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)  # Fully connected layer for output prediction

    def forward(self, input_data):
        # Initialize hidden and cell states for LSTM
        initial_hidden = torch.zeros(self.num_layers, input_data.size(0), self.hidden_size).to(input_data.device)
        initial_cell = torch.zeros(self.num_layers, input_data.size(0), self.hidden_size).to(input_data.device)

        # Forward propagate through LSTM
        lstm_output, _ = self.lstm(input_data, (initial_hidden, initial_cell))
        # reshaping data for dense layer next
        output = lstm_output[:, -1, :]
        output = self.fc(output)
        return output
