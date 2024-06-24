# coding=utf-8

import torch
from torch import nn


class StockPriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(StockPriceLSTM, self).__init__()
        self.hidden_size = hidden_size  # Size of the hidden state in the LSTM
        self.num_layers = num_layers    # Number of LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # LSTM layer
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, output_size)  # Fully connected layer for output prediction
        self.relu = nn.ReLU()

    def forward(self, input_data):
        # Initialize hidden and cell states for LSTM
        initial_hidden = torch.zeros(self.num_layers, input_data.size(0), self.hidden_size).to(input_data.device)
        initial_cell = torch.zeros(self.num_layers, input_data.size(0), self.hidden_size).to(input_data.device)

        # Forward propagate through LSTM
        lstm_output, (hn, cn) = self.lstm(input_data, (initial_hidden, initial_cell))  # Output shape: (batch_size, seq_length, hidden_size)
        # reshaping data for dense layer next
        # only works for 1 lstm layer, I have to study how to add stacked layers
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        output = self.fc(out)

        return output

