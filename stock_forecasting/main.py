# coding=utf-8

import sys
sys.path.append('./')
# custom
from stock_dataset import StockDataset
from ml.lstm import IntradayTraderLSTM
# pure python
from os import listdir
from os.path import join, isfile
import time
import math
from datetime import date
# other libs
import torch
from torch.utils.data import ConcatDataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error
# plot
import matplotlib.pyplot as plt
from mpl_interactions import panhandler, zoom_factory
import seaborn as sns
sns.set_style("darkgrid")

model_filename = str(date.today())+'-stock-model'+'.pyt'
model_path_to_save = 'data/model/'+model_filename
csv_dir = 'data'
train_dir = join(csv_dir, 'train')
validation_dir = join(csv_dir, 'validation')

num_epochs = 500
learning_rate = 1e-4
input_dim = 61
hidden_dim = 50
num_layers = 1
output_dim = 60

sample_size = 9600
batch_size = 16


if __name__ == '__main__':
    # Load and preprocess the data

    csv_files = [f for f in listdir(train_dir) if isfile(join(train_dir, f))]
    datasets = []
    for idx, file in enumerate(csv_files):
        print('reading dataset #', idx, 'in file: ', file)
        dataset = StockDataset(join(train_dir, file), n_rowsto_drop=30, sample_size=sample_size, steps_ahead=output_dim)
        datasets.append(dataset)

    dataset = ConcatDataset(datasets)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    model = IntradayTraderLSTM(input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim, num_layers=num_layers)
    model = model.cuda()
    criterion = torch.nn.MSELoss(reduction='mean')
    criterion = criterion.cuda()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    hist = np.zeros(num_epochs)
    start_time = time.time()
    epoch_time = start_time
    lstm = []

    print("Start training")
    try:
        for t in range(num_epochs):
            hist[t] = 0
            counter = 0
            for batch, train in dataloader:
                optimiser.zero_grad(set_to_none=False)
                output_prediction = model(batch)
                loss = criterion(output_prediction, train)
                hist[t] += loss.item()
                counter += 1
                loss.backward()
                optimiser.step()
            epoch_time = int(time.time() - start_time)
            hist[t] = hist[t]/counter
            print("Epoch ", t, "Mean MSE: ", hist[t], "elapsed time ", epoch_time)
    except KeyboardInterrupt as e:
        print("Training process manually interrupted")
    finally:
        training_time = time.time() - start_time
        print("Training time: {}".format(training_time))

    torch.save(model.state_dict(), model_path_to_save)

    model.eval()

    validation_dataset = StockDataset(join(validation_dir, 'B3_PETR3.csv'), n_rowsto_drop=30, sample_size=sample_size, steps_ahead=output_dim)
    dataloader = DataLoader(validation_dataset, shuffle=False, batch_size=1)

    print("Validation dataset shape ", validation_dataset.df.shape)

    batch, validation = next(iter(dataloader))
    predictions = model(batch)

    validation = validation.to("cpu").detach().numpy()
    validation = validation_dataset.invert_standardize(validation)
    # print("validation data: ", validation)

    predictions = predictions.to("cpu").detach().numpy()
    predictions = validation_dataset.invert_standardize(predictions)

    testScore = math.sqrt(mean_squared_error(predictions, validation))
    print('Test Score: %.2f RMSE' % testScore)

    # plot predictions against original data
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    plt.subplot(1, 2, 1)
    sns.lineplot(x=[i for i in range(len(validation[0]))], y=validation[0], label="Data", color='royalblue')
    ax = sns.lineplot(x=[i for i in range(len(predictions[0]))], y=predictions[0], label="Training Prediction (LSTM)", color='tomato')
    ax.set_title('Stock price', size=14, fontweight='bold')
    ax.set_xlabel("Hours", size=14)
    ax.set_ylabel("Cost (BRL)", size=14)
    ax.set_xticklabels('', size=10)
    zoom_factory(ax)

    plt.subplot(1, 2, 2)
    ax = sns.lineplot(data=hist, color='royalblue')
    ax.set_xlabel("Epoch", size=14)
    ax.set_ylabel("Loss", size=14)
    ax.set_title("Training Loss", size=14, fontweight='bold')
    fig.set_figheight(6)
    fig.set_figwidth(16)
    zoom_factory(ax)

    # Enable scrolling and panning with the help of MPL
    # Interactions library function like panhandler.
    pan_handler = panhandler(fig)
    plt.show()
