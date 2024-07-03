# coding=utf-8

import sys
sys.path.append('./')
# custom
from directional_dataset_classes import DirectionalClassesDataset
from ml.lstm import IntradayTraderLSTM
# pure python
from os import listdir
from os.path import join, isfile
import time
import datetime
from datetime import date
# other libs
import torch
from torch.utils.data import ConcatDataset, DataLoader
import numpy as np
# plot
import matplotlib.pyplot as plt
from mpl_interactions import zoom_factory
import seaborn as sns
sns.set_style("darkgrid")

csv_dir = 'data'
train_dir = join(csv_dir, 'train')
validation_dir = join(csv_dir, 'validation')
validation_file = 'B3_PETR3.csv'

num_epochs = 5000
learning_rate = 1e-4
input_dim = 61
hidden_dim = 100
num_layers = 5
output_dim = 3
close_at_steps_ahead = 12

sample_size = 240
batch_size = 32


def load_dataset(dir_to_read, filename):
    return DirectionalClassesDataset(
        join(dir_to_read, filename),
        n_rowsto_drop=60,
        sample_size=sample_size,
        steps_ahead=close_at_steps_ahead,
        target_percent=0.005
    )


def get_model():
    return IntradayTraderLSTM(
        input_size=input_dim,
        hidden_size=hidden_dim,
        num_layers=num_layers,
        dropout=0.2,
        output_size=output_dim)


# @profile
def load_datasets():
    # Load and preprocess the data

    csv_files = [f for f in listdir(train_dir) if isfile(join(train_dir, f))]
    datasets = []
    for idx, file in enumerate(csv_files):
        print('reading dataset #', idx, 'in file: ', file)
        dataset = load_dataset(train_dir, file)
        datasets.append(dataset)

    dataset = ConcatDataset(datasets)
    return DataLoader(dataset, shuffle=False, batch_size=batch_size, persistent_workers=False)


def plot(validations, predictions, loss_history):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    plt.subplot(1, 2, 1)
    sns.lineplot(x=[i for i in range(len(validations))], y=validations, label="Data", color='royalblue')
    ax = sns.lineplot(x=[i for i in range(len(predictions))], y=predictions, label="Training Prediction (LSTM)", color='tomato')
    ax.set_title('Stock price', size=14, fontweight='bold')
    ax.set_xlabel("Hours", size=14)
    ax.set_ylabel("Cost (BRL)", size=14)
    ax.set_xticklabels('', size=10)
    zoom_factory(ax)

    plt.subplot(1, 2, 2)
    ax = sns.lineplot(data=loss_history, color='royalblue')
    ax.set_xlabel("Epoch", size=14)
    ax.set_ylabel("Loss", size=14)
    ax.set_title("Training Loss", size=14, fontweight='bold')
    fig.set_figheight(6)
    fig.set_figwidth(16)
    zoom_factory(ax)

    # Enable scrolling and panning with the help of MPL
    # Interactions library function like panhandler.
    plt.show()


def validation_step(model, hist):
    validation_dataset = load_dataset(validation_dir, validation_file)
    dataloader = DataLoader(validation_dataset, shuffle=False, batch_size=1)
    print("Validation dataset shape ", validation_dataset.df.shape)
    output_predictions = []
    output_validations = []
    for batch, train in dataloader:
        predictions = torch.argmax(model(batch), 1)
        output_predictions.append(predictions.to("cpu").detach().numpy()[0])
        output_validations.append(train.to("cpu").detach().numpy()[0])
        batch.detach()
        train.detach()
    plot(output_validations, output_predictions, hist)


def save_model(model):
    file_prefix = '-multiclass-directional-model.pyt'
    now = datetime.datetime.now()
    hour = '-'+str(now.hour)
    minute = '-'+str(now.minute)
    model_filename = str(date.today())+hour+minute+file_prefix
    model_path_to_save = 'data/model/'+model_filename
    torch.save(model.state_dict(), model_path_to_save)


# @profile
def run_epoch(model, criterion, optimiser, dataloader):
    counter = 0
    loss_accumulator = 0.
    for batch, train in dataloader:
        output_prediction = model(batch)
        loss = criterion(output_prediction, train)
        counter += 1
        loss_accumulator += float(str(loss.item()))
        loss.backward()
        optimiser.step()
        model.zero_grad()
    torch.cuda.empty_cache()
    return loss_accumulator/counter


# @profile
def train_step(model, criterion, optimiser, dataloader):
    hist = np.zeros(num_epochs)
    start_time = time.time()
    print("Training model...")
    try:
        for t in range(num_epochs):
            hist[t] += run_epoch(model, criterion, optimiser, dataloader)
            epoch_time = int(time.time() - start_time)
            print("Epoch ", t, "Mean loss: ", hist[t], "elapsed time ", epoch_time)

    except KeyboardInterrupt:
        print("Training process manually interrupted")
    finally:
        training_time = time.time() - start_time
        print("Training time: {}".format(training_time))
    save_model(model)
    model.eval()
    return model, hist


# @profile
def main():
    dataloader = load_datasets()
    model = get_model()
    model = model.cuda(device="cuda")
    weights = torch.Tensor([0.2, 1., 1.])
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    criterion = criterion.cuda(device="cuda")
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model, loss_hist = train_step(model, criterion, optimiser, dataloader)
    validation_step(model, loss_hist)


if __name__ == '__main__':
    main()

