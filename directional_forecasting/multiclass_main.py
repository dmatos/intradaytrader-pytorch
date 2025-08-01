# coding=utf-8

import sys

sys.path.append('./')

from directional_forecasting.model_controller import save_model, get_last_model
# custom
from directional_dataset_classes import DirectionalClassesDataset
from logger import logger
# pure python
from os import listdir
from os.path import join, isfile
import time
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
validation_file = 'B3_PETR3_5.csv'

num_epochs = 10000
batch_size = 32


def load_dataset(dir_to_read, filename, steps_ahead, sample_size, targets):
    return DirectionalClassesDataset(
        join(dir_to_read, filename),
        n_rowsto_drop=60,
        sample_size=sample_size,
        steps_ahead=steps_ahead,
        targets=targets
    )


# @profile
def load_datasets(steps_ahead, sample_size, targets):
    # Load and preprocess the data

    csv_files = [f for f in listdir(train_dir) if isfile(join(train_dir, f))]
    datasets = []
    for idx, file in enumerate(csv_files):
        logger.info('reading dataset #' + str(idx) + ' in file: ' + str(file))
        dataset = load_dataset(train_dir, file, steps_ahead, sample_size, targets)
        datasets.append(dataset)

    dataset = ConcatDataset(datasets)
    return DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=False)


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


def validation_step(model, hist, steps_ahead, sample_size, targets):
    validation_dataset = load_dataset(validation_dir, validation_file, steps_ahead, sample_size, targets)
    dataloader = DataLoader(validation_dataset, shuffle=False, batch_size=1)
    logger.info("Validation dataset shape " + str(validation_dataset.df.shape))
    output_predictions = []
    output_validations = []
    for batch, train in dataloader:
        predictions = torch.argmax(model(batch), 1)
        output_predictions.append(predictions.to("cpu").detach().numpy()[0])
        output_validations.append(train.to("cpu").detach().numpy()[0])
        batch.detach()
        train.detach()
    plot(output_validations, output_predictions, hist)


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
    # torch.cuda.empty_cache()
    return loss_accumulator/counter


# @profile
def train_step(model, criterion, optimiser, dataloader):
    hist = np.zeros(num_epochs)
    start_time = time.time()
    logger.info("Training model...")
    try:
        for t in range(num_epochs):
            hist[t] += run_epoch(model, criterion, optimiser, dataloader)
            epoch_time = int(time.time() - start_time)
            logger.info("Epoch " + str(t) + " Mean loss: " + str(hist[t]) + " elapsed time " + str(epoch_time))

    except KeyboardInterrupt:
        logger.warn("Training process manually interrupted")
    finally:
        training_time = time.time() - start_time
        logger.info("Training time: {}".format(training_time))
    save_model(model)
    model.eval()
    return model, hist


# @profile
def main():
    model, params, _ = get_last_model()
    # get_last_modelmodel, params, filename = load_model_from_disk_2(validation=False)
    dataloader = load_datasets(params["steps_ahead"], params["sample_size"], params["targets"])
    model = model.cuda(device=torch.device("cuda:0"))
    # weights = torch.Tensor([1., 2., 2.])
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.cuda(device=torch.device("cuda:0"))
    optimiser = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    model, loss_hist = train_step(model, criterion, optimiser, dataloader)
    validation_step(model, loss_hist, params["steps_ahead"], params["sample_size"], params["targets"])


if __name__ == '__main__':
    main()

