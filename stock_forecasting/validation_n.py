# coding=utf-8

import sys
sys.path.append('./')
# custom
from stock_dataset import StockDataset
from ml.lstm import StockPriceLSTM
# pure python
from os.path import join
import math
# other libs
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
# plot
import matplotlib.pyplot as plt
from mpl_interactions import panhandler, zoom_factory
import seaborn as sns
sns.set_style("darkgrid")


model_name = '2024-06-21-stock-model.pyt'
model_path_to_load = 'data/model/'+model_name
csv_dir = 'data'
validation_dir = join(csv_dir, 'validation')
csv_to_validate = join(validation_dir, 'B3_PRIO3.csv')

input_dim = 12
hidden_dim = 50
num_layers = 1
output_dim = 60

sample_size = 9600
batch_size = 1

if __name__ == '__main__':

    model = StockPriceLSTM(input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path_to_load))
    model = model.cuda()
    model.eval()

    validation_dataset = StockDataset(csv_to_validate, n_rowsto_drop=14, sample_size=sample_size, steps_ahead=output_dim)
    dataloader = DataLoader(validation_dataset, shuffle=False, batch_size=batch_size)

    print("Validation dataset shape ", validation_dataset.df.shape)

    n_predictions_ahead = []
    predictions_array = []
    validation_array = []

    index = 0
    with torch.no_grad():
        for batch, validation in dataloader:

            # TODO aqui que o bicho pega, pai
            # percebi que é bem mais complexo que isso...
            # preciso ter a previsão de open,close,high,low
            # porque precisaria calcular as demais variáveis (rsi, macd, and shit...)

            predictions = model(batch)

            validation = validation.to("cpu").detach().numpy()
            validation = validation_dataset.invert_standardize(validation)
            # print("validation data: ", validation)

            predictions = predictions.to("cpu").detach().numpy()
            predictions = validation_dataset.invert_standardize(predictions)

            # print("Validation values: ", validation)
            # print("Predicted values: ", predictions)

            # predictions_array.append(predictions[0][0])
            # validation_array.append(validation[0][0])

            testScore = math.sqrt(mean_squared_error(predictions, validation))
            print('Test Score: %.2f RMSE' % testScore)

            if index % output_dim == 0:
                validation = validation[:, :int(output_dim/4)]
                predictions = predictions[:, :int(output_dim/4)]

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

                # Enable scrolling and panning with the help of MPL
                # Interactions library function like panhandler.
                pan_handler = panhandler(fig)
                plt.show()

            index += 1

    # plot predictions against original data
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    plt.subplot(1, 2, 1)
    sns.lineplot(x=[i for i in range(len(validation_array))], y=validation_array, label="Data", color='royalblue')
    ax = sns.lineplot(x=[i for i in range(len(predictions_array))], y=predictions_array, label="Training Prediction (LSTM)", color='tomato')
    ax.set_title('Stock price', size=14, fontweight='bold')
    ax.set_xlabel("Hours", size=14)
    ax.set_ylabel("Cost (BRL)", size=14)
    ax.set_xticklabels('', size=10)
    zoom_factory(ax)

    # Enable scrolling and panning with the help of MPL
    # Interactions library function like panhandler.
    pan_handler = panhandler(fig)
    plt.show()

