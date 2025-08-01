# coding=utf-8
import sys

sys.path.append('./')

from directional_forecasting.model_controller import load_model_from_disk_2

from directional_forecasting.multiclass_main import load_dataset
# custom
# pure python
from os.path import join
# other libs
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.metrics import classification_report
# plot
import matplotlib.pyplot as plt
from mpl_interactions import zoom_factory
import seaborn as sns
sns.set_style("darkgrid")


csv_dir = 'data'
validation_dir = join(csv_dir, 'validation')
validation_file = 'B3_PETR3_5.csv'


def plot(validations, predictions):
    """plot predictions against original data
    """
    plt.plot()
    sns.scatterplot(x=[i for i in range(len(validations))], y=validations, label="Data", color='royalblue')
    ax = sns.scatterplot(x=[i for i in range(len(predictions))], y=predictions, label="Training Prediction (LSTM)", color='tomato')
    ax.set_title('Stock price', size=14, fontweight='bold')
    ax.set_xlabel("Hours", size=14)
    ax.set_ylabel("Cost (BRL)", size=14)
    ax.set_xticklabels('', size=10)
    zoom_factory(ax)

    # Enable scrolling and panning with the help of MPL
    # Interactions library function like panhandler.
    plt.show()


def report_confusion_matrix(validations, predictions, targets):
    confusion_matrix = metrics.confusion_matrix(validations, predictions)
    print(confusion_matrix)

    target_names = [str(i) for i in targets]
    print(classification_report(validations, predictions, target_names=target_names))

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=target_names)
    cm_display.plot()
    plt.show()


if __name__ == '__main__':

    model, params, _ = load_model_from_disk_2()

    validation_dataset = load_dataset(validation_dir, validation_file,
                                      steps_ahead=params['steps_ahead'],
                                      sample_size=params['sample_size'],
                                      targets=params['targets'])
    dataloader = DataLoader(validation_dataset, shuffle=False, batch_size=1)

    print("Validation dataset shape ", validation_dataset.df.shape)

    output_predictions = []
    output_validations = []
    with torch.no_grad():
        for batch, validation in dataloader:
            predictions = torch.argmax(model(batch), 1)
            output_predictions.append(predictions.to("cpu").detach().numpy()[0])
            output_validations.append(validation.to("cpu").detach().numpy()[0])

    report_confusion_matrix(output_validations, output_predictions, params['targets'])
    plot(output_validations, output_predictions)
