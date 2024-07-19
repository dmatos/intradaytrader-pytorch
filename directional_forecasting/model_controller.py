# coding=utf-8
import sys
sys.path.append('./')

import datetime
import torch
from datetime import date
from ml.lstm import IntradayTraderLSTM


def get_last_model_params():
    last_model_filename = '2024-07-18-22-25-last.pyt'
    return {
        "learning_rate": 1e-4,
        "input_dim": 79,
        "hidden_dim": 200,
        "num_layers": 6,
        "output_dim": 3,
        "steps_ahead": 24,
        "dropout": 0.20,
        "sample_size": 96,
        "targets": [-0.005, 0.005, 2.]
    }, last_model_filename


def get_last_model_params_beta():
    last_model_filename_beta = '2024-07-10-23-57-last.pyt'
    return {
        "learning_rate": 1e-3,
        "input_dim": 54,
        "hidden_dim": 100,
        "num_layers": 5,
        "output_dim": 3,
        "steps_ahead": 24,
        "dropout": 0.2,
        "sample_size": 288,
        "target_percent": 0.005
    }, last_model_filename_beta


def get_last_model():
    params, filename = get_last_model_params()
    return IntradayTraderLSTM(
        input_size=params["input_dim"],
        hidden_size=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        output_size=params["output_dim"]), params, filename


def get_last_model_beta():
    params, filename = get_last_model_params_beta()
    return IntradayTraderLSTM(
        input_size=params["input_dim"],
        hidden_size=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        output_size=params["output_dim"]), params, filename


def load_model_from_disk_2(validation=True):
    model, params, filename = get_last_model()
    path_to_load_model = 'data/model/'+filename
    model.load_state_dict(torch.load(path_to_load_model))
    model = model.cuda()
    if validation:
        model.eval()
    else:
        model.train()
    return model, params, filename


def save_model(model, filename="last"):
    file_suffix = '-'+filename+'.pyt'
    now = datetime.datetime.now()
    hour = '-'+str(now.hour)
    minute = '-'+str(now.minute)
    model_filename = str(date.today())+hour+minute+file_suffix
    path_to_save_model = 'data/model/'+model_filename
    torch.save(model.state_dict(), path_to_save_model)

