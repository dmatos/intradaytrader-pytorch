# coding=utf-8

from os.path import join

import pandas as pd
import requests


def download(json_req, input_file_path):
    endpoint = 'http://localhost:8080/intraday/csv/lstm'
    data_dir = 'data'
    df = pd.read_csv(input_file_path)
    # print(df)
    for i in range(0, len(df)):
        print('ticker: ', df.iloc[i]['ticker'])
        json_req['tickerCode'] = df.iloc[i]['ticker']
        response = requests.post(endpoint, json=json_req)
        # print(response.text)
        filename = '{}_{}_{}.csv'.format(json_req['stockExchange'], json_req['tickerCode'], json_req['timeframe'])
        with open(join(data_dir, filename), 'w') as csv_file:
            csv_file.write(response.text)


def download_data():
    json_req = {
        "stockExchange": "B3",
        "tickerCode": "PETR4",
        "begin": "2024-07-01T00:00:00.000Z",
        "end": "2024-07-30T23:59:00.000Z",
        "timeframe": 5,
        "signal": 30,
        "signalDelay": 0,
        "chronoUnit": "MINUTES",
        "macdSlow": [18, 36, 42, 48],
        "macdFast": [6, 12, 18, 24],
        "macdSignal": [6, 12, 18, 24],
        "rsi": [],
        "rwi": [6, 12, 18, 36, 48],
        "atr": [6, 12, 18, 36, 48],
        "ado": [6, 12, 18, 36, 48],
        "bands": [6, 12, 18, 24, 36, 48],
        "slopes": [6, 12, 18, 24, 36, 48, 96]
    }
    download(json_req, 'resources/ibovespa_train_tickers.csv')


if __name__ == '__main__':
    download_data()
