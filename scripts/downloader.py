# coding=utf-8

from os.path import join

import pandas as pd
import requests


def download_data():
    df = pd.read_csv('resources/ticker_codes.csv')
    # print(df)
    json_req = {
        "tickerCode": "MGLU3",
        "begin": "2024-01-01T00:00:00.000Z",
        "end": "2024-05-31T23:59:00.000Z",
        "timeframe": 5,
        "signal": 30,
        "signalDelay": 0,
        "chronoUnit": "MINUTES",
        "stockExchange": "B3",
        "macdSlowEMA": 30,
        "macdFastEMA": 10,
        "macdSignal": 1515,
        "rsiNumberOfCandles": 31
    }

    endpoint = 'http://localhost:8080/intraday/csv/lstm'
    data_dir = 'data'

    for i in range(0, len(df)):
        print('exchange: ', df.iloc[i]['exchange'], ', ticker: ', df.iloc[i]['ticker'])
        json_req['tickerCode'] = df.iloc[i]['ticker']
        json_req['stockExchange'] = df.iloc[i]['exchange']
        response = requests.post(endpoint, json=json_req)
        # print(response.text)
        filename = '{}_{}.csv'.format(json_req['stockExchange'], json_req['tickerCode'])
        with open(join(data_dir, filename), 'w') as csv_file:
            csv_file.write(response.text)


if __name__ == '__main__':
    download_data()
