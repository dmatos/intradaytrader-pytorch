# coding=utf-8

import json
import pandas as pd
import requests
from os.path import join


def download_csv_lstm(exchange, json_req, input_file_path):
    endpoint = 'http://localhost:8080/intraday/csv/lstm'
    data_dir = 'data'
    df = pd.read_csv(input_file_path)
    # print(df)
    download_all(endpoint, data_dir, df, exchange, json_req)


def download_csv_data():
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
    download_csv_lstm(json_req['stockExchange'], json_req, 'resources/ibovespa_train_tickers.csv')


def download_candlestick(exchange, tickers_file_path):
    endpoint = 'http://localhost:8080/intraday/metadata/candlestick/{}'.format(exchange)
    data_dir = 'data'
    df = pd.read_csv(tickers_file_path)
    json_req = {
        "tickerCode": "",
        "begin": "2024-01-01T00:00:00.428Z",
        "end": "2024-08-28T23:59:59.428Z",
        "chronoUnit": "MINUTES",
        "timeframe": 1
    }
    download_all(endpoint, data_dir, df, json_req, exchange, converter_candlestick_response)


def download_all(endpoint, data_dir, df, json_req, exchange, converter=None):
    for i in range(0, len(df)):
        print('ticker: ', df.iloc[i]['ticker'])
        json_req['tickerCode'] = df.iloc[i]['ticker']
        filename = '{}_{}_{}.csv'.format(exchange, json_req['tickerCode'], json_req['timeframe'])
        response = requests.post(endpoint, json=json_req)
        # print(response.text)
        text = response.text
        if converter is not None:
            text = converter(json.loads(response.text))
        with open(join(data_dir, filename), 'w') as csv_file:
            csv_file.write(text)


def converter_candlestick_response(json_response):
    candles = json_response['candles']
    text = 'timestamp,low,high,open,close\n'
    for candle in candles:
        text = text + '{},{},{},{},{}\n'.format(
            candle['timestamp'], candle['low'], candle['high'], candle['open'], candle['close'])
    return text


if __name__ == '__main__':
    # download_csv_data()
    download_candlestick('B3', 'resources/ibovespa_train_tickers.csv')
