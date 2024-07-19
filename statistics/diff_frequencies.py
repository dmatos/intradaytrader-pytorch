# coding=utf-8

import sys
sys.path.append('./')

import pandas as pd
import numpy as np

from mpl_interactions import zoom_factory
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join

from scripts.downloader import download
from logger import logger

data_dir = 'data'
output_dir = join(data_dir, 'stats')
input_dir = join(output_dir, 'input')
input_file_path = 'resources/ibovespa_ticker_codes.csv'

frequency_buckets = np.array([-0.01, -0.005, -0.0025, 0, 0.0025, 0.005, 0.01, 2.])

timeframes = [15, 30, 45, 60, 90, 120]

json_req = {
    "stockExchange": "B3",
    "tickerCode": "PETR4",
    "begin": "2024-01-01T00:00:00.000Z",
    "end": "2024-07-15T23:59:00.000Z",
    "timeframe": 5,
    "signal": 30,
    "signalDelay": 0,
    "chronoUnit": "MINUTES",
    "macdSlow": [],
    "macdFast": [],
    "macdSignal": [],
    "rsi": [],
    "rwi": [],
    "atr": [],
    "ado": [],
    "bands": [],
    "slopes": []
}


def download_csv_for_timeframes(timeframes):
    for t in timeframes:
        json_req['timeframe'] = t
        download(json_req, input_file_path)


def read_ticker_codes():
    return pd.read_csv(input_file_path)['ticker']


def frequencies(df):
    frequencies = np.zeros(len(frequency_buckets))
    previous_close = df.iloc[0]['close']
    date_format = '%Y-%m-%d %H:%M:%S'

    for i in range(1, len(df)):
        current_day = pd.to_datetime(df.index[i], format=date_format).day
        last_day = pd.to_datetime(df.index[i-1], format=date_format).day
        if current_day != last_day:
            continue
        current_close = df.iloc[i]['close']
        diff = current_close - previous_close
        percentual_diff = diff/previous_close
        previous_close = current_close
        bucket_index = np.argmax(frequency_buckets > percentual_diff)
        logger.debug('value {} bucket_index {}'.format(percentual_diff, bucket_index))
        logger.debug("bucket ceil: {} ".format(frequency_buckets[bucket_index]))
        frequencies[bucket_index] += 1
    logger.info('absolute fr: {}'.format(frequencies))
    relative_frequencies = frequencies/sum(frequencies)
    logger.info('relative fr: {}'.format(relative_frequencies))
    return frequencies, relative_frequencies


def generate_frequencies_files():
    tickers_df = read_ticker_codes()
    for ticker in tickers_df:
        code = ticker
        ticker_dict_freqs = {}
        ticker_dict_rel_freqs = {}
        for timeframe in timeframes:
            file = 'B3_{}_{}.csv'.format(code, timeframe)
            file_path = join(input_dir, file)
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['timestampUTC'], unit='s', errors='coerce')
            df.set_index('date', inplace=True)
            df.drop(columns=['timestampUTC'], inplace=True)
            logger.info("ticker {} timeframe {}".format(ticker, timeframe))
            freqs, rel_freqs = frequencies(df)
            ticker_dict_freqs[timeframe] = {}
            ticker_dict_rel_freqs[timeframe] = {}
            for index in range(0, len(frequency_buckets)):
                ticker_dict_freqs[timeframe][frequency_buckets[index]] = freqs[index]
                ticker_dict_rel_freqs[timeframe][frequency_buckets[index]] = rel_freqs[index]
        df_freqs = pd.DataFrame.from_dict(ticker_dict_freqs)
        df_freqs.to_csv(join(output_dir, ticker+'_abs.csv'), sep=',')
        df_rel_freqs = pd.DataFrame.from_dict(ticker_dict_rel_freqs)
        df_rel_freqs.to_csv(join(output_dir, ticker+'_rel.csv'), sep=',')


def read_frequencies_files(plot=False):
    tickers_df = read_ticker_codes()
    tickers_of_interest = []
    for ticker in tickers_df:
        # df_freqs = pd.read_csv(join(output_dir, '{}_abs.csv'.format(ticker)), sep=',')
        # df_freqs.columns.values[0] = 'var'
        # df_freqs.set_index('var', inplace=True)
        # df_freqs.drop([-1.000], inplace=True, axis='index')
        df_rel_freqs = pd.read_csv(join(output_dir, '{}_rel.csv'.format(ticker)), sep=',')
        df_rel_freqs.columns.values[0] = 'var'
        df_rel_freqs.set_index('var', inplace=True)
        # df_rel_freqs.drop([-1.000], inplace=True, axis='index')
        logger.info("ticker: {}".format(ticker))
        logger.info("< -0.005 {}".format(df_rel_freqs['120'][-0.005]))
        logger.info(">= +0.005 {}".format(df_rel_freqs['120'][2.0]))
        logger.info("rel_freqs\n {}".format(df_rel_freqs))
        if df_rel_freqs['120'][-0.005] >= 0.2 and df_rel_freqs['120'][1.] >= 0.2:
            tickers_of_interest.append(ticker)
        if plot:
            ax = sns.heatmap(xticklabels=[x for x in timeframes], yticklabels=[y for y in frequency_buckets], data=df_rel_freqs, label="Data", color='royalblue')
            ax.set_title(ticker, size=14, fontweight='bold')
            ax.set_xlabel("Hours", size=14)
            ax.set_ylabel("Volatility", size=14)
            zoom_factory(ax)
            plt.show()
    logger.info("{} tickers of interest: {}".format(len(tickers_of_interest), tickers_of_interest))


def main():
    # download_csv_for_timeframes(timeframes)
    # generate_frequencies_files()
    read_frequencies_files(plot=False)


if __name__ == '__main__':
    main()
