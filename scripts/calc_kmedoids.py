# coding=utf-8
import sys
sys.path.append('./')

from os import listdir
from os.path import isfile, join
import time

import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import kmedoids

from logger import logger


def add_col_total(df):
    df['total'] = df['high'] - df['low']
    logger.info('df with total col {}'.format(df))
    return df


def add_col_top_wick(df):
    logger.info('top_wick calculations')
    df['top_wick'] = (df['high']-df[['open', 'close']].max(axis='columns'))/df['total']
    logger.info('df with top wick col {}'.format(df))
    return df


def add_col_bottom_wick(df):
    df['bottom_wick'] = (df[['open', 'close']].min(axis='columns')-df['low'])/df['total']
    logger.info('df with bottom wick col {}'.format(df))
    return df


def add_col_body(df):
    df['body'] = abs(df['close']-df['open'])/df['total']
    logger.info('df with body col {}'.format(df))
    return df


def get_dist_matrix(numpy_array, max_items_to_fit):
    logger.info("Calculating distance matrix")
    dist_matrix = euclidean_distances(numpy_array[:max_items_to_fit])
    logger.info(dist_matrix)
    return dist_matrix


def get_optimal_num_of_clusters(dist_matrix, kmin, kmax):
    logger.info("Running DynMSC")
    start = time.time()
    dm = kmedoids.dynmsc(dist_matrix, kmax, kmin)
    logger.info("DynMSC took: {}s".format((time.time() - start)))
    logger.info("Optimal number of clusters according to the Medoid Silhouette: {}".format(dm.bestk))
    logger.info("Medoid Silhouette over range of k: {}".format(dm.losses))
    logger.info("Range of k: {}".format(dm.rangek))
    return dm.bestk


def transform_first_quarter_for_col_index(df, col_index):
    df.loc[df[col_index] < 0.25, col_index] = 1
    return df


def transform_second_quarter_for_col_index(df, col_index):
    df.loc[(0.25 <= df[col_index]) & (df[col_index] < 0.5), col_index] = 2
    return df


def transform_third_quarter_for_col_index(df, col_index):
    df.loc[(0.5 <= df[col_index]) & (df[col_index] < 0.75), col_index] = 3
    return df


def transform_last_quarter_for_col_index(df, col_index):
    df.loc[0.75 <= df[col_index], col_index] = 4
    return df


def transform_into_quarter_classes(df):
    logger.info('transform into classes of quarters')
    for index in ['body', 'bottom_wick', 'top_wick']:
        df = transform_last_quarter_for_col_index(df, index)
        df = transform_third_quarter_for_col_index(df, index)
        df = transform_second_quarter_for_col_index(df, index)
        df = transform_first_quarter_for_col_index(df, index)
    logger.info('DF transformed into classes:\n{}'.format(df))
    return df


def get_kmedoids(dist_matrix, nclusters):
    logger.info("Running FasterPAM")
    start = time.time()
    centers = kmedoids.fasterpam(dist_matrix, nclusters)
    logger.info("FasterPAM took: {}s".format((time.time() - start)))
    logger.info('Loss is: {}'.format(centers.loss))
    logger.info(centers)
    return centers


def add_cols_for_relative_bodies_and_wicks(df):
    body_wicks_df = df.pipe(add_col_total).pipe(add_col_body).pipe(add_col_top_wick).pipe(add_col_bottom_wick)
    body_wicks_df.fillna(0, inplace=True)
    return body_wicks_df


def get_medoids_for_body_and_wicks(concat_df):
    body_wicks_df = add_cols_for_relative_bodies_and_wicks(concat_df)
    body_wicks_df.drop(columns=['timestamp', 'open', 'low', 'high', 'close', 'total'], inplace=True)
    body_wicks_classes_df = transform_into_quarter_classes(body_wicks_df)
    numpy_array = body_wicks_classes_df.to_numpy()
    dist_matrix = get_dist_matrix(numpy_array, max_items_to_fit=50000)
    nclusters = 64
    # nclusters = get_optimal_num_of_clusters(dist_matrix, 50, 100)
    medoids_result = get_kmedoids(dist_matrix, nclusters)
    logger.info('medoids: \n{}'.format(concat_df.iloc[medoids_result.medoids]))
    return medoids_result


def add_col_for_black_filled_candles(df):
    # Black Filled Candlesticks occur when the close is greater than the prior close but lower than the open
    df['black_filled'] = df.loc[(df['close'] >= df['close'].shift()) & (df['close'] < df['open']), 'close']
    df['black_filled'].fillna(0, inplace=True)
    df.loc[df['black_filled'] > 0, 'black_filled'] = 1
    return df


def add_col_for_black_hollow_candles(df):
    # Black Hollow Candlesticks occur when the close is greater than the prior close and the open.
    df['black_hollow'] = df.loc[(df['close'] >= df['close'].shift()) & (df['close'] >= df['open']), 'close']
    df['black_hollow'].fillna(0, inplace=True)
    df.loc[df['black_hollow'] > 0, 'black_hollow'] = 1
    return df


def add_col_for_red_filled_candles(df):
    # Red Filled Candlesticks occur when the close is below the open and prior close.
    df['red_filled'] = df.loc[(df['close'] < df['close'].shift()) & (df['close'] < df['open']), 'close']
    df['red_filled'].fillna(0, inplace=True)
    df.loc[df['red_filled'] > 0, 'red_filled'] = 1
    return df


def add_col_for_red_hollow_candles(df):
    # Red Hollow Candlesticks occur when the close is greater than the open but lower than the prior close.
    df['red_hollow'] = df.loc[(df['close'] < df['close'].shift()) & (df['close'] >= df['open']), 'close']
    df['red_hollow'].fillna(0, inplace=True)
    df.loc[df['red_hollow'] > 0, 'red_hollow'] = 1
    return df


def add_cols_for_red_candlestick(df):
    df = add_col_for_black_filled_candles(df)
    df = add_col_for_black_hollow_candles(df)
    df = add_col_for_red_filled_candles(df)
    df = add_col_for_red_hollow_candles(df)
    logger.info('Dataframe with red candlestick\n{}'.format(df))
    return df


def get_max_min_body(df):
    df['max_min_body'] = abs(df['close']-df['open'])
    max_body = df['max_min_body'].max()
    df['max_min_body'] = df['max_min_body'] / max_body
    return df


if __name__ == '__main__':
    timeframeInMinutes = '5'
    csv_dir = 'data/train/'+timeframeInMinutes+'m'
    concat_df = None
    csv_files = [f for f in listdir(csv_dir) if isfile(join(csv_dir, f))]
    for idx, file in enumerate(csv_files):
        logger.info('reading dataset #{} in file {}'.format(idx, file))
        df = pd.read_csv(join(csv_dir, file))
        df['timestamp'] = df['timestamp']+'-'+str(idx)
        df = add_cols_for_red_candlestick(df)
        df = get_max_min_body(df)
        df = get_max_min_body(df)
        if concat_df is not None:
            concat_df = pd.concat((concat_df, df))
        else:
            concat_df = df
    concat_df = add_cols_for_relative_bodies_and_wicks(concat_df)
    logger.info('Final dataframe:\n{}'.format(concat_df))
    # TODO ajeitar df que vai ali para o get_medoids, não podem ir os demais campos anexados
    # body_and_wicks_medoids = get_medoids_for_body_and_wicks(concat_df)
