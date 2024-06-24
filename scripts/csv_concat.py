# coding=utf-8

from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd

csv_dir = 'data'
result_dir = join('data', 'input')

n_rows_to_drop = 14

# cols_to_keep = ['timestampUTC', 'open', 'low', 'high', 'close']
cols_to_keep = None

if __name__ == '__main__':
    csv_files = [f for f in listdir(csv_dir) if isfile(join(csv_dir, f))]
    final_df = None
    for idx, file in enumerate(csv_files):
        print('reading dataset #', idx, 'in file: ', file)
        df = pd.read_csv(join(csv_dir, file))
        if cols_to_keep is not None:
            df = df[cols_to_keep]
        # df['date'] = pd.to_datetime(df['timestampUTC'], unit='s', errors='coerce')
        df['groupId'] = file
        df.drop(columns=['timestampUTC'], inplace=True)
        # TODO trocar timestampUTC por um index qualquer
        df.drop([df.index[i] for i in range(n_rows_to_drop)], inplace=True, axis='index')
        df['index'] = np.arange(df.shape[0])
        if final_df is not None:
            final_df = pd.concat((final_df, df))
        else:
            final_df = df
    print(final_df)
    result_csv_path = join(result_dir, 'stocks_input.csv')
    final_df.to_csv(result_csv_path, sep=',', index=False, encoding='utf-8')
    print('Write to ', result_csv_path, ' finished')



