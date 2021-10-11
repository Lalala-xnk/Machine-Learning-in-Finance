# -*- coding: utf-8 -*-

import data_processing as dp

path = 'LoanStats_2017Q1.csv'

if __name__ == '__main__':
    df = dp.read_csv(path)
    df = dp.drop_post_feature(df)
    df = dp.drop_missing_feature(df)
    df = dp.format_data(df)

    df_target = df['loan_status']
    df_training = df.drop(['loan_status'], axis=1)
    # print(df_training)
    df_training = dp.dimension_reduction(df_training)
    # print(df_training)
