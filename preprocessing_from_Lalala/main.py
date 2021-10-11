# -*- coding: utf-8 -*-

import data_processing as dp
import pandas as pd

path = 'LoanStats_2017Q1.csv'

if __name__ == '__main__':
    df = dp.read_csv(path)
    df = dp.drop_post_feature(df)
    df = dp.drop_missing_feature(df)
    df = dp.format_data(df)

    df_target = df['loan_status']
    df_training = df.drop(['loan_status'], axis=1)
    df_training = dp.dimension_reduction(df_training)
    df = pd.concat([df_training, df_target], axis=1)

    df = dp.vif(df, 10)
    newDF, woeDF = dp.iv_woe(df, 'loan_status')
    del_feas = newDF[newDF.IV < 0.01].Variable.tolist()
    df = df.drop(del_feas, axis=1)
    df_target = df['loan_status']
    df_training = df.drop(['loan_status'], axis=1)

    df_training = dp.feature_engineering(df_training, df_target)
