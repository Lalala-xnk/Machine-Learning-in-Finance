# -*- coding: utf-8 -*-

import data_processing as dp
import pandas as pd

path = 'LoanStats_2017Q1.csv'
path_train = 'LoanStats_2017Q1_Train.csv'
path_test = 'LoanStats_2017Q1_test.csv'

if __name__ == '__main__':
    df = dp.read_csv(path_train)
    df = dp.drop_post_feature(df)
    df = dp.drop_missing_feature(df)
    # print(list(df.columns))
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

    df_training.to_csv('x_train.csv')
    df_target.to_csv('y_train.csv')
    df_test = dp.read_csv(path_test)
    df_test[list(df_training.columns)].to_csv('x_test.csv')
    df_test['loan_status'].to_csv('y_test.csv')
