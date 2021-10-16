# -*- coding: utf-8 -*-

import data_processing as dp
import pandas as pd

path = 'LoanStats_2017Q1.csv'
path_train = 'LoanStats_2017Q1_Train.csv'
path_test = 'LoanStats_2017Q1_Test.csv'

if __name__ == '__main__':
    df_train = dp.read_csv(path_train)
    df_test = dp.read_csv(path_test)

    df_train = dp.drop_post_feature(df_train)
    df_test = df_test[list(df_train.columns)]

    df_train = dp.drop_missing_feature(df_train)
    df_test = df_test[list(df_train.columns)]
    df_test.fillna(value=999, inplace=True)

    df_train = dp.format_data(df_train)
    df_test = dp.format_data(df_test)

    df_train_y = df_train['loan_status']
    df_train_x = df_train.drop(['loan_status'], axis=1)
    df_train_x = dp.dimension_reduction(df_train_x)
    df_train = pd.concat([df_train_x, df_train_y], axis=1)

    df_train = dp.vif(df_train, 10)
    newDF, woeDF = dp.iv_woe(df_train, 'loan_status')
    del_feas = newDF[newDF.IV < 0.01].Variable.tolist()
    df_train = df_train.drop(del_feas, axis=1)
    df_test = df_test[list(df_train.columns)]

    df_train_y = df_train['loan_status']
    df_train_x = df_train.drop(['loan_status'], axis=1)
    df_test_y = df_test['loan_status']
    df_test_x = df_test.drop(['loan_status'], axis=1)

    print(df_test_x.shape)
    print(df_train_x.shape)

    df_train_x, df_test_x = dp.feature_engineering(df_train_x, df_train_y, df_test_x)

    df_train_x.to_csv('x_train.csv')
    df_train_y.to_csv('y_train.csv')
    df_test_x.to_csv('x_test.csv')
    df_test_y.to_csv('y_test.csv')
