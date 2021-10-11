# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from time import *
from scipy.stats import mode
import warnings

warnings.filterwarnings('ignore')


def read_csv(path):
    df = pd.read_csv(path, header=1, low_memory=False)
    df.drop(df.tail(2).index, inplace=True)
    return df


def drop_post_feature(df):
    post_feature = ['initial_list_status', 'out_prncp', 'out_prncp_inv', 'total_pymnt',
                    'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
                    'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
                    'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d']

    try:
        df.drop(post_feature, axis=1, inplace=True)
        print('post features dropped')
    except KeyError:
        print('no post features')

    return df


def drop_missing_feature(df):
    missing_feature_90 = [name for name in df.columns if (df[name].isna().sum() * 1.0 / len(df)) > 0.9]
    missing_feature_10 = [name for name in df.columns if (df[name].isna().sum() * 1.0 / len(df)) < 0.1]

    try:
        df.drop(missing_feature_90, axis=1, inplace=True)
        df.dropna(axis=0, how='any', subset=missing_feature_10, inplace=True)
        df.fillna(999, inplace=True)
        print('missing features dropped')
    except KeyError:
        print('no missing features')

    return df


def format_data(df):
    df['term'] = [float(re.sub(r'\D', '', term)) for term in df['term']]
    df['int_rate'] = [float(re.sub(r'\D', '', int_rate)) / 100.0 for int_rate in df['int_rate']]
    df['emp_length'] = [float(re.sub(r'\D', '', str(emp_length))) for emp_length in df['emp_length']]
    df['revol_util'] = [float(re.sub(r'\D', '', str(revol_util))) / 100.0 for revol_util in df['revol_util']]
    df['loan_status'] = [0 if str(loan_status) in ['Fully Paid', 'In Grace Period', 'Current'] else 1
                         for loan_status in df['loan_status']]
    df['issue_d'] = [mktime(strptime(issue_d, '%b-%Y')) if issue_d else 0 for issue_d in df['issue_d']]
    df['earliest_cr_line'] = [mktime(strptime(earliest_cr_line, '%b-%Y')) if earliest_cr_line else 0
                              for earliest_cr_line in df['earliest_cr_line']]
    df['sub_grade'] = [ord(sub_grade[0]) - 64 + (int(sub_grade[1]) - 1) / 5.0 for sub_grade in df['sub_grade']]
    df['zip_code'] = [int(str(zip_code)[:3]) for zip_code in df['zip_code']]
    df.drop(['grade', 'addr_state', 'emp_title'], axis=1, inplace=True)

    # binning_list = []
    # for col in list(df.columns):
    #     if type(df[col][0]) == str and int(df[col].nunique()) >= 100:
    #         binning_list.append(col)

    name_list = ['home_ownership', 'verification_status', 'purpose', 'title', 'hardship_flag', 'disbursement_method',
                 'debt_settlement_flag', 'pymnt_plan', 'application_type']
    for name in name_list:
        df[name] = pd.factorize(df[name])[0].astype(float)

    return df


def dimension_reduction(df):
    df_corr = df.corr()
    df_uncorr = ~(df_corr.mask(np.tril(np.ones([len(df_corr)] * 2, dtype=bool))).abs() > 0.9).any()
    un_corr_idx = df_uncorr.loc[df_uncorr.apply(lambda x: x is True)].index
    df = df[un_corr_idx]

    duplicate_features = []
    for col in df.columns:
        if mode(df[col])[1][0] * 1.0 / len(df) > 0.9:
            duplicate_features.append(col)
    df.drop(duplicate_features, axis=1, inplace=True)

    # scaler = StandardScaler()
    # scaler.fit(df)
    # df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
    #
    # pca = PCA()
    # pca.fit(df_scaled)
    #
    # # explained_var_ratio = pca.explained_variance_ratio_
    # plt.plot(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    # plt.show()
    #
    # # i = 0
    # # for i in range(len(df_scaled) - 1):
    # #     if explained_var_ratio[i] > 10 * explained_var_ratio[i + 1]:
    # #         break
    #
    # # i = min(np.argwhere(explained_var_ratio < 0.01))[0]
    #
    # pca = PCA(n_components=0.9)
    # pca.fit(df_scaled)
    # df = pd.DataFrame(pca.transform(df_scaled))

    return df
