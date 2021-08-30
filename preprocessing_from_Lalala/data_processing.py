# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


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
    len_ = len(df)
    missing_feature = [name for name in df.columns if (df[name].isna().sum() * 1.0 / len(df)) > 0.95]
    try:
        df.drop(missing_feature, axis=1, inplace=True)
        print('missing features dropped')
        print('missing per:', len(missing_feature) * 1.0 / len_)
    except KeyError:
        print('no missing features')
    df.fillna(0, inplace=True)
    return df


def format_data(df):
    df['term'] = [float(re.sub(r'\D', '', term)) for term in df['term']]
    df['int_rate'] = [float(re.sub(r'\D', '', int_rate)) / 100.0 for int_rate in df['int_rate']]
    df['emp_length'] = [float(re.sub(r'\D', '', str(emp_length))) for emp_length in df['emp_length']]
    df['revol_util'] = [float(re.sub(r'\D', '', revol_util)) / 100.0 for revol_util in df['revol_util']]
    df['loan_status'] = [0 if str(loan_status) in ['Fully Paid', 'In Grace Period'] else 1
                         for loan_status in df['loan_status']]

    df.drop(['issue_d', 'zip_code', 'earliest_cr_line'], axis=1, inplace=True)

    name_list = ['grade', 'sub_grade', 'home_ownership', 'emp_title', 'verification_status', 'purpose', 'title',
                 'hardship_flag', 'disbursement_method', 'debt_settlement_flag', 'pymnt_plan', 'addr_state',
                 'application_type']
    for name in name_list:
        df[name] = pd.factorize(df[name])[0].astype(float)

    return df


def dimension_reduction(df):
    df_corr = df.corr()
    df_uncorr = ~(df_corr.mask(np.tril(np.ones([len(df_corr)] * 2, dtype=bool))).abs() > 0.95).any()
    un_corr_idx = df_uncorr.loc[df_uncorr.apply(lambda x: x is True)].index
    df = df[un_corr_idx]

    scaler = StandardScaler()
    scaler.fit(df)
    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

    pca = PCA()
    pca.fit(df_scaled)

    explained_var_ratio = pca.explained_variance_ratio_
    plt.plot(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    plt.show()

    # i = 0
    # for i in range(len(df_scaled) - 1):
    #     if explained_var_ratio[i] > 10 * explained_var_ratio[i + 1]:
    #         break

    i = min(np.argwhere(explained_var_ratio < 0.01))[0]

    pca = PCA(n_components=i)
    pca.fit(df_scaled)
    df = pd.DataFrame(pca.transform(df_scaled))

    return df
