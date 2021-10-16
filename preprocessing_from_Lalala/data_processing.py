# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from collections import Counter
from math import sqrt
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from time import *
from scipy.stats import chi2_contingency
from scipy.stats import mode
import warnings

warnings.filterwarnings('ignore')


def read_csv(path):
    df = pd.read_csv(path, header=0, low_memory=False)
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
    # df['emp_length'] = [float(re.sub(r'\D', '', str(emp_length))) for emp_length in df['emp_length']]
    df['revol_util'] = [float(re.sub(r'\D', '', str(revol_util))) / 100.0 for revol_util in df['revol_util']]
    df['loan_status'] = [0 if str(loan_status) in ['Fully Paid', 'In Grace Period', 'Current'] else 1
                         for loan_status in df['loan_status']]
    # df['issue_d'] = [int(mktime(strptime(issue_d, '%b-%Y')) / 1e8) if issue_d else 0 for issue_d in df['issue_d']]
    # df['earliest_cr_line'] = [int(mktime(strptime(earliest_cr_line, '%b-%Y')) / 1e8) if earliest_cr_line else 0
    #                           for earliest_cr_line in df['earliest_cr_line']]
    df['sub_grade'] = [ord(sub_grade[0]) - 64 + (int(sub_grade[1]) - 1) / 5.0 for sub_grade in df['sub_grade']]
    # df['zip_code'] = [int(str(zip_code)[:2]) for zip_code in df['zip_code']]
    df.drop(['grade', 'zip_code', 'emp_title', 'issue_d', 'earliest_cr_line'], axis=1, inplace=True)

    df.emp_length.replace({"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
                           "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
                           "8 years": 8, "9 years": 9, "10+ years": 10, 'n/a': 0}, inplace=True)

    # binning_list = []
    # for col in list(df.columns):
    #     if type(df[col][0]) == str and int(df[col].nunique()) >= 100:
    #         binning_list.append(col)

    name_list = ['home_ownership', 'verification_status', 'purpose', 'title', 'hardship_flag', 'disbursement_method',
                 'debt_settlement_flag', 'pymnt_plan', 'addr_state', 'application_type']
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


def vif(x, threshold):
    x_m = np.matrix(x)
    vif_list = [VIF(x_m, i) for i in range(x_m.shape[1])]
    maxvif = pd.DataFrame(vif_list, index=x.columns, columns=["vif"])
    col_save = list(maxvif[maxvif.vif <= float(threshold)].index)
    return x[col_save]


def feature_engineering(df_train_x, df_train_y, df_test_x):
    binning_feature = []
    for col in df_train_x.columns:
        if df_train_x[col].nunique() > 15:
            binning_feature.append(col)

    for feature in binning_feature:
        splits = min(50, df_train_x[feature].nunique())
        n = df_train_x[feature].nunique()
        df_train_x[feature], df_test_x[feature] = chi_merge(df_train_x[feature], df_train_y, df_test_x[feature], splits=splits,
                                                   max_pvalue=0.05)
        print(feature + ':', n, df_train_x[feature].nunique())

    # for feature in binning_feature:
    #     # print(feature, sqrt(df[feature].var()) * 1.0 / df[feature].mean())
    #     plt.plot(list(df[feature].sort_values()))
    #     plt.title(feature + str(sqrt(df[feature].var()) * 1.0 / df[feature].mean()))
    #     plt.show()

    # tmp = df['mths_since_last_delinq'][df['mths_since_last_delinq'] < 1000]
    # plt.plot(list(tmp.sort_values()))
    # plt.show()

    return df_train_x, df_test_x


def tagcount(series, tags):
    result = []
    countseries = series.value_counts()
    for tag in tags:
        try:
            result.append(countseries[tag])
        except:
            result.append(0)
    return result


def chi_merge(feature, target, feature_test, splits, max_pvalue=0.1, max_iter=15, min_iter=2):
    tags = [0, 1]
    percent = feature.quantile([1.0 * i / splits for i in range(splits + 1)], interpolation="lower")\
        .drop_duplicates(keep="last").tolist()
    percent = percent[1:]
    np_regroup = []
    for i in range(len(percent)):
        if i == 0:
            tmp = tagcount(target[feature <= percent[i]], tags)
            tmp.insert(0, percent[i])
        elif i == len(percent) - 1:
            tmp = tagcount(target[feature > percent[i - 1]], tags)
            tmp.insert(0, percent[i])
        else:
            tmp = tagcount(target[(feature > percent[i - 1]) & (feature <= percent[i])], tags)
            tmp.insert(0, percent[i])
        np_regroup.append(tmp)
    np_regroup = pd.DataFrame(np_regroup)
    np_regroup = np.array(np_regroup)

    i = 0
    while i <= np_regroup.shape[0] - 2:
        check = 0
        for j in range(len(tags)):
            if np_regroup[i, j + 1] == 0 and np_regroup[i + 1, j + 1] == 0:
                check += 1
        if check > 0:
            np_regroup[i, 1:] = np_regroup[i, 1:] + np_regroup[i + 1, 1:]
            np_regroup[i, 0] = np_regroup[i + 1, 0]
            np_regroup = np.delete(np_regroup, i + 1, 0)
            i = i - 1
        i = i + 1

    chi_table = np.array([])
    for i in np.arange(np_regroup.shape[0] - 1):
        temparray = np_regroup[i:i + 2, 1:]
        pvalue = chi2_contingency(temparray, correction=False)[1]
        chi_table = np.append(chi_table, pvalue)
    temp = max(chi_table)

    while True:
        if len(chi_table) < max_iter and temp <= max_pvalue:
            break
        if len(chi_table) < min_iter:
            break

        num = np.argwhere(chi_table == temp)
        for i in range(num.shape[0] - 1, -1, -1):
            chi_min_index = num[i][0]
            np_regroup[chi_min_index, 1:] = np_regroup[chi_min_index, 1:] + np_regroup[chi_min_index + 1, 1:]
            np_regroup[chi_min_index, 0] = np_regroup[chi_min_index + 1, 0]

            np_regroup = np.delete(np_regroup, chi_min_index + 1, 0)

            if chi_min_index == np_regroup.shape[0] - 1:
                temparray = np_regroup[chi_min_index - 1:chi_min_index + 1, 1:]
                chi_table[chi_min_index - 1] = chi2_contingency(temparray, correction=False)[1]
                chi_table = np.delete(chi_table, chi_min_index, axis=0)

            elif chi_min_index == 0:
                temparray = np_regroup[chi_min_index:chi_min_index + 2, 1:]
                chi_table[chi_min_index] = chi2_contingency(temparray, correction=False)[1]
                chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)

            else:
                temparray = np_regroup[chi_min_index - 1:chi_min_index + 1, 1:]
                chi_table[chi_min_index - 1] = chi2_contingency(temparray, correction=False)[1]
                temparray = np_regroup[chi_min_index:chi_min_index + 2, 1:]
                chi_table[chi_min_index] = chi2_contingency(temparray, correction=False)[1]
                chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)

        temp = max(chi_table)

    val = 0
    feature = feature.apply(lambda x: val if x <= np_regroup[0, 0] else x)
    feature_test = feature_test.apply(lambda x: val if x <= np_regroup[0, 0] else x)
    val += 1
    for i in range(1, np_regroup.shape[0] - 1):
        feature = feature.apply(lambda x: val if np_regroup[i - 1, 0] < x <= np_regroup[i, 0] else x)
        feature_test = feature_test.apply(lambda x: val if np_regroup[i - 1, 0] < x <= np_regroup[i, 0] else x)
        val += 1
    feature = feature.apply(lambda x: val if x > np_regroup[np_regroup.shape[0] - 2, 0] else x)
    feature_test = feature_test.apply(lambda x: val if x > np_regroup[np_regroup.shape[0] - 2, 0] else x)

    return feature, feature_test


def iv_woe(data, target, bins=10):
    newDF, woeDF = pd.DataFrame(), pd.DataFrame()
    cols = data.columns
    for ivars in cols[~cols.isin([target])]:
        # 数据类型在bifc中、且数据>10则分箱
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars])) > 10):
            binned_x = pd.qcut(data[ivars], bins, duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events'] / d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
#        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(), 6)))
        temp = pd.DataFrame({"Variable": [ivars], "IV": [d['IV'].sum()]}, columns=["Variable", "IV"])
        newDF = pd.concat([newDF, temp], axis=0)
        woeDF = pd.concat([woeDF, d], axis=0)
    return newDF, woeDF
