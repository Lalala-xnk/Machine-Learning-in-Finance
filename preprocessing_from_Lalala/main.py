# -*- coding: utf-8 -*-

import data_processing as dp
import svm_classifier

path = 'LoanStats_2017Q1.csv'
path_train = 'LoanStats_2017Q1_Train.csv'
path_test = 'LoanStats_2017Q1_Test.csv'

if __name__ == '__main__':
    try:
        dp.data_processing(path_train, path_test)
        print('SUCCESS')
    except:
        print('ERROR')

