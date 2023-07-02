import matplotlib
import argparse

import torch

from Methods import split_shuffle_val_train_csvfile, get_test_data_csv
from My_Model import My_Model_Blindtesting_musculus
import sys



def main():
    # ！！！需要修改变量的有5处：residues； Tablefile_train； Tablefile_test； name_onehot_file_train； name_onehot_file_test
    predicttype = 'general'
    mode = 'test'
    residues = 'ST'
    Tablefile_train = '../data/Table_All_train_general_ST.csv'
    Tablefile_test = '../data/Table_blind_musculus_ST.csv'

    if predicttype == 'general':
        if mode == 'blind_predict':
            modelname = "general_model_{:s}".format(residues)
            # X_val, Y_val, _, _, _ = get_val_data(training_set, residues, m, n)
            # 划分,shuffle： val 和train
            # 生成已经 分割 且 shuffle 好的 Table_val_file 和 Table_train_file
            shuffled_table_df_val, _, _ = split_shuffle_val_train_csvfile(Tablefile_train)
        if mode == 'test':
            modelname = 'general_model_{:s}'.format(residues)
            shuffled_table_df_val, _, _ = split_shuffle_val_train_csvfile(Tablefile_train)
            shuffled_table_df_test = get_test_data_csv(Tablefile_test)

    My_Model_Blindtesting_musculus(shuffled_table_df_test, modelname, shuffled_table_df_val, torch.device('cuda:2'))


if __name__ == "__main__":
    main()

