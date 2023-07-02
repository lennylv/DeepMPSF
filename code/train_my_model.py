import matplotlib
import argparse
import torch
import random
import numpy as np
from Methods import split_shuffle_val_train_csvfile, get_train_data_csv
from My_Model import My_Model_training
matplotlib.use('Agg')


def main():

    # 随机种子
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    m = 35
    n = 35
    # ！！！需要修改变量的有4处：residues； Tablefile； name_onehot_file；name_tapeEmbed_file;
    Tablefile = '../data/Table_All_train_general_ST.csv'
    residues = 'ST'
    traintype = 'general'

    # 划分,shuffle： val 和train
    # 生成已经 分割 且 shuffle 好的 Table_val_file 和 Table_train_file
    shuffled_table_df_val, shuffled_table_df_positive_train, shuffled_table_df_negative_train = split_shuffle_val_train_csvfile(Tablefile)
    num = int(len(shuffled_table_df_negative_train) / (1 *len(shuffled_table_df_positive_train))) # number of loops
    if num > 25:
        num = 25

    if torch.cuda.is_available():
        device = torch.device('cuda:2')
        print('GPU is available.')
    else:
        device = None

    print('Training Loop is %d' % num)
    for idx in range(num):  # loop to get x_train and y_train
        print('------------------Training_id is %d' % idx)
        print('yeah!ori_local_global!!!')
        df_train = get_train_data_csv(idx, shuffled_table_df_positive_train, shuffled_table_df_negative_train)
        if traintype == 'general':
            modelname = "general_model_{:s}".format(residues)
            My_Model_training(df_train, shuffled_table_df_val, modelname, idx, device)

if __name__ == "__main__":
    main()
