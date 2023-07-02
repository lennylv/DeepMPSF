import numpy as np
import random
import pandas as pd

def to_binary(base):
    base = np.array(base)
    return np.vstack([base, 1-base])


def to_categorical(base):
    base = base.astype('int32')
    seed = np.diag([1] * (base.max() + 1))
    ret = []
    for item in base:
        tmp = []
        for v in item:
            tmp.append(seed[v])
        ret.append(np.vstack(tmp))
    return np.array(ret)


def file2str(filename):
    fr = open(filename)  #
    numline = fr.readlines()  #

    index = -1
    A = []
    F = []
    for eachline in numline:
        index += 1
        if '>' in eachline:
            A.append(index)
    A.append(index+1)

    B = []
    for eachline in numline:
        line = eachline.strip()
        listfoemline = line.split()
        B.append(listfoemline)

    name = []
    for i in range(len(A) - 1):
        K = A[i]
        input_sequence = str(B[K])
        input_sequence = input_sequence[3:-2]
        name.append(input_sequence)

    for i in range(len(A)-1):
        K = A[i]
        input_sequence = B[K + 1]
        input_sequence = str(input_sequence)
        input_sequence = input_sequence[1:-1]

        for j in range(A[i + 1] - A[i]):
            if K < A[i + 1] - 2:
                C = str(B[K + 2])
                input_sequence = input_sequence + C[1:-1]
                K += 1
        input_sequence = input_sequence.replace('\'', '')
        F.append(input_sequence)

    return name,F


def separt_positive(sequence, m, n):
    indexs = []
    for k in range(len(sequence)):
        sequence[k] = '****************************************************************************************************************************************************************************************************************************' + sequence[k] + '**********************************************************************************************************************************************************************************************************************************************************************************************'
        index = []
        kk = 0
        for i in range(len(sequence[k])):
            if sequence[k][i] == '#':
                index.append(i - kk - 1)
                kk += 1
        indexs.append(index)
    sub_sequences = []
    for i in range(len(sequence)):
        # sequence[i] = sequence[i].translate(str.maketrans('', '', '#'))
        sequence[i] = sequence[i].replace('#', '')
        for k in range(len(indexs[i])):
            sub_sequence = sequence[i][indexs[i][k] - m:indexs[i][k] + n + 1]
            sub_sequences.append(sub_sequence)
    return sub_sequences


def separt_negative(sequence, m, n):
    sequences = []
    for i in range(len(sequence)):
        if '#' in sequence[i]:
            sequences.append(sequence[i])
    sequence = sequences
    indexs = []
    for k in range(len(sequence)):
        sequence[k] = '**************************************************************************************************************************************************************************************************************************************************' + sequence[k] + '**************************************************************************************************************************************************************************************************************************************************'
        index = []
        kk = 0
        for i in range(len(sequence[k])):
            if sequence[k][i] == '#':
                index.append(i - kk - 1)
                kk += 1
        indexs.append(index)
    indexs2 = []
    for k in range(len(sequence)):
        # sequence[k] = sequence[k].translate(str.maketrans('', '', '#'))
        sequence[k] = sequence[k].replace('#', '')
        index = []
        for i in range(len(sequence[k])):
            if sequence[k][i] == 'S' or sequence[k][i] == 'T':
                index.append(i)
        indexs2.append(index)
    indexs3 = []
    for i in range(len(indexs)):
        c = [x for x in indexs[i] if x in indexs2[i]]
        d = [y for y in (indexs[i] + indexs2[i]) if y not in c]
        indexs3.append(d)
    sub_sequences = []
    for i in range(len(sequence)):
        for k in range(len(indexs3[i])):
            sub_sequence = sequence[i][indexs3[i][k] - m:indexs3[i][k] + n + 1]
            sub_sequences.append(sub_sequence)
    return sub_sequences


def separt_positive_2(sequence, m, n):
    indexs = []
    for k in range(len(sequence)):
        sequence[k] = '****************************************************************************************************************************************************************************************************************************' + sequence[k] + '**********************************************************************************************************************************************************************************************************************************************************************************************'
        index = []
        kk = 0
        for i in range(len(sequence[k])):
            if sequence[k][i] == '#':
                index.append(i - kk - 1)
                kk += 1
        indexs.append(index)
    sub_sequences = []
    for i in range(len(sequence)):
        sequence[i] = sequence[i].translate(str.maketrans('', '', '#'))
        for k in range(len(indexs[i])):
            sub_sequence = sequence[i][indexs[i][k] - m:indexs[i][k] + n + 1]
            sub_sequences.append(sub_sequence)
    return sub_sequences


def separt_negative_2(sequence, m, n):
    sequences = []
    for i in range(len(sequence)):
        if '#' in sequence[i]:
            sequences.append(sequence[i])
        else:
            print('??')
    sequence = sequences
    indexs = []
    for k in range(len(sequence)):
        sequence[k] = '**************************************************************************************************************************************************************************************************************************************************' + sequence[k] + '**************************************************************************************************************************************************************************************************************************************************'
        index = []
        kk = 0
        for i in range(len(sequence[k])):
            if sequence[k][i] == '#':
                index.append(i - kk - 1)
                kk += 1
        indexs.append(index)
    indexs2 = []
    for k in range(len(sequence)):

        sequence[k] = sequence[k].translate(str.maketrans('', '', '#'))
        index = []
        for i in range(len(sequence[k])):
            if sequence[k][i] == 'Y':
                index.append(i)
        indexs2.append(index)
    indexs3 = []
    for i in range(len(indexs)):
        c = [x for x in indexs[i] if x in indexs2[i]]
        d = [y for y in (indexs[i] + indexs2[i]) if y not in c]
        indexs3.append(d)
    sub_sequences = []
    for i in range(len(sequence)):
        for k in range(len(indexs3[i])):
            sub_sequence = sequence[i][indexs3[i][k] - m:indexs3[i][k] + n + 1]
            sub_sequences.append(sub_sequence)
    return sub_sequences


def get_test_file_ST(sequence, m, n):
    id = []
    indexs2 = []
    site_types = []
    for k in range(len(sequence)):
        sequence[k] = '**************************************************************************************************************************************************************************************************************************************************' + \
                 sequence[k] + '**************************************************************************************************************************************************************************************************************************************************'
        index = []
        site_type = []
        for i in range(len(sequence[k])):
            if sequence[k][i] == 'S' or sequence[k][i] == 'T':
                index.append(i)
                site_type.append(sequence[k][i])
        indexs2.append(index)
        id.append(np.array(index) - 241)
        site_types.append(site_type)
    sub_sequences = []
    for i in range(len(sequence)):
        for k in range(len(indexs2[i])):
            sub_sequence = sequence[i][indexs2[i][k] - m:indexs2[i][k] + n + 1]
            sub_sequences.append(sub_sequence)
    return sub_sequences, id, site_types


def get_test_file_Y(sequence, m, n):
    id = []
    site_types = []
    indexs2 = []
    for k in range(len(sequence)):
        sequence[
            k] = '**************************************************************************************************************************************************************************************************************************************************' + \
                 sequence[
                     k] + '**************************************************************************************************************************************************************************************************************************************************'
        index = []
        site_type = []
        for i in range(len(sequence[k])):
            if sequence[k][i] == 'Y':
                index.append(i)
                site_type.append(sequence[k][i])
        indexs2.append(index)
        site_types.append(site_type)
        id.append(np.array(index) - 241)
    sub_sequences = []
    for i in range(len(sequence)):
        for k in range(len(indexs2[i])):
            sub_sequence = sequence[i][indexs2[i][k] - m:indexs2[i][k] + n + 1]
            sub_sequences.append(sub_sequence)
    return sub_sequences, id, site_types


def str2dic(input_sequence):
    char = sorted(
        ['G', 'A', 'V', 'L', 'I', 'P', 'F', 'Y', 'W', 'S', 'T', 'C', 'M', 'N', 'Q', 'D', 'E', 'K', 'R', 'H'])

    char_to_index = {}
    index = 1
    result_index = []
    for c in char:
        char_to_index[c] = index
        index = index + 1
    char.append('*')
    char.append('U')
    char.append('B')
    char_to_index['*'] = 0
    char_to_index['U'] = char_to_index['D']
    char_to_index['B'] = char_to_index['D']
    for word in input_sequence:
        result_index.append(char_to_index[word])
    return result_index


def vec_to_onehot(mat, pc, kk, mmm=51):
    m = len(mat)
    return_mat = np.zeros((m, mmm, kk))
    for i in range(len(mat)):
        metrix = np.zeros((mmm, kk))
        for j in range(len(mat[i])):
            metrix[j] = pc[mat[i][j]]
        return_mat[i,:,:] = metrix
    return return_mat


def split_shuffle_val_train_csvfile(Tablefile):
    # 读取table切片总表文件，并计算所有切片样本中positive 和 negative的个数, num_val_half,生成positive行数list和negative行数list
    table_df = pd.read_csv(Tablefile, index_col=0)

    # 用于数据集划分和切割
    table_df_negative = table_df[table_df['state'] == 0]
    table_df_positive = table_df[table_df['state'] == 1]

    # 读取正负例num 以及 验证集一半的num
    num_positive = len(table_df_positive)
    num_negative = len(table_df_negative)
    num_val_half = int(num_positive/10)
    num_train_positive = num_positive - num_val_half
    num_train_negative = num_negative - num_val_half

    # 后续要看随机怎样固定
    # 分别shuffle 正例的df 和负例的df
    shuffled_table_df_positive = table_df_positive.sample(frac=1, random_state=1).reset_index(drop=True)
    shuffled_table_df_negative = table_df_negative.sample(frac=1, random_state=1).reset_index(drop=True)

    # 划分val集和train集
    shuffled_table_df_positive_val = shuffled_table_df_positive.iloc[:num_val_half]
    shuffled_table_df_negative_val = shuffled_table_df_negative.iloc[:num_val_half]
    shuffled_table_df_positive_train = shuffled_table_df_positive.iloc[num_val_half:].reset_index(drop=True)
    shuffled_table_df_negative_train = shuffled_table_df_negative.iloc[num_val_half:].sample(frac=1, random_state=1).reset_index(drop=True)

    shuffled_table_df_val = pd.concat([shuffled_table_df_positive_val, shuffled_table_df_negative_val]).sample(frac=1, random_state=1).reset_index(drop=True)
    return shuffled_table_df_val, shuffled_table_df_positive_train, shuffled_table_df_negative_train


def get_train_data_csv(idx, shuffled_table_df_positive_train, shuffled_table_df_negative_train):
    part_table_df_negative_train = shuffled_table_df_negative_train.iloc[1 * len(shuffled_table_df_positive_train) * idx : 1 * len(shuffled_table_df_positive_train) * (idx +1)]
    df_train = pd.concat([shuffled_table_df_positive_train, part_table_df_negative_train]).sample(frac=1, random_state=1).reset_index(drop=True)
    return df_train


def get_test_data_csv(Tablefile):
    # 读取test对应总表的df
    table_df = pd.read_csv(Tablefile, index_col=0)
    shuffled_table_df_test = table_df.sample(frac=1, random_state=1).reset_index(drop=True)
    return shuffled_table_df_test




# et test x,y & the indices of test data
def get_test_data(Tablefile, name_onehot_file, name_tapeEmbed_file, m, n):
    # 读入onehot特征的dict
    dict_onehot = np.load(name_onehot_file, allow_pickle=True).item()

    # 读入tapeEmbed特征的dict
    dict_tapeEmbed = np.load(name_tapeEmbed_file, allow_pickle=True).item()


    # 检验onehot和tapeEmbed的name顺序是否一致
    # onehot_name = list(df_sort_name_onehot['name'])
    # tapeEmbed_name = list(df_sort_name_tapeEmbed['name'])
    # if onehot_name == tapeEmbed_name:
    #     print('match')
    # else:
    #     print('not match!')


    # 读入切片总表
    table_df = pd.read_csv(Tablefile, index_col=0)
    table = table_df.values.tolist()

    # 按总表切割x_test_positive, x_test_negative, global_test_positive, global_test_negative;  index为左半边残基数， length-1-index为右半边残基数, xing 为切片的填充符号*
    x_test_positive = []
    x_test_negative = []
    tapeEmbed_test_positive = []
    tapeEmbed_test_negative = []
    global_test_positive = []
    global_test_negative = []
    xing = [0] * 21
    xing[0] = 1
    xing = np.array(xing).reshape(1, 21)

    # 遍历总表，根据总表对读入的onehot特征及tapeEmbed特征进行切片
    for i in range(len(table)):
        # print读入切片的进度
        # if i % 50000 == 0:
        #     print('slice {} has done'.format(i))
        #  切片，分四类：很靠左(左补0)，中间（不补零），很靠右（右补零），中间但两边都缺(左右都补零)
        name = table[i][0]
        index = table[i][1]
        length = table[i][2]
        # 左边残基个数为index个，右边残基个数为length-1-index个：分类按照，左边残基数和m比较，右边残基数和n比较
        if index < m and (length - 1 - index) >= n:
            supply = xing.repeat(m - index, axis=0)
            onehot_feature = dict_onehot[name]
            main = onehot_feature[:index + n + 1]
            slice_fea = np.concatenate((supply, main), axis=0)

        elif index >= m and (length - 1 - index) >= n:
            onehot_feature = dict_onehot[name]
            main = onehot_feature[index - m:index + n + 1]
            slice_fea = main

        elif length - 1 - index < n and index >= m:
            supply = xing.repeat(n - (length - 1 - index), axis=0)
            onehot_feature = dict_onehot[name]
            main = onehot_feature[index - m:]
            slice_fea = np.concatenate((main, supply), axis=0)

        elif index < m and length - 1 - index < n:
            supply_left = xing.repeat(m - index, axis=0)
            supply_right = xing.repeat(n - (length - 1 - index), axis=0)
            onehot_feature = dict_onehot[name]
            main = onehot_feature
            slice_fea = np.concatenate((supply_left, main, supply_right), axis=0)

        if slice_fea.shape != (m+n+1, 21):
            print(slice_fea.shape)
            print(table[i])


        # 获取tapeEmbed_feature
        tapeEmbed_feature = dict_tapeEmbed[name]

        # 获取2000长度的onehot  global特征
        global_main = dict_onehot[name]
        if len(global_main) < 2000:
            supply_left = xing.repeat(2000 - len(global_main), axis=0)
            global_fea = np.concatenate((supply_left, global_main), axis=0)
        else:
            global_fea = global_main[:2000]



        # 对切片好的特征分正负类， 得到总正例切片列表x_positive ,global_test_positive  总负例切片列表x_negative，global_test_negative
        if table[i][3] == 1:
            x_test_positive.append(slice_fea)
            tapeEmbed_test_positive.append(tapeEmbed_feature)
            global_test_positive.append(global_fea)
        else:
            x_test_negative.append(slice_fea)
            tapeEmbed_test_negative.append(tapeEmbed_feature)
            global_test_negative.append(global_fea)

    x_test_positive = np.array(x_test_positive)
    x_test_negative = np.array(x_test_negative)
    tapeEmbed_test_positive = np.array(tapeEmbed_test_positive)
    tapeEmbed_test_negative = np.array(tapeEmbed_test_negative)
    global_test_positive = np.array(global_test_positive)
    global_test_negative = np.array(global_test_negative)

    X_test = np.vstack((x_test_positive, x_test_negative))
    tapeEmbed_test = np.vstack((tapeEmbed_test_positive, tapeEmbed_test_negative))
    Global_test = np.vstack((global_test_positive, global_test_negative))
    Y_test = np.array([1] * len(x_test_positive) + [0] * len(x_test_negative))
    return X_test, Global_test, tapeEmbed_test, Y_test

# get val x,y & the indices of train data
# def get_val_data(Tablefile, name_onehot_file, name_tapeEmbed_file, m, n):
#     # 读入onehot特征的dict
#     dict_onehot = np.load(name_onehot_file, allow_pickle=True).item()
#
#     # 读入tapeEmbed特征的dic
#     dict_tapeEmbed = np.load(name_tapeEmbed_file, allow_pickle=True).item()
#
#
#     # 读入切片总表
#     table_df = pd.read_csv(Tablefile, index_col=0)
#     table = table_df.values.tolist()
#
#     # 按总表切割x_positive, x_negative,global_positive, global_negative  index为左半边残基数， length-1-index为右半边残基数, xing 为切片的填充符号*
#     x_positive = []
#     x_negative = []
#     tapeEmbed_positive = []
#     tapeEmbed_negative = []
#     global_positive = []
#     global_negative = []
#     xing = [0] * 21
#     xing[0] = 1
#     xing = np.array(xing).reshape(1, 21)
#
#     # 遍历总表，根据总表对读入的onehot特征进行切片
#     for i in range(len(table)):
#         # print读入切片的进度
#         # if i % 50000 == 0:
#         #     print('slice {} has done'.format(i))
#         #  切片，分三类：很靠左(左补0)，中间（不补零），很靠右（右补零）
#         name = table[i][0]
#         index = table[i][1]
#         length = table[i][2]
#         # 左边残基个数为index个，右边残基个数为length-1-index个：分类按照，左边残基数和m比较，右边残基数和n比较
#         if index < m and (length - 1 - index) >= n:
#             supply = xing.repeat(m - index, axis=0)
#             onehot_feature = dict_onehot[name]
#             main = onehot_feature[:index + n + 1]
#             slice_fea = np.concatenate((supply, main), axis=0)
#
#         elif index >= m and (length - 1 - index) >= n:
#             onehot_feature = dict_onehot[name]
#             main = onehot_feature[index - m:index + n + 1]
#             slice_fea = main
#
#         elif length - 1 - index < n and index >= m:
#             supply = xing.repeat(n - (length - 1 - index), axis=0)
#             onehot_feature = dict_onehot[name]
#             main = onehot_feature[index - m:]
#             slice_fea = np.concatenate((main, supply), axis=0)
#
#         elif index < m and length - 1 - index < n:
#             supply_left = xing.repeat(m - index, axis=0)
#             supply_right = xing.repeat(n - (length - 1 - index), axis=0)
#             onehot_feature = dict_onehot[name]
#             main = onehot_feature
#             slice_fea = np.concatenate((supply_left, main, supply_right), axis=0)
#
#         if slice_fea.shape != (m+n+1, 21):
#             print(slice_fea.shape)
#             print(table[i])
#
#         # 获取tapeEmbed_feature
#         tapeEmbed_feature = dict_tapeEmbed[name]
#
#         # 获取2000长度的onehot  global特征
#         global_main = dict_onehot[name]
#         if len(global_main) < 2000:
#             supply_left = xing.repeat(2000 - len(global_main), axis=0)
#             global_fea = np.concatenate((supply_left, global_main), axis=0)
#         else:
#             global_fea = global_main[:2000]
#
#         # 对切片好的特征分正负类， 得到总正例切片列表x_positive ,总负例切片列表x_negative
#         if table[i][3] == 1:
#             x_positive.append(slice_fea)
#             tapeEmbed_positive.append(tapeEmbed_feature)
#             global_positive.append(global_fea)
#         else:
#             x_negative.append(slice_fea)
#             tapeEmbed_negative.append(tapeEmbed_feature)
#             global_negative.append(global_fea)
#
#         # sequence_positive = separt_positive(input_sequence, m, n)
#         # sequence_negative = separt_negative(input_sequence_2, m, n)
#     # 计算x_val/2 的个数
#     num_positive = len(x_positive)
#     num_negative = len(x_negative)
#     num_val = int(num_positive/10)
#
#
#     random.seed(0)
#     X_train_positive = np.array(x_positive)
#     X_train_negative = np.array(x_negative)
#     tapeEmbed_train_positive = np.array(tapeEmbed_positive)
#     tapeEmbed_train_negative = np.array(tapeEmbed_negative)
#     global_train_positive = np.array(global_positive)
#     global_train_negative = np.array(global_negative)
#
#     ls = list((range(len(X_train_positive))))
#     random.shuffle(ls)
#
#     X_val_positive = X_train_positive[ls][:num_val]
#     X_train_positive = X_train_positive[ls][num_val:]
#     tapeEmbed_val_positive = tapeEmbed_train_positive[ls][:num_val]
#     tapeEmbed_train_positive = tapeEmbed_train_positive[ls][num_val:]
#     global_val_positive = global_train_positive[ls][:num_val]
#     global_train_positive = global_train_positive[ls][num_val:]
#
#     random.seed(1)
#     ls2 = list((range(len(X_train_negative))))
#     random.shuffle(ls2)
#     X_val_negative = X_train_negative[ls2][:num_val]
#     X_train_negative = X_train_negative[ls2][num_val:]
#     tapeEmbed_val_negative = tapeEmbed_train_negative[ls2][:num_val]
#     tapeEmbed_train_negative = tapeEmbed_train_negative[ls2][num_val:]
#     global_val_negative = global_train_negative[ls2][:num_val]
#     global_train_negative = global_train_negative[ls2][num_val:]
#
#     X_val = np.vstack((X_val_positive, X_val_negative))
#     Y_val = np.array([1] * num_val + [0] * num_val)
#     tapeEmbed_val = np.vstack((tapeEmbed_val_positive, tapeEmbed_val_negative))
#     Global_val = np.vstack((global_val_positive, global_val_negative))
#
#     Y = np.array([1] * (num_positive - num_val) + [0] * (num_negative - num_val))
#     return X_val, Global_val, tapeEmbed_val, Y_val, X_train_positive, global_train_positive, tapeEmbed_train_positive,\
#            X_train_negative, global_train_negative, tapeEmbed_train_negative,  Y


# def get_train_data(idx, X_train_positive, Global_train_positive, X_train_negative, Global_train_negative, Y):
#     # train
#     if len(X_train_positive) * (idx + 1) < len(X_train_negative):
#         X_train = np.vstack(
#             (X_train_positive, X_train_negative[len(X_train_positive) * idx:len(X_train_positive) * (idx + 1)]))
#         Global_train = np.vstack(
#             (Global_train_positive, Global_train_negative[len(Global_train_positive) * idx:len(Global_train_positive) * (idx + 1)])
#         )
#     else:
#         X_train = np.vstack((X_train_positive, X_train_negative[len(X_train_negative) - len(X_train_positive):]))
#         Global_train = np.vstack((Global_train_positive, Global_train_negative[len(Global_train_negative) - len(Global_train_positive):]))
#
#     Y_train = np.array([1] * len(X_train_positive) + [0] * len(X_train_positive))
#     return X_train, Global_train, Y_train

def get_val_data(Tablefile, ls_val_positive, ls_val_negative, name_onehot_file, name_tapeEmbed_file, m, n):
    # 读入onehot特征的dict
    dict_onehot = np.load(name_onehot_file, allow_pickle=True).item()

    # 读入tapeEmbed特征的dic
    dict_tapeEmbed = np.load(name_tapeEmbed_file, allow_pickle=True).item()


    # 读入切片总表
    table_df = pd.read_csv(Tablefile, index_col=0)
    table = table_df.values.tolist()

    # 按总表顺序和读入的两个ls,切割x_positive,global_positive  index为左半边残基数， length-1-index为右半边残基数, xing 为切片的填充符号*
    X_val_positive = []
    X_val_negative = []
    tapeEmbed_val_positive = []
    tapeEmbed_val_negative = []
    Global_val = []
    Global_val_positive = []
    Global_val_negative = []

    xing = [0] * 21
    xing[0] = 1
    xing = np.array(xing).reshape(1, 21)

    # 遍历ls_val_positive，生成各类特征
    for i in ls_val_positive:
        if table[i][3] == 0:
            print('error!')
            break
        #  切片，分三类：很靠左(左补0)，中间（不补零），很靠右（右补零）
        name = table[i][0]
        index = table[i][1]
        length = table[i][2]
        # 左边残基个数为index个，右边残基个数为length-1-index个：分类按照，左边残基数和m比较，右边残基数和n比较
        if index < m and (length - 1 - index) >= n:
            supply = xing.repeat(m - index, axis=0)
            onehot_feature = dict_onehot[name]
            main = onehot_feature[:index + n + 1]
            slice_fea = np.concatenate((supply, main), axis=0)

        elif index >= m and (length - 1 - index) >= n:
            onehot_feature = dict_onehot[name]
            main = onehot_feature[index - m:index + n + 1]
            slice_fea = main

        elif length - 1 - index < n and index >= m:
            supply = xing.repeat(n - (length - 1 - index), axis=0)
            onehot_feature = dict_onehot[name]
            main = onehot_feature[index - m:]
            slice_fea = np.concatenate((main, supply), axis=0)

        elif index < m and length - 1 - index < n:
            supply_left = xing.repeat(m - index, axis=0)
            supply_right = xing.repeat(n - (length - 1 - index), axis=0)
            onehot_feature = dict_onehot[name]
            main = onehot_feature
            slice_fea = np.concatenate((supply_left, main, supply_right), axis=0)

        if slice_fea.shape != (m+n+1, 21):
            print(slice_fea.shape)
            print(table[i])

        # 获取tapeEmbed_feature
        tapeEmbed_feature = dict_tapeEmbed[name]

        # 获取2000长度的onehot  global特征
        global_main = dict_onehot[name]
        if len(global_main) < 2000:
            supply_left = xing.repeat(2000 - len(global_main), axis=0)
            global_fea = np.concatenate((supply_left, global_main), axis=0)
        else:
            global_fea = global_main[:2000]

        # 得到 positive各类特征
        X_val_positive.append(slice_fea)
        tapeEmbed_val_positive.append(tapeEmbed_feature)
        Global_val_positive.append(global_fea)
    print('positive over')


    # 遍历ls_val_negative，生成各类特征
    for i in ls_val_negative:
        if table[i][3] == 1:
            print('error!')
            break
        #  切片，分三类：很靠左(左补0)，中间（不补零），很靠右（右补零）
        name = table[i][0]
        index = table[i][1]
        length = table[i][2]
        # 左边残基个数为index个，右边残基个数为length-1-index个：分类按照，左边残基数和m比较，右边残基数和n比较
        if index < m and (length - 1 - index) >= n:
            supply = xing.repeat(m - index, axis=0)
            onehot_feature = dict_onehot[name]
            main = onehot_feature[:index + n + 1]
            slice_fea = np.concatenate((supply, main), axis=0)

        elif index >= m and (length - 1 - index) >= n:
            onehot_feature = dict_onehot[name]
            main = onehot_feature[index - m:index + n + 1]
            slice_fea = main

        elif length - 1 - index < n and index >= m:
            supply = xing.repeat(n - (length - 1 - index), axis=0)
            onehot_feature = dict_onehot[name]
            main = onehot_feature[index - m:]
            slice_fea = np.concatenate((main, supply), axis=0)

        elif index < m and length - 1 - index < n:
            supply_left = xing.repeat(m - index, axis=0)
            supply_right = xing.repeat(n - (length - 1 - index), axis=0)
            onehot_feature = dict_onehot[name]
            main = onehot_feature
            slice_fea = np.concatenate((supply_left, main, supply_right), axis=0)

        if slice_fea.shape != (m+n+1, 21):
            print(slice_fea.shape)
            print(table[i])

        # 获取tapeEmbed_feature
        tapeEmbed_feature = dict_tapeEmbed[name]

        # 获取2000长度的onehot  global特征
        global_main = dict_onehot[name]
        if len(global_main) < 2000:
            supply_left = xing.repeat(2000 - len(global_main), axis=0)
            global_fea = np.concatenate((supply_left, global_main), axis=0)
        else:
            global_fea = global_main[:2000]

        # 得到 positive各类特征
        X_val_negative.append(slice_fea)
        tapeEmbed_val_negative.append(tapeEmbed_feature)
        Global_val_negative.append(global_fea)
    print('negative over')

    X_val = np.vstack((np.array(X_val_positive), np.array(X_val_negative)))
    Global_val = np.vstack((np.array(Global_val_positive), np.array(Global_val_negative)))
    tapeEmbed_val = np.vstack((np.array(tapeEmbed_val_positive), np.array(tapeEmbed_val_negative)))
    Y_val = np.array([1] * len(X_val_positive) + [0] * len(X_val_positive))

    return X_val, Global_val, tapeEmbed_val, Y_val


def get_subtrainset_data(idx_trainset, num_split_trainset, ls_train_model_positive, ls_train_model_negative,

                                           Tablefile, name_onehot_file, name_tapeEmbed_file, m, n):
    # 读入onehot特征的dict
    dict_onehot = np.load(name_onehot_file, allow_pickle=True).item()

    # 读入tapeEmbed特征的dic
    dict_tapeEmbed = np.load(name_tapeEmbed_file, allow_pickle=True).item()


    # 读入切片总表
    table_df = pd.read_csv(Tablefile, index_col=0)
    table = table_df.values.tolist()

    # 根据 idx_trainset, ls_train_model_positive, ls_train_model_negative，重新计算subtrainset的两个list
    num_sub_positive = int(len(ls_train_model_positive) / num_split_trainset)
    num_sub_negative = int(len(ls_train_model_negative) / num_split_trainset)
    ls_subtrain_positive = ls_train_model_positive[num_sub_positive * (idx_trainset) : num_sub_positive * (idx_trainset + 1)]
    ls_subtrain_negative = ls_train_model_negative[num_sub_negative * (idx_trainset) : num_sub_negative * (idx_trainset + 1)]

    # 按总表顺序和制作的两个ls,切割x_positive,global_positive  index为左半边残基数， length-1-index为右半边残基数, xing 为切片的填充符号*
    X_subtrain_positive = []
    X_subtrain_negative = []
    tapeEmbed_subtrain_positive = []
    tapeEmbed_subtrain_negative = []
    Global_subtrain_positive = []
    Global_subtrain_negative = []

    xing = [0] * 21
    xing[0] = 1
    xing = np.array(xing).reshape(1, 21)

    # 遍历ls_val_positive，生成各类特征
    for i in ls_subtrain_positive:
        if table[i][3] == 0:
            print('error!')
            break
        #  切片，分三类：很靠左(左补0)，中间（不补零），很靠右（右补零）
        name = table[i][0]
        index = table[i][1]
        length = table[i][2]
        # 左边残基个数为index个，右边残基个数为length-1-index个：分类按照，左边残基数和m比较，右边残基数和n比较
        if index < m and (length - 1 - index) >= n:
            supply = xing.repeat(m - index, axis=0)
            onehot_feature = dict_onehot[name]
            main = onehot_feature[:index + n + 1]
            slice_fea = np.concatenate((supply, main), axis=0)

        elif index >= m and (length - 1 - index) >= n:
            onehot_feature = dict_onehot[name]
            main = onehot_feature[index - m:index + n + 1]
            slice_fea = main

        elif length - 1 - index < n and index >= m:
            supply = xing.repeat(n - (length - 1 - index), axis=0)
            onehot_feature = dict_onehot[name]
            main = onehot_feature[index - m:]
            slice_fea = np.concatenate((main, supply), axis=0)

        elif index < m and length - 1 - index < n:
            supply_left = xing.repeat(m - index, axis=0)
            supply_right = xing.repeat(n - (length - 1 - index), axis=0)
            onehot_feature = dict_onehot[name]
            main = onehot_feature
            slice_fea = np.concatenate((supply_left, main, supply_right), axis=0)

        if slice_fea.shape != (m+n+1, 21):
            print(slice_fea.shape)
            print(table[i])

        # 获取tapeEmbed_feature
        tapeEmbed_feature = dict_tapeEmbed[name]

        # 获取2000长度的onehot  global特征
        global_main = dict_onehot[name]
        if len(global_main) < 2000:
            supply_left = xing.repeat(2000 - len(global_main), axis=0)
            global_fea = np.concatenate((supply_left, global_main), axis=0)
        else:
            global_fea = global_main[:2000]

        # 得到 positive各类特征
        X_subtrain_positive.append(slice_fea)
        tapeEmbed_subtrain_positive.append(tapeEmbed_feature)
        Global_subtrain_positive.append(global_fea)
    print('positive over')


    # 遍历ls_val_negative，生成各类特征
    for i in ls_subtrain_negative:
        if table[i][3] == 1:
            print('error!')
            break
        #  切片，分三类：很靠左(左补0)，中间（不补零），很靠右（右补零）
        name = table[i][0]
        index = table[i][1]
        length = table[i][2]
        # 左边残基个数为index个，右边残基个数为length-1-index个：分类按照，左边残基数和m比较，右边残基数和n比较
        if index < m and (length - 1 - index) >= n:
            supply = xing.repeat(m - index, axis=0)
            onehot_feature = dict_onehot[name]
            main = onehot_feature[:index + n + 1]
            slice_fea = np.concatenate((supply, main), axis=0)

        elif index >= m and (length - 1 - index) >= n:
            onehot_feature = dict_onehot[name]
            main = onehot_feature[index - m:index + n + 1]
            slice_fea = main

        elif length - 1 - index < n and index >= m:
            supply = xing.repeat(n - (length - 1 - index), axis=0)
            onehot_feature = dict_onehot[name]
            main = onehot_feature[index - m:]
            slice_fea = np.concatenate((main, supply), axis=0)

        elif index < m and length - 1 - index < n:
            supply_left = xing.repeat(m - index, axis=0)
            supply_right = xing.repeat(n - (length - 1 - index), axis=0)
            onehot_feature = dict_onehot[name]
            main = onehot_feature
            slice_fea = np.concatenate((supply_left, main, supply_right), axis=0)

        if slice_fea.shape != (m+n+1, 21):
            print(slice_fea.shape)
            print(table[i])

        # 获取tapeEmbed_feature
        tapeEmbed_feature = dict_tapeEmbed[name]

        # 获取2000长度的onehot  global特征
        global_main = dict_onehot[name]
        if len(global_main) < 2000:
            supply_left = xing.repeat(2000 - len(global_main), axis=0)
            global_fea = np.concatenate((supply_left, global_main), axis=0)
        else:
            global_fea = global_main[:2000]

        # 得到 positive各类特征
        X_subtrain_negative.append(slice_fea)
        tapeEmbed_subtrain_negative.append(tapeEmbed_feature)
        Global_subtrain_negative.append(global_fea)
    print('negative over')

    X_subtrain = np.vstack((np.array(X_subtrain_positive), np.array(X_subtrain_negative)))
    Global_subtrain = np.vstack((np.array(Global_subtrain_positive), np.array(Global_subtrain_negative)))
    tapeEmbed_subtrain = np.vstack((np.array(tapeEmbed_subtrain_positive), np.array(tapeEmbed_subtrain_negative)))
    Y_subtrain = np.array([1] * num_sub_positive + [0] * num_sub_negative)

    return X_subtrain, Global_subtrain, tapeEmbed_subtrain, Y_subtrain


def get_Matrix_Label_2 (filename, site_type, m, n):
    name, input_sequence = file2str(filename)
    if site_type == 'Y':
        sequence, indexs, site_types = get_test_file_Y(input_sequence, m, n)
    else:
        sequence, indexs, site_types = get_test_file_ST(input_sequence, m, n)
    X_train = []
    for i in range(len(sequence)):
        result_index = str2dic(sequence[i])
        X_train.append(result_index)
    random.seed(0)
    X_test = np.array(X_train)
    X_test = to_categorical(X_test)
    name, input_sequence = file2str(filename)
    return X_test, indexs, name, site_types  # indexs, name, site_types ???


def write_result(modelname,indexs,name,site_types,result_probe,outputfile):
    with open(modelname + outputfile, 'w') as f:
        num = 0
        for k in range(len(indexs)):
            f.write('>')
            f.write(name[k])
            f.write('\n')
            for u in range(len(indexs[k])):
                f.write(str(indexs[k][u]))
                f.write('\t')
                f.write(site_types[k][u])
                f.write('\t')
                f.write(str(result_probe[num,0]))
                f.write('\n')
                num +=1
    print('Successfully predict the phosphorylation site ! prediction results are stored in ' + modelname + outputfile)
