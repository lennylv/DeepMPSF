import os
from sklearn.metrics import roc_curve, auc, roc_auc_score, matthews_corrcoef, recall_score, precision_score, f1_score, \
    accuracy_score, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from Methods import *
import math
from torch.autograd import Variable
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F



# 提前再cuda上存入特征
global ProtT5Embed_dic
ProtT5Embed_dic = np.load('../data/Train_All_general_ST_ProtT5Embed_dict_new.npy', allow_pickle=True).item()

count_train = 0
for proname in ProtT5Embed_dic.keys():
    if count_train <= 5000:
        fea = torch.from_numpy(ProtT5Embed_dic[proname]).cpu().to(torch.device('cuda:3'))
    elif count_train <= 10000 and count_train > 5000:
        fea = torch.from_numpy(ProtT5Embed_dic[proname]).cpu().to(torch.device('cuda:1'))
    else:
        fea = torch.from_numpy(ProtT5Embed_dic[proname]).cpu().to(torch.device('cuda:2'))
    ProtT5Embed_dic[proname] = fea
    count_train += 1


global Test_dic
Test_dic = np.load('../data/Test_All_general_ST_ProtT5Embed_dict_new.npy', allow_pickle=True).item()

count_test = 0
for proname in Test_dic.keys():
    if count_test <= 700:
        fea = torch.from_numpy(Test_dic[proname]).cpu().to(torch.device('cuda:3'))
    elif count_test > 700 and count_test <= 1600:
        fea = torch.from_numpy(Test_dic[proname]).cpu().to(torch.device('cuda:1'))
    else:
        fea = torch.from_numpy(Test_dic[proname]).cpu().to(torch.device('cuda:2'))
    Test_dic[proname] = fea
    count_test += 1


class MyDataset(Dataset):
    def __init__(self, df_table, batchsize, name_seqindex_file, istest=False):
        self.table = df_table.values
        self.batchsize = batchsize
        self.istest = istest
        self.Seqindex_dic = np.load(name_seqindex_file, allow_pickle=True).item()

    def __len__(self):
        return len(self.table)

    def get_local(self, name, residx, prolen, m, n):
        # 获取71长度的 ProtT5特征
        xing = [0] * 1024
        xing = np.array(xing).reshape(1, 1024)

        xing_seq = [0]
        xing_seq = np.array(xing_seq)

        # 左边残基个数为residx个，右边残基个数为prolen-1-residx个：分类按照，左边残基数和m比较，右边残基数和n比较
        if residx < m and (prolen - 1 - residx) >= n:
            if self.istest == True:
                ProtT5_feature = Test_dic[name]
            else:
                ProtT5_feature = ProtT5Embed_dic[name]

            main = ProtT5_feature[:residx + n + 1]
            main_seq = self.Seqindex_dic[name][:residx + n + 1]
            supply = torch.from_numpy(xing.repeat(m - residx, axis=0)).cpu().to(main.device)
            supply_seq = xing_seq.repeat(m - residx, axis=0)
            slice_fea = torch.cat((supply, main), axis=0)
            slice_fea_seq = np.concatenate((supply_seq, main_seq), axis=0)

        elif residx >= m and (prolen - 1 - residx) >= n:
            if self.istest == True:
                ProtT5_feature = Test_dic[name]
            else:
                ProtT5_feature = ProtT5Embed_dic[name]
            main = ProtT5_feature[residx - m:residx + n + 1]
            main_seq = self.Seqindex_dic[name][residx - m:residx + n + 1]
            slice_fea = main
            slice_fea_seq = main_seq

        elif prolen - 1 - residx < n and residx >= m:
            if self.istest == True:
                ProtT5_feature = Test_dic[name]
            else:
                ProtT5_feature = ProtT5Embed_dic[name]
            main = ProtT5_feature[residx - m:]
            main_seq = self.Seqindex_dic[name][residx - m:]
            supply = torch.from_numpy(xing.repeat(n - (prolen - residx - 1), axis=0)).cpu().to(main.device)
            supply_seq = xing_seq.repeat(n - (prolen - residx - 1), axis=0)
            slice_fea = torch.cat((main, supply), axis=0)
            slice_fea_seq = np.concatenate((main_seq, supply_seq), axis=0)

        elif residx < m and prolen - 1 - residx < n:
            if self.istest == True:
                ProtT5_feature = Test_dic[name]
            else:
                ProtT5_feature = ProtT5Embed_dic[name]
            main = ProtT5_feature
            main_seq = self.Seqindex_dic[name]
            supply_left = torch.from_numpy(xing.repeat(m - residx, axis=0)).cpu().to(main.device)
            supply_left_seq = xing_seq.repeat(m - residx, axis=0)
            supply_right = torch.from_numpy(xing.repeat(n - (prolen - residx - 1 ), axis=0)).cpu().to(main.device)
            supply_right_seq = xing_seq.repeat(n - (prolen - residx - 1 ), axis=0)
            slice_fea = torch.cat((supply_left, main, supply_right), axis=0)
            slice_fea_seq = np.concatenate((supply_left_seq, main_seq, supply_right_seq), axis=0)

        return slice_fea.to(torch.device('cuda:2')), torch.from_numpy(slice_fea_seq).to(torch.device('cuda:2'))


    def __getitem__(self, index):
        name = self.table[index][0]
        residx = self.table[index][1]
        prolen = self.table[index][2]
        y = int(self.table[index][3])
        x_local, x_local_seq = self.get_local(name, residx, prolen, m=35, n=35)
        if self.istest == True:
            x_global = Test_dic[name].mean(0).to(torch.device('cuda:2'))
        else:
            x_global = ProtT5Embed_dic[name].mean(0).to(torch.device('cuda:2'))
        return {
            'x_local': x_local,
            'x_local_seq': x_local_seq,
            'x_global': x_global,
            'name': name,
            'residx': residx,
            'y': y}


class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.BCELoss(weight=weight, reduction=reduction)
        self.alpha = 0.75

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = self.alpha * (1 - p) ** self.gamma * logp
        return loss.mean()


class MyEmbed(nn.Module):
    def __init__(self, vocab_size, d_model, slice_len):
        super(MyEmbed, self).__init__()
        '''

        :param vocab_size: 词的种类数量
        :param d_model: 最后embedding的输出dim 
        :param slice_len: slice切片的长度
        '''
        self.aa_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(slice_len, d_model)
        self.norm = nn.LayerNorm(128)

    def forward(self, x):
        slice_len = x.size(1)
        pos = torch.arange(slice_len, device=torch.device('cuda:2'), dtype=torch.long)
        pos = pos.reshape(1, slice_len)
        pos = pos.repeat(x.size(0), 1)

        embedding = self.pos_embed(pos).to(torch.device('cuda:2'))
        embedding = embedding + self.aa_embed(x.to(torch.int64))
        embedding = self.norm(embedding)

        return embedding

class Encoder(nn.Module):
    def __init__(self, input_features, nhead, dim_feedforward):
        super(Encoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(input_features, nhead, dim_feedforward)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 1)

    def forward(self, x):
        return self.encoder(x)


class Attention(nn.Module):
    def __init__(self, in_features, hidden_units, num_task):
        super(Attention, self).__init__()

        self.W1 = nn.Sequential(nn.Linear(in_features=in_features, out_features=hidden_units), nn.Dropout(0.5))
        self.W2 = nn.Sequential(nn.Linear(in_features=in_features, out_features=hidden_units), nn.Dropout(0.5))
        self.V = nn.Linear(in_features=hidden_units, out_features=num_task)

        # self.W1 = nn.Linear(in_features=in_features, out_features=hidden_units)
        # self.W2 = nn.Linear(in_features=in_features, out_features=hidden_units)
        # self.V = nn.Linear(in_features=hidden_units, out_features=num_task)

    def forward(self, hidden_Yates, values):
        # hidden_Yates---->h_n:[b, in_fea]
        # values--->output:[b,len, in_fea]
        hidden_with_time_axis = torch.unsqueeze(hidden_Yates, dim=1)   # hidden_with_time_axis: [b, 1, in_fea]

        score = self.V(nn.Tanh()(self.W1(values) + self.W2(hidden_with_time_axis)))  # [b, len, 1]
        attention_weights = nn.Softmax(dim=1)(score)  # [b, len, 1]
        values = torch.transpose(values, 1,
                                 2)  # transpose to make it suitable for matrix multiplication   [b, input_dim, len]
        # print(attention_weights.shape,values.shape)
        context_vector = torch.matmul(values, attention_weights)  # [b, input_dim, 1]
        # context_vector = torch.transpose(context_vector, 1, 2) # [b, 1, input_dim]
        # return context_vector, attention_weights
        return context_vector


class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()

        self.Embed_local = nn.Sequential(MyEmbed(21, 128, 71), nn.Dropout(0.5))
        self.GRU_front = nn.GRU(128, 256, 1,  batch_first=True)
        self.GRU_after = nn.GRU(128, 256, 1,  batch_first=True)
        self.global_dense = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU())
        self.protT5_local_dense = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU())

        self.dense = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_seq, x_global):
        b, _, _ = x.size()  # [b, 71, 1024]
        token_pos_embed = self.Embed_local(x_seq)
        protT5_local = torch.mean(x, dim=1)

        x_front = token_pos_embed[:, :36, :]
        x_after = token_pos_embed[:, 35:, :]
        x_after = torch.flip(x_after, dims=[1])

        _, h_n_front = self.GRU_front(x_front)
        _, h_n_after = self.GRU_after(x_after)
        dense_part = (h_n_front + h_n_after).view(b, -1)

        protT5_global_out = self.global_dense(x_global)
        protT5_local_out = self.protT5_local_dense(protT5_local)
        out = self.dense(torch.hstack((protT5_global_out, protT5_local_out, dense_part)))

        return out


def My_Model_training(df_train
                           , df_val
                           , modelname
                           , idx
                           , device
                      ):
    model = Mymodel().to(device)
    adam = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam, factor=0.5, mode='max', patience=3, verbose=True,
                                                           min_lr=1e-4)
    criterion = FocalLoss(gamma=2).to(device)
    #epoch y:10, st:10
    epoch = 10
    Batchsize = 512

    train_dataset = MyDataset(df_train, batchsize=Batchsize, name_seqindex_file='../data/Train_All_general_ST_seqindex_fea_dic.npy'
                              )  # train_sampler = SubsetRandomSampler(train_idxs)  # index [3, 7, 840, 1, 2, ..]

    val_dataset = MyDataset(df_val, batchsize=Batchsize, name_seqindex_file='../data/Train_All_general_ST_seqindex_fea_dic.npy'
                            )  # val_sampler = SubsetRandomSampler(val_idx)

    # early stop
    best_acc = -float('Inf')
    # patience = 0
    Errors = 0.001

    for j in range(epoch):
        print('Current Epoch is {}'.format(j))
        print('learning_rate:{}'.format(adam.state_dict()['param_groups'][0]['lr']))
        train_losses = []
        train_accs = []
        # train_loader = DataLoader(train_dataset, batch_size=Batchsize, sampler=train_sampler, num_workers=0, drop_last=True)  # sampler -> random
        train_loader = DataLoader(train_dataset, batch_size=Batchsize, num_workers=0, drop_last=True, pin_memory=False)
        for id, item in enumerate(train_loader):
            if (id + 1) % 100 == 0:
                print('{} batch has done'.format(id))
            # 1. open Dropout
            model.train()  # train mode,

            # 2. prepare data
            x_train, x_train_seq, y_train = item['x_local'], item['x_local_seq'], item['y'].view(-1)  # (256, 51, 21),  (256, 2)  get data from dataloader
            x_train, x_train_seq, y_train = x_train.float(), x_train_seq.float(), y_train.float().to(device)  # convert long data to float
            x_global = item['x_global'].float()


            # print(torch.Tensor(data_token).to(torch.int64))
            # data_token = torch.tensor(data_token).float().to(device)
            # 3. run deep learning model
            y_pred = model(x_train, x_train_seq, x_global)

            # 4. calculate loss
            y_pred = y_pred.reshape([-1])  # reshape to adapt 'y_train'
            loss = criterion(y_pred, y_train)  # prediction, target

            # 5. accumulate loss to calculate the overall loss of the current epoch
            train_losses.append(loss.cpu().data.numpy())  # torch.Tensor -> numpy.ndarray

            # 6. process the output of the model
            y_train, y_pred = y_train.cpu().data.numpy(), y_pred.cpu().data.numpy()  # tensor to numpy

            # 7. metrics
            fpr, tpr, threshold = roc_curve(y_train, y_pred)  # target, prediction
            train_acc = accuracy_score(y_train.round(), y_pred.round(), normalize=True)
            train_accs.append(train_acc)
            # print('AUC: {}'.format(auc(fpr, tpr)))

            # 8. loss back propagation
            adam.zero_grad()
            loss.backward()
            adam.step()
        print('Train Loss: {}'.format(np.mean(np.array(train_losses))))
        print('Train acc: {}'.format(np.mean(np.array(train_accs))))

        val_losses = []
        val_accs = []
        val_AUCS = []
        # val_loader = DataLoader(val_dataset, batch_size=1024, sampler=train_sampler, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=Batchsize, num_workers=0, drop_last=True, pin_memory=False)
        for _, item in enumerate(val_loader):
            # close Dropout
            model.eval()
            x_val, x_val_seq, y_val = item['x_local'], item['x_local_seq'], item['y']
            x_val, x_val_seq, y_val = x_val.float(), x_val_seq.float(),  y_val.float().to(device)
            name, residx = item['name'], item['residx']
            # data_x, data_edge_index, data_len_x, data_site_ls = generate_Data(name, residx, Batchsize, train_graph_index_dic, train_edge_index_dic)
            # data_x = generate_Data(name, residx, Batchsize, train_graph_index_dic)
            x_global = item['x_global'].float()

            # y_pred_val = model(x_val, x_val_seq, data_x)
            y_pred_val = model(x_val, x_val_seq, x_global)
            y_pred_val = y_pred_val.reshape([-1])
            loss = criterion(y_pred_val, y_val)
            val_losses.append(loss.cpu().data.numpy())
            y_val, y_pred_val = y_val.cpu().data.numpy(), y_pred_val.cpu().data.numpy()
            fpr, tpr, threshold = roc_curve(y_val, y_pred_val)
            val_AUCS.append(auc(fpr, tpr))
            val_acc = accuracy_score(y_val.round(), y_pred_val.round(), normalize=True)
            val_accs.append(val_acc)
        current_acc = np.mean(np.array(val_accs))
        print('Validation Loss: {}'.format(np.mean(np.array(val_losses))))
        print('Validation acc: {}'.format(current_acc))
        print('Validation AUC: {}'.format(np.mean(np.array(val_AUCS))))

        scheduler.step(current_acc)

        # early stop & best model saving
        # if current_acc > best_acc:
        #     best_acc = current_acc
        #     patience = 0
        if current_acc - best_acc >= Errors:
            best_acc = current_acc

            # saving best model
            sp = os.path.join(
                '../',
                modelname)
            if not os.path.exists(sp):
                os.mkdir(sp)
            torch.save(model, os.path.join(sp, 'weight' + str(idx) + '.pkl'))
            print('Best Model saving-----')


def My_Model_testing(df_test, modelname, df_val, device):
    sp = os.path.join(
        '../model',
        modelname)
    fns = os.listdir(sp)  # parameter files
    Batchsize = 512

    X_pred_test = np.zeros((len(df_test), len(fns)))
    Y_test = np.zeros((1, len(df_test)))
    test_dataset = MyDataset(df_test, batchsize=Batchsize, istest=True, name_seqindex_file='../data/Test_All_general_ST_seqindex_fea_dic.npy'
                             )

    test_loader = DataLoader(test_dataset, batch_size=Batchsize, num_workers=0, drop_last=True, pin_memory=False)
    for f_idx in range(len(fns)):
        print("Test %d ." % f_idx)
        model = torch.load(os.path.join(sp, 'weight' + str(f_idx) + '.pkl'), map_location=device)
        for idx, item in enumerate(test_loader):
            model.eval()
            x_test, x_test_seq, y_test = item['x_local'], item['x_local_seq'], item['y']
            x_test, x_test_seq, y_test = x_test.float(), x_test_seq.float(), y_test.to(device)
            x_global = item['x_global'].float()

            y_test = y_test.cpu().data.numpy().reshape([-1])
            y_pred_test = model(x_test, x_test_seq, x_global)
            y_pred_test = y_pred_test.cpu().data.numpy()
            y_pred_test = y_pred_test.reshape([-1])
            if len(df_test) // Batchsize == idx:
                X_pred_test[idx * Batchsize:, f_idx] = y_pred_test
                Y_test[:, idx * Batchsize:] = y_test
            else:
                X_pred_test[idx * Batchsize: (idx + 1) * Batchsize, f_idx] = y_pred_test
                Y_test[:, idx * Batchsize: (idx + 1) * Batchsize] = y_test

    Y_test = np.array(Y_test).reshape([-1])

    X_pred_val = np.zeros((len(df_val), len(fns)))
    Y_val = np.zeros((1, len(df_val)))
    val_dataset = MyDataset(df_val, batchsize=Batchsize, name_seqindex_file='../data/Train_All_general_ST_seqindex_fea_dic.npy'
                            )
    val_loader = DataLoader(val_dataset, batch_size=Batchsize, num_workers=0, drop_last=True, pin_memory=False)
    for f_idx in range(len(fns)):
        print("Validation %d ." % f_idx)
        model = torch.load(os.path.join(sp, 'weight' + str(f_idx) + '.pkl'), map_location=device)
        for idx, item in enumerate(val_loader):
            model.eval()
            x_val, x_val_seq, y_val = item['x_local'], item['x_local_seq'], item['y']
            x_val, x_val_seq, y_val = x_val.float(), x_val_seq.float(), y_val.to(device)
            x_global = item['x_global'].float()

            y_val = y_val.cpu().data.numpy().reshape([-1])
            y_pred_val = model(x_val, x_val_seq, x_global)
            y_pred_val = y_pred_val.cpu().data.numpy()
            y_pred_val = y_pred_val.reshape([-1])
            if len(df_val) // Batchsize == idx:
                X_pred_val[idx * Batchsize:, f_idx] = y_pred_val
                Y_val[:, idx * Batchsize:] = y_val
            else:
                X_pred_val[idx * Batchsize: (idx + 1) * Batchsize, f_idx] = y_pred_val
                Y_val[:, idx * Batchsize: (idx + 1) * Batchsize] = y_val

    lr = LogisticRegression(C=0.1)
    Y_val = np.array(Y_val).reshape([-1])

    lr.fit(X_pred_val, Y_val)
    pred = lr.predict(X_pred_test)  # return
    pred_probe = lr.predict_proba(X_pred_test)  # return

    # probe_mean = np.mean(pred_probe, axis=1)
    fpr, tpr, threshold = roc_curve(Y_test, pred_probe[:, 1])

    # caculate prc

    precision, recall, thresholds = precision_recall_curve(Y_test, pred_probe[:, 1])

    # import pdb; pdb.set_trace()
    auc_score = auc(fpr, tpr)

    # prc_score
    prc_score = average_precision_score(Y_test, pred_probe[:, 1], average='macro', pos_label=1, sample_weight=None)

    # AUC
    print('AUC : %.4f' % auc_score)

    # PRC
    print('PRC : %.4f' % prc_score)

    # Pre_score
    Pre_score = precision_score(Y_test, pred_probe[:, 1].round())
    print('Pre_score : %.4f' % Pre_score)

    # Recall_score
    Re_score = recall_score(Y_test, pred_probe[:, 1].round())
    print('Re_score : %.4f' % Re_score)

    # Mcc_score
    MCC_value = matthews_corrcoef(Y_test, pred_probe[:, 1].round())
    print('MCC_value : %.4f' % MCC_value)

    # F1_score
    F1_score = f1_score(Y_test, pred_probe[:, 1].round())
    print('f1_score : %.4f' % F1_score)


def My_Model_Blindtesting(df_test, modelname, df_val, device):

    global Test_dic
    Test_dic = np.load('../data/blind_human_ST_ProtT5Embed_dict_new.npy', allow_pickle=True).item()

    count_test = 0
    for proname in Test_dic.keys():
        if count_test <= 700:
            fea = torch.from_numpy(Test_dic[proname]).cpu()
        elif count_test > 700 and count_test <= 1600:
            fea = torch.from_numpy(Test_dic[proname]).cpu()
        else:
            fea = torch.from_numpy(Test_dic[proname]).cpu()
        Test_dic[proname] = fea
        count_test += 1


    sp = os.path.join(
        '../model',
        modelname)
    fns = os.listdir(sp)  # parameter files
    Batchsize = 512

    X_pred_test = np.zeros((len(df_test), len(fns)))
    Y_test = np.zeros((1, len(df_test)))
    test_dataset = MyDataset(df_test, batchsize=Batchsize, istest=True, name_seqindex_file='../data/blind_human_ST_seqindex_fea_dic.npy')

    test_loader = DataLoader(test_dataset, batch_size=Batchsize, num_workers=0, drop_last=True, pin_memory=False)
    for f_idx in range(len(fns)):
        print("Test %d ." % f_idx)
        model = torch.load(os.path.join(sp, 'weight' + str(f_idx) + '.pkl'), map_location=device)
        for idx, item in enumerate(test_loader):
            model.eval()
            x_test, x_test_seq, y_test = item['x_local'], item['x_local_seq'], item['y']
            x_test, x_test_seq, y_test = x_test.float(), x_test_seq.float(), y_test.to(device)
            x_global = item['x_global'].float()

            y_test = y_test.cpu().data.numpy().reshape([-1])
            # y_pred_test = model(x_test, x_test_seq, data_x)
            y_pred_test = model(x_test, x_test_seq, x_global)
            y_pred_test = y_pred_test.cpu().data.numpy()
            y_pred_test = y_pred_test.reshape([-1])
            if len(df_test) // Batchsize == f_idx:
                X_pred_test[idx * Batchsize:, f_idx] = y_pred_test
                Y_test[:, idx * Batchsize:] = y_test
            else:
                X_pred_test[idx * Batchsize: (idx + 1) * Batchsize, f_idx] = y_pred_test
                Y_test[:, idx * Batchsize: (idx + 1) * Batchsize] = y_test

    Y_test = np.array(Y_test).reshape([-1])

    X_pred_val = np.zeros((len(df_val), len(fns)))
    Y_val = np.zeros((1, len(df_val)))
    val_dataset = MyDataset(df_val, batchsize=Batchsize, name_seqindex_file='../data/Train_All_general_ST_seqindex_fea_dic.npy'
                            )
    val_loader = DataLoader(val_dataset, batch_size=Batchsize, num_workers=0, drop_last=True, pin_memory=False)
    for f_idx in range(len(fns)):
        print("Validation %d ." % f_idx)
        model = torch.load(os.path.join(sp, 'weight' + str(f_idx) + '.pkl'), map_location=device)
        for idx, item in enumerate(val_loader):
            model.eval()
            x_val, x_val_seq, y_val = item['x_local'], item['x_local_seq'], item['y']
            x_val, x_val_seq, y_val = x_val.float(), x_val_seq.float(), y_val.to(device)
            x_global = item['x_global'].float()

            y_val = y_val.cpu().data.numpy().reshape([-1])
            y_pred_val = model(x_val, x_val_seq, x_global)
            y_pred_val = y_pred_val.cpu().data.numpy()
            y_pred_val = y_pred_val.reshape([-1])
            if len(df_val) // Batchsize == f_idx:
                X_pred_val[idx * Batchsize:, f_idx] = y_pred_val
                Y_val[:, idx * Batchsize:] = y_val
            else:
                X_pred_val[idx * Batchsize: (idx + 1) * Batchsize, f_idx] = y_pred_val
                Y_val[:, idx * Batchsize: (idx + 1) * Batchsize] = y_val

    lr = LogisticRegression(C=0.1)
    Y_val = np.array(Y_val).reshape([-1])

    lr.fit(X_pred_val, Y_val)
    pred = lr.predict(X_pred_test)  # return
    pred_probe = lr.predict_proba(X_pred_test)  # return

    # probe_mean = np.mean(pred_probe, axis=1)
    fpr, tpr, threshold = roc_curve(Y_test, pred_probe[:, 1])

    # caculate prc

    precision, recall, thresholds = precision_recall_curve(Y_test, pred_probe[:, 1])

    # import pdb; pdb.set_trace()
    auc_score = auc(fpr, tpr)

    # prc_score
    prc_score = average_precision_score(Y_test, pred_probe[:, 1], average='macro', pos_label=1, sample_weight=None)

    # AUC
    print('AUC : %.4f' % auc_score)

    # PRC
    print('PRC : %.4f' % prc_score)

    # Pre_score
    Pre_score = precision_score(Y_test, pred_probe[:, 1].round())
    print('Pre_score : %.4f' % Pre_score)

    # Recall_score
    Re_score = recall_score(Y_test, pred_probe[:, 1].round())
    print('Re_score : %.4f' % Re_score)

    # Mcc_score
    MCC_value = matthews_corrcoef(Y_test, pred_probe[:, 1].round())
    print('MCC_value : %.4f' % MCC_value)

    # F1_score
    F1_score = f1_score(Y_test, pred_probe[:, 1].round())
    print('f1_score : %.4f' % F1_score)



def My_Model_Blindtesting_musculus(df_test, modelname, df_val, device):

    global Test_dic
    Test_dic = np.load('../data/blind_musculus_ST_ProtT5Embed_dict_new.npy', allow_pickle=True).item()

    count_test = 0
    for proname in Test_dic.keys():
        if count_test <= 700:
            fea = torch.from_numpy(Test_dic[proname]).cpu()
        elif count_test > 700 and count_test <= 1600:
            fea = torch.from_numpy(Test_dic[proname]).cpu()
        else:
            fea = torch.from_numpy(Test_dic[proname]).cpu()
        Test_dic[proname] = fea
        count_test += 1


    sp = os.path.join(
        '../model',
        modelname)
    fns = os.listdir(sp)  # parameter files
    Batchsize = 512

    X_pred_test = np.zeros((len(df_test), len(fns)))
    Y_test = np.zeros((1, len(df_test)))
    test_dataset = MyDataset(df_test, batchsize=Batchsize, istest=True, name_seqindex_file='../data/blind_musculus_ST_seqindex_fea_dic.npy')
    test_loader = DataLoader(test_dataset, batch_size=Batchsize, num_workers=0, drop_last=True, pin_memory=False)
    for f_idx in range(len(fns)):
        print("Test %d ." % f_idx)
        model = torch.load(os.path.join(sp, 'weight' + str(f_idx) + '.pkl'), map_location=device)
        for idx, item in enumerate(test_loader):
            model.eval()
            x_test, x_test_seq, y_test = item['x_local'], item['x_local_seq'], item['y']
            x_test, x_test_seq, y_test = x_test.float(), x_test_seq.float(), y_test.to(device)
            x_global = item['x_global'].float()

            y_test = y_test.cpu().data.numpy().reshape([-1])
            # y_pred_test = model(x_test, x_test_seq, data_x)
            y_pred_test = model(x_test, x_test_seq, x_global)
            y_pred_test = y_pred_test.cpu().data.numpy()
            y_pred_test = y_pred_test.reshape([-1])
            if len(df_test) // Batchsize ==  idx:
                X_pred_test[idx * Batchsize:, f_idx] = y_pred_test
                Y_test[:, idx * Batchsize:] = y_test
            else:
                X_pred_test[idx * Batchsize: (idx + 1) * Batchsize, f_idx] = y_pred_test
                Y_test[:, idx * Batchsize: (idx + 1) * Batchsize] = y_test

    Y_test = np.array(Y_test).reshape([-1])

    X_pred_val = np.zeros((len(df_val), len(fns)))
    Y_val = np.zeros((1, len(df_val)))
    # val_idx = [*range(X_val.shape[0])]
    val_dataset = MyDataset(df_val, batchsize=Batchsize, name_seqindex_file='../data/Train_All_general_ST_seqindex_fea_dic.npy')
    # val_sampler = SubsetRandomSampler(val_idx)
    val_loader = DataLoader(val_dataset, batch_size=Batchsize, num_workers=0, drop_last=True, pin_memory=False)
    for f_idx in range(len(fns)):
        print("Validation %d ." % f_idx)
        model = torch.load(os.path.join(sp, 'weight' + str(f_idx) + '.pkl'), map_location=device)
        for idx, item in enumerate(val_loader):
            model.eval()
            x_val, x_val_seq, y_val = item['x_local'], item['x_local_seq'], item['y']
            x_val, x_val_seq, y_val = x_val.float(), x_val_seq.float(), y_val.to(device)
            x_global = item['x_global'].float()
            y_val = y_val.cpu().data.numpy().reshape([-1])
            y_pred_val = model(x_val, x_val_seq, x_global)
            y_pred_val = y_pred_val.cpu().data.numpy()
            y_pred_val = y_pred_val.reshape([-1])
            if len(df_val) // Batchsize == idx:
                X_pred_val[idx * Batchsize:, f_idx] = y_pred_val
                Y_val[:, idx * Batchsize:] = y_val
            else:
                X_pred_val[idx * Batchsize: (idx + 1) * Batchsize, f_idx] = y_pred_val
                Y_val[:, idx * Batchsize: (idx + 1) * Batchsize] = y_val

    lr = LogisticRegression(C=0.1)
    Y_val = np.array(Y_val).reshape([-1])

    lr.fit(X_pred_val, Y_val)
    pred = lr.predict(X_pred_test)  # return
    pred_probe = lr.predict_proba(X_pred_test)  # return

    # probe_mean = np.mean(pred_probe, axis=1)
    fpr, tpr, threshold = roc_curve(Y_test, pred_probe[:, 1])

    # caculate prc

    precision, recall, thresholds = precision_recall_curve(Y_test, pred_probe[:, 1])

    # import pdb; pdb.set_trace()
    auc_score = auc(fpr, tpr)

    # prc_score
    prc_score = average_precision_score(Y_test, pred_probe[:, 1], average='macro', pos_label=1, sample_weight=None)

    # AUC
    print('AUC : %.4f' % auc_score)

    # PRC
    print('PRC : %.4f' % prc_score)

    # Pre_score
    Pre_score = precision_score(Y_test, pred_probe[:, 1].round())
    print('Pre_score : %.4f' % Pre_score)

    # Recall_score
    Re_score = recall_score(Y_test, pred_probe[:, 1].round())
    print('Re_score : %.4f' % Re_score)

    # Mcc_score
    MCC_value = matthews_corrcoef(Y_test, pred_probe[:, 1].round())
    print('MCC_value : %.4f' % MCC_value)

    # F1_score
    F1_score = f1_score(Y_test, pred_probe[:, 1].round())
    print('f1_score : %.4f' % F1_score)


def My_Model_Blindtesting_Rattus(df_test, modelname, df_val, device):

    global Test_dic
    Test_dic = np.load('../data/blind_Rattus_ST_ProtT5Embed_dict_new.npy', allow_pickle=True).item()

    count_test = 0
    for proname in Test_dic.keys():
        if count_test <= 700:
            fea = torch.from_numpy(Test_dic[proname]).cpu()
        elif count_test > 700 and count_test <= 1600:
            fea = torch.from_numpy(Test_dic[proname]).cpu()
        else:
            fea = torch.from_numpy(Test_dic[proname]).cpu()
        Test_dic[proname] = fea
        count_test += 1


    sp = os.path.join(
        '../model',
        modelname)
    fns = os.listdir(sp)  # parameter files
    Batchsize = 512

    X_pred_test = np.zeros((len(df_test), len(fns)))
    Y_test = np.zeros((1, len(df_test)))
    test_dataset = MyDataset(df_test, batchsize=Batchsize, istest=True, name_seqindex_file='../data/blind_Rattus_ST_seqindex_fea_dic.npy')
    test_loader = DataLoader(test_dataset, batch_size=Batchsize, num_workers=0, drop_last=True, pin_memory=False)
    for f_idx in range(len(fns)):
        print("Test %d ." % f_idx)
        model = torch.load(os.path.join(sp, 'weight' + str(f_idx) + '.pkl'), map_location=device)
        for idx, item in enumerate(test_loader):
            model.eval()
            x_test, x_test_seq, y_test = item['x_local'], item['x_local_seq'], item['y']
            x_test, x_test_seq, y_test = x_test.float(), x_test_seq.float(), y_test.to(device)
            x_global = item['x_global'].float()

            y_test = y_test.cpu().data.numpy().reshape([-1])
            y_pred_test = model(x_test, x_test_seq, x_global)
            y_pred_test = y_pred_test.cpu().data.numpy()
            y_pred_test = y_pred_test.reshape([-1])
            if len(df_test) // Batchsize == idx:
                X_pred_test[idx * Batchsize:, f_idx] = y_pred_test
                Y_test[:, idx * Batchsize:] = y_test
            else:
                X_pred_test[idx * Batchsize: (idx + 1) * Batchsize, f_idx] = y_pred_test
                Y_test[:, idx * Batchsize: (idx + 1) * Batchsize] = y_test

    Y_test = np.array(Y_test).reshape([-1])

    X_pred_val = np.zeros((len(df_val), len(fns)))
    Y_val = np.zeros((1, len(df_val)))
    # val_idx = [*range(X_val.shape[0])]
    val_dataset = MyDataset(df_val, batchsize=Batchsize, name_seqindex_file='../data/Train_All_general_ST_seqindex_fea_dic.npy'
                            )
    # val_sampler = SubsetRandomSampler(val_idx)
    val_loader = DataLoader(val_dataset, batch_size=Batchsize, num_workers=0, drop_last=True, pin_memory=False)
    for f_idx in range(len(fns)):
        print("Validation %d ." % f_idx)
        model = torch.load(os.path.join(sp, 'weight' + str(f_idx) + '.pkl'), map_location=device)
        for idx, item in enumerate(val_loader):
            model.eval()
            x_val, x_val_seq, y_val = item['x_local'], item['x_local_seq'], item['y']
            x_val, x_val_seq, y_val = x_val.float(), x_val_seq.float(), y_val.to(device)
            x_global = item['x_global'].float()

            y_val = y_val.cpu().data.numpy().reshape([-1])
            y_pred_val = model(x_val, x_val_seq, x_global)
            y_pred_val = y_pred_val.cpu().data.numpy()
            y_pred_val = y_pred_val.reshape([-1])
            if len(df_val) // Batchsize == idx:
                X_pred_val[idx * Batchsize:, f_idx] = y_pred_val
                Y_val[:, idx * Batchsize:] = y_val
            else:
                X_pred_val[idx * Batchsize: (idx + 1) * Batchsize, f_idx] = y_pred_val
                Y_val[:, idx * Batchsize: (idx + 1) * Batchsize] = y_val

    lr = LogisticRegression(C=0.1)
    Y_val = np.array(Y_val).reshape([-1])

    lr.fit(X_pred_val, Y_val)
    pred = lr.predict(X_pred_test)  # return
    pred_probe = lr.predict_proba(X_pred_test)  # return

    # probe_mean = np.mean(pred_probe, axis=1)
    fpr, tpr, threshold = roc_curve(Y_test, pred_probe[:, 1])

    # caculate prc

    precision, recall, thresholds = precision_recall_curve(Y_test, pred_probe[:, 1])

    # prc_score
    prc_score = average_precision_score(Y_test, pred_probe[:, 1], average='macro', pos_label=1, sample_weight=None)

    # AUC
    print('AUC : %.4f' % auc_score)

    # PRC
    print('PRC : %.4f' % prc_score)

    # Pre_score
    Pre_score = precision_score(Y_test, pred_probe[:, 1].round())
    print('Pre_score : %.4f' % Pre_score)

    # Recall_score
    Re_score = recall_score(Y_test, pred_probe[:, 1].round())
    print('Re_score : %.4f' % Re_score)

    # Mcc_score
    MCC_value = matthews_corrcoef(Y_test, pred_probe[:, 1].round())
    print('MCC_value : %.4f' % MCC_value)

    # F1_score
    F1_score = f1_score(Y_test, pred_probe[:, 1].round())
    print('f1_score : %.4f' % F1_score)

