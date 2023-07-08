#!/usr/bin/python
# -- coding:utf8 --


import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from sklearn import svm


import sys
import os
import numpy as np

np.set_printoptions(threshold=np.inf)
import pdb
from torchvision import transforms
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from einops import rearrange
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score,average_precision_score
import random
import gzip
import pickle
import timeit
import argparse

import matplotlib.pyplot as plt
import torch.utils.data as Data

from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor



# Is it possible to use the GPU ?
if torch.cuda.is_available():
    cuda = True
    print('===> Using GPU')
    #torch.cuda.manual_seed_all(seed)
else:
    cuda = False
    print('===> Using CPU')
    #torch.cuda.manual_seed_all(seed)


# Pad the RNA sequence to window_size with 'N'
def padding_sequence_new(seq, window_size=101, repkey='N'):
    seq_len = len(seq)
    new_seq = seq
    if seq_len < window_size:
        gap_len = window_size - seq_len
        new_seq = seq + repkey * gap_len
    return new_seq


# If the length of the RNA sequence is less than max_len,
# use 'N' to pad to the length of max_len.
# If the length of the RNA sequence is greater than max_len,
# only the RNA sequence with the length of max_len is retained from front to back.
def padding_sequence(seq, max_len=501, repkey='N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len - seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq


# The length of the RNA sequence is L,
# and the RNA sequence is turned into a one-hot encoding matrix with (L+6) rows and 4 columns.
# The first 3 rows and the last 3 rows are filled with [0.25, 0.25, 0.25, 0.25] represented by N
def get_RNA_seq_concolutional_array(seq, motif_len=4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    row = (len(seq) + 2 * motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len - 1):
        new_array[i] = np.array([0.25] * 4)

    for i in range(row - 3, row):
        new_array[i] = np.array([0.25] * 4)

    for i, val in enumerate(seq):
        i = i + motif_len - 1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25] * 4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
    return new_array


# Process RNA sequences into partially overlapping subsequences
def split_overlap_seq(seq, window_size):
    overlap_size = 50
    bag_seqs = []
    seq_len = len(seq)
    if seq_len >= window_size:
        num_ins = (seq_len - window_size) / (window_size - overlap_size) + 1
        remain_ins = (seq_len - window_size) % (window_size - overlap_size)
    else:
        num_ins = 0
    bag = []
    end = 0
    for ind in range(int(num_ins)):
        start = end - overlap_size
        if start < 0:
            start = 0
        end = start + window_size
        subseq = seq[start:end]
        bag_seqs.append(subseq)
    if num_ins == 0:
        seq1 = seq
        pad_seq = padding_sequence_new(seq1, window_size)
        bag_seqs.append(pad_seq)
    else:
        if remain_ins > 10:
            seq1 = seq[-window_size:]
            pad_seq = padding_sequence_new(seq1, window_size)
            bag_seqs.append(pad_seq)
    return bag_seqs


# Read fa file line by line, and generate labels
def read_seq_graphprot(seq_file, label=1):
    seq_list = []
    labels = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
            else:
                seq = line[:-1].upper()
                seq = seq.replace('T', 'U')
                seq_list.append(seq)
                labels.append(label)

    return seq_list, labels


# When processing RNA sequences into multiple partially overlapping subsequences,
# generate a one-hot encoding matrix with multiple channels and corresponding labels
def get_bag_data(data, channel=7, window_size=101):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        # pdb.set_trace()
        bag_seqs = split_overlap_seq(seq, window_size=window_size)
        # flat_array = []
        bag_subt = []
        for bag_seq in bag_seqs:
            tri_fea = get_RNA_seq_concolutional_array(bag_seq)
            bag_subt.append(tri_fea.T)
        num_of_ins = len(bag_subt)
        if num_of_ins > channel:
            start = (num_of_ins - channel) / 2
            bag_subt = bag_subt[start: start + channel]
        if len(bag_subt) < channel:
            rand_more = channel - len(bag_subt)
            for ind in range(rand_more):
                # bag_subt.append(random.choice(bag_subt))
                tri_fea = get_RNA_seq_concolutional_array('N' * window_size)
                bag_subt.append(tri_fea.T)
        bags.append(np.array(bag_subt))

    return bags, labels


# When using 'N' to pad the RNA sequence to a certain length,
# generate a one-hot encoding matrix with one channel and corresponding labels
def get_bag_data_1_channel(data, max_len=501):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        # pdb.set_trace()
        # bag_seqs = split_overlap_seq(seq)
        bag_seq = padding_sequence(seq, max_len=max_len)
        # flat_array = []
        bag_subt = []
        # for bag_seq in bag_seqs:
        tri_fea = get_RNA_seq_concolutional_array(bag_seq)
        # print tri_fea
        bag_subt.append(tri_fea.T)
        # print tri_fea.T

        bags.append(np.array(bag_subt))
        # print bags

    return bags, labels






# base deep learning model frame
class Estimator(object):

    def __init__(self, model):
        self.model = model

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_f = loss

    def _fit(self, train_loader):
        """
        train one epoch
        """
        loss_list = []
        acc_list = []
        for idx, (X, y) in enumerate(train_loader):
            # for X, y in zip(X_train, y_train):
            # X_v = Variable(torch.from_numpy(X.astype(np.float32)))
            # y_v = Variable(torch.from_numpy(np.array(ys)).long())
            X_v = Variable(X)
            y_v = Variable(y)
            if cuda:
                X_v = X_v.cuda()
                y_v = y_v.cuda()

            self.optimizer.zero_grad()
            y_pred = self.model(X_v)
            loss = self.loss_f(y_pred, y_v)
            loss.backward()
            self.optimizer.step()

            ## for log
            loss_list.append(loss.item())  # need change to loss_list.append(loss.item()) for pytorch v0.4 or above

        return sum(loss_list) / len(loss_list)

    def fit(self, X, y, batch_size=32, nb_epoch=10, validation_data=()):
        # X_list = batch(X, batch_size)
        # y_list = batch(y, batch_size)
        # pdb.set_trace()
        print(X.shape)
        train_set = TensorDataset(torch.from_numpy(X.astype(np.float32)),
                                  torch.from_numpy(y.astype(np.float32)).long().view(-1))
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        self.model.train()
        for t in range(nb_epoch):
            loss = self._fit(train_loader)
            # print ('%.5f'%loss)
            # rint("Epoch %s/%s loss: %06.4f - acc: %06.4f %s" % (t, nb_epoch, loss, acc, val_log))

    def evaluate(self, X, y, batch_size=32):

        y_pred = self.predict(X)

        y_v = Variable(torch.from_numpy(y).long(), requires_grad=False)
        if cuda:
            y_v = y_v.cuda()
        loss = self.loss_f(y_pred, y_v)
        predict = y_pred.data.cpu().numpy()[:, 1].flatten()
        auc = roc_auc_score(y, predict)
        # lasses = torch.topk(y_pred, 1)[1].data.numpy().flatten()
        # cc = self._accuracy(classes, y)
        return loss.data[0], auc

    def _accuracy(self, y_pred, y):
        return float(sum(y_pred == y)) / y.shape[0]

    def predict(self, X):
        X = Variable(torch.from_numpy(X.astype(np.float32)))
        if cuda:
            X = X.cuda()
        y_pred = self.model(X)
        return y_pred

    def predict_proba(self, X):
        self.model.eval()
        return self.model.predict_proba(X)


class RSBU_CW(nn.Module): 
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.shrinkage = Shrinkage(out_channels, gap_size=(1, 1))
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels *RSBU_CW.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * RSBU_CW.expansion),
            self.shrinkage
        )
        # shortcut
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels !=RSBU_CW.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * RSBU_CW.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * RSBU_CW.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        average = x
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2).unsqueeze(2)
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x


class ECAlayer(nn.Module):
    def __init__(self, channel, gamma=2, bias=1):
        super(ECAlayer, self).__init__()
        # x: input features with shape [b, c, h, w]
        self.channel = channel
        self.gamma = gamma
        self.bias = bias

        k_size = int(
            abs((math.log(self.channel, 2) + self.bias) / self.gamma))  
        k_size = k_size if k_size % 2 else k_size + 1  

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x) 
        # b,c,1,1
        y = self.conv(y.squeeze(-1).transpose(-1, -2))    
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
    
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out 
        return self.sigmoid(out)  

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
       
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  
        max_out, _ = torch.max(x, dim=1, keepdim=True)  
        x = torch.cat([avg_out, max_out], dim=1) 
        x = self.conv1(x) 
        return self.sigmoid(x) 


class cbamblock(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(cbamblock, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)  
        x = x * self.spatialattention(x)  
        return x


class DRSN(nn.Module):
    def __init__(self, nb_filter, channel=7, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3), labcounts=32,
                 window_size=12, hidden_size=256, stride=(1, 1), padding=0):
        super(DRSN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, stride=stride))
        out1_size = int((window_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1] + 1)
        maxpool_size = int((out1_size + 2 * padding - (pool_size[1] - 1) - 1) / stride[1] + 1)
        self.reslayer1 = self._make_layer(RSBU_CW, 16, 1, 1)
        self.reslayer11 = cbamblock(16)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 10), stride=stride, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, stride=stride))
        out2_size = int((maxpool_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1] + 1)
        maxpool2_size = int((out2_size + 2 * padding - (pool_size[1] - 1) - 1) / stride[1] + 1)
        self.reslayer2 =self._make_layer(RSBU_CW, 32, 1, 1)
        self.reslayer20 = ECAlayer(32)
        self.reslayer21 = cbamblock(32)


        self.drop1 = nn.Dropout(p=0.25)
        print('maxpool_size', maxpool_size)
        self.fc1 = nn.Linear(maxpool2_size * 32, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        

    def _make_layer(self, RSBU_CW, out_channels, num_blocks, stride):
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(RSBU_CW(out_channels, out_channels, stride))
            self.in_channels = out_channels *  RSBU_CW.expansion
        return nn.Sequential(*layers)

        
        
        

    def forward(self, x):
        out = self.layer1(x)
        out = self.reslayer1(out)
        out = self.layer2(out)
        out = self.reslayer2(out)
        out = self.reslayer20(out)
        out = self.reslayer21(out)
        out = out.view(out.size(0), -1)
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
   
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        # x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.layer1(x)
        temp = out.data.cpu().numpy()
        return temp

    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        # x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]


 
 
# Integrate positive and negative sample files
def read_data_file(posifile, negafile=None, train=True):
    data = dict()
    seqs, labels = read_seq_graphprot(posifile, label=1)
    if negafile:
        seqs2, labels2 = read_seq_graphprot(negafile, label=0)
        seqs = seqs + seqs2
        labels = labels + labels2
        # print(labels)

    data["seq"] = seqs
    data["Y"] = np.array(labels)

    return data


# returns the one-hot encoded matrix and labels of RNA sequences
def get_data(posi, nega=None, channel=7, window_size=101, train=True):
    data = read_data_file(posi, nega, train=train)
    if channel == 1:
        train_bags, label = get_bag_data_1_channel(data, max_len=window_size)

    else:
        train_bags, label = get_bag_data(data, channel=channel, window_size=window_size)

    return train_bags, label


# train and save a deep learning model
def train_network(model_type, X_train, y_train, channel=7, window_size=107, model_file='model.pkl', batch_size=128,
                  n_epochs=50, num_filters=16):
    print('model training for ', model_type)
    # nb_epos= 5
    if model_type == 'DRSN':
        model = DRSN(nb_filter=num_filters, labcounts=4, window_size=window_size, channel=channel)
    else:
        print('only support DRSN model')

    if cuda:
        model = model.cuda()

    clf = Estimator(model)

    clf.compile(optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001),
                loss=nn.CrossEntropyLoss())
    clf.fit(X_train, y_train, batch_size=batch_size, nb_epoch=n_epochs)
    torch.save(model.state_dict(), model_file)
    # print 'predicting'
    # pred = model.predict_proba(test_bags)
    # return model


# test the trained deep learning mode
def predict_network(model_type, X_test, channel=7, window_size=107, model_file='model.pkl', batch_size=128, n_epochs=50,
                    num_filters=16):
    print('model training for ', model_type)
    # nb_epos= 5
    if model_type == 'DRSN':
        model = DRSN(nb_filter=num_filters, labcounts=4, window_size=window_size, channel=channel)

    else:
        print('only support DRSN model')

    if cuda:
        model = model.cuda()

    model.load_state_dict(torch.load(model_file))
    try:
        pred = model.predict_proba(X_test)
    except:  # to handle the out-of-memory when testing
        test_batch = batch(X_test)
        pred = []
        for test in test_batch:
            pred_test1 = model.predict_proba(test)[:, 1]
            pred = np.concatenate((pred, pred_test1), axis=0)
    return pred

def Indicators(y, y_pre):
    '''
    :param y: array，True value
    :param y_pre: array，Predicted value
    :return: float
    '''
    lenall = len(y)
    TP, FP, FN, TN = 0, 0, 0, 0
    for i in range(lenall):
        if y_pre[i] == 1:
            if y[i] == 1:
                TP += 1
            if y[i] == 0:
                FP += 1
        if y_pre[i] == 0:
            if y[i] == 1:
                FN += 1
            if y[i] == 0:
                TN += 1
    member = TP * TN - FP * FN
    mcc = float(TP * TN - FP * FN) / (np.sqrt((TP + FP) * (TP + FN) * (TN + FN) * (TN + FP)))
    acc = float((TP + TN) / (TP + TN + FP + FN))
    recall = float(TP / (TP + FN))
    pre = float(TP / (TP + FP))
    f1 = float(2 * pre * recall / (pre + recall))
    # demember = ((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN)) ** 0.5
    # mcc = member / demember
    return mcc, acc, recall, pre, f1
    
# MARSNet main function
def RUN_MARSNet(parser):
    # data_dir = './GraphProt_CLIP_sequences/'
    posi = parser.posi
    nega = parser.nega
    model_type = parser.model_type
    out_file = parser.out_file
    train = parser.train
    model_file = parser.model_file
    predict = parser.predict
    batch_size = parser.batch_size
    n_epochs = parser.n_epochs
    num_filters = parser.num_filters
    testfile = parser.testfile
    start_time = timeit.default_timer()
    # pdb.set_trace()
    if predict:
        train = False
        if testfile == '':
            print('you need specify the fasta file for predicting when predict is True')
            return
    if train:
        if posi == '' or nega == '':
            print('you need specify the training positive and negative fasta file for training when train is True')
            return
  
    if train:

        print("1011")
        train_bags, train_labels = [], []
        train_bags, train_labels = get_data(posi, nega, channel=7, window_size=101)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=7, window_size=101 + 6,
                              model_file=model_file + '.1011', batch_size=batch_size, n_epochs=n_epochs,
                              num_filters=num_filters)


        print("2011")
        train_bags, train_labels = [], []
        train_bags, train_labels = get_data(posi, nega, channel=3, window_size=201)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=3, window_size=201 + 6,
                              model_file=model_file + '.2011', batch_size=batch_size, n_epochs=n_epochs,
                              num_filters=num_filters)

        print('3011')
        file_out = open('time_train.txt', 'a')
        train_bags, train_labels = get_data(posi, nega, channel=2, window_size=301)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2, window_size=301 + 6,
                              model_file=model_file + '.3011', batch_size=batch_size, n_epochs=n_epochs,
                              num_filters=num_filters)


        print("4011")
        train_bags, train_labels = [], []
        train_bags, train_labels = get_data(posi, nega, channel=1, window_size=401)
        # print(np.array(train_bags).shape)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1, window_size=401 + 6,
                              model_file=model_file + '.4011', batch_size=batch_size, n_epochs=n_epochs,
                              num_filters=num_filters)

        print("1012")
        train_bags, train_labels = [], []
        train_bags, train_labels = get_data(posi, nega, channel=7, window_size=101)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=7, window_size=101 + 6,
                              model_file=model_file + '.1012', batch_size=batch_size, n_epochs=n_epochs,
                              num_filters=num_filters)


        print("2012")
        train_bags, train_labels = [], []
        train_bags, train_labels = get_data(posi, nega, channel=3, window_size=201)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=3, window_size=201 + 6,
                              model_file=model_file + '.2012', batch_size=batch_size, n_epochs=n_epochs,
                              num_filters=num_filters)

        print('3012')
        file_out = open('time_train.txt', 'a')
        train_bags, train_labels = get_data(posi, nega, channel=2, window_size=301)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2, window_size=301 + 6,
                              model_file=model_file + '.3012', batch_size=batch_size, n_epochs=n_epochs,
                              num_filters=num_filters)


        print("4012")
        train_bags, train_labels = [], []
        train_bags, train_labels = get_data(posi, nega, channel=1, window_size=401)
        # print(np.array(train_bags).shape)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1, window_size=401 + 6,
                              model_file=model_file + '.4012', batch_size=batch_size, n_epochs=n_epochs,
                              num_filters=num_filters)

        
        print("1013")
        train_bags, train_labels = [], []
        train_bags, train_labels = get_data(posi, nega, channel=7, window_size=101)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=7, window_size=101 + 6,
                              model_file=model_file + '.1013', batch_size=batch_size, n_epochs=n_epochs,
                              num_filters=num_filters)


        print("2013")
        train_bags, train_labels = [], []
        train_bags, train_labels = get_data(posi, nega, channel=3, window_size=201)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=3, window_size=201 + 6,
                              model_file=model_file + '.2013', batch_size=batch_size, n_epochs=n_epochs,
                              num_filters=num_filters)

        print('3013')
        file_out = open('time_train.txt', 'a')
        train_bags, train_labels = get_data(posi, nega, channel=2, window_size=301)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2, window_size=301 + 6,
                              model_file=model_file + '.3013', batch_size=batch_size, n_epochs=n_epochs,
                              num_filters=num_filters)


        print("4013")
        train_bags, train_labels = [], []
        train_bags, train_labels = get_data(posi, nega, channel=1, window_size=401)
        # print(np.array(train_bags).shape)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1, window_size=401 + 6,
                              model_file=model_file + '.4013', batch_size=batch_size, n_epochs=n_epochs,
                              num_filters=num_filters)


        print("1014")
        train_bags, train_labels = [], []
        train_bags, train_labels = get_data(posi, nega, channel=7, window_size=101)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=7, window_size=101 + 6,
                              model_file=model_file + '.1014', batch_size=batch_size, n_epochs=n_epochs,
                              num_filters=num_filters)


        print("2014")
        train_bags, train_labels = [], []
        train_bags, train_labels = get_data(posi, nega, channel=3, window_size=201)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=3, window_size=201 + 6,
                              model_file=model_file + '.2014', batch_size=batch_size, n_epochs=n_epochs,
                              num_filters=num_filters)

        print('3014')
        file_out = open('time_train.txt', 'a')
        train_bags, train_labels = get_data(posi, nega, channel=2, window_size=301)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2, window_size=301 + 6,
                              model_file=model_file + '.3014', batch_size=batch_size, n_epochs=n_epochs,
                              num_filters=num_filters)


        print("4014")
        train_bags, train_labels = [], []
        train_bags, train_labels = get_data(posi, nega, channel=1, window_size=401)
        # print(np.array(train_bags).shape)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=1, window_size=401 + 6,
                              model_file=model_file + '.4014', batch_size=batch_size, n_epochs=n_epochs,
                              num_filters=num_filters)
        
        end_time = timeit.default_timer()
        file_out.write(str(round(float(end_time - start_time), 3)) + '\n')
        file_out.close()
        # print ("Training final took: %.2f min" % float((end_time - start_time)/60))
    elif predict:
        fw = open(out_file, 'w')
        file_out = open('pre_auc.txt', 'a')
        file_out2 = open('time_test.txt', 'a')


        X_test, X_labels = get_data(testfile, nega, channel=7, window_size=101)
        predict11 = predict_network(model_type, np.array(X_test), channel=7, window_size=101 + 6,
                                    model_file=model_file + '.1011', batch_size=batch_size, n_epochs=n_epochs,
                                    num_filters=num_filters)

        X_test, X_labels = get_data(testfile, nega, channel=3, window_size=201)
        predict12 = predict_network(model_type, np.array(X_test), channel=3, window_size=201 + 6,
                                    model_file=model_file + '.1012', batch_size=batch_size, n_epochs=n_epochs,
                                    num_filters=num_filters)

        X_test, X_labels = get_data(testfile, nega, channel=2, window_size=301)
        predict13 = predict_network(model_type, np.array(X_test), channel=2, window_size=301 + 6,
                                    model_file=model_file + '.1013', batch_size=batch_size, n_epochs=n_epochs,
                                    num_filters=num_filters)

        X_test, X_labels = get_data(testfile, nega, channel=1, window_size=401)
        predict14 = predict_network(model_type, np.array(X_test), channel=1, window_size=401 + 6,
                                    model_file=model_file + '.1014', batch_size=batch_size, n_epochs=n_epochs,
                                    num_filters=num_filters)

        X_test, X_labels = get_data(testfile, nega, channel=7, window_size=101)
        predict21 = predict_network(model_type, np.array(X_test), channel=7, window_size=101 + 6,
                                    model_file=model_file + '.2011', batch_size=batch_size, n_epochs=n_epochs,
                                    num_filters=num_filters)

        X_test, X_labels = get_data(testfile, nega, channel=3, window_size=201)
        predict22 = predict_network(model_type, np.array(X_test), channel=3, window_size=201 + 6,
                                    model_file=model_file + '.2012', batch_size=batch_size, n_epochs=n_epochs,
                                    num_filters=num_filters)

        X_test, X_labels = get_data(testfile, nega, channel=2, window_size=301)
        predict23 = predict_network(model_type, np.array(X_test), channel=2, window_size=301 + 6,
                                    model_file=model_file + '.2013', batch_size=batch_size, n_epochs=n_epochs,
                                    num_filters=num_filters)

        X_test, X_labels = get_data(testfile, nega, channel=1, window_size=401)
        predict24 = predict_network(model_type, np.array(X_test), channel=1, window_size=401 + 6,
                                    model_file=model_file + '.2014', batch_size=batch_size, n_epochs=n_epochs,
                                    num_filters=num_filters)

        X_test, X_labels = get_data(testfile, nega, channel=7, window_size=101)
        predict31 = predict_network(model_type, np.array(X_test), channel=7, window_size=101 + 6,
                                    model_file=model_file + '.3011', batch_size=batch_size, n_epochs=n_epochs,
                                    num_filters=num_filters)

        X_test, X_labels = get_data(testfile, nega, channel=3, window_size=201)
        predict32 = predict_network(model_type, np.array(X_test), channel=3, window_size=201 + 6,
                                    model_file=model_file + '.3012', batch_size=batch_size, n_epochs=n_epochs,
                                    num_filters=num_filters)

        X_test, X_labels = get_data(testfile, nega, channel=2, window_size=301)
        predict33 = predict_network(model_type, np.array(X_test), channel=2, window_size=301 + 6,
                                    model_file=model_file + '.3013', batch_size=batch_size, n_epochs=n_epochs,
                                    num_filters=num_filters)

        X_test, X_labels = get_data(testfile, nega, channel=1, window_size=401)
        predict34 = predict_network(model_type, np.array(X_test), channel=1, window_size=401 + 6,
                                    model_file=model_file + '.3014', batch_size=batch_size, n_epochs=n_epochs,
                                    num_filters=num_filters)

        X_test, X_labels = get_data(testfile, nega, channel=7, window_size=101)
        predict41 = predict_network(model_type, np.array(X_test), channel=7, window_size=101 + 6,
                                    model_file=model_file + '.4011', batch_size=batch_size, n_epochs=n_epochs,
                                    num_filters=num_filters)

        X_test, X_labels = get_data(testfile, nega, channel=3, window_size=201)
        predict42 = predict_network(model_type, np.array(X_test), channel=3, window_size=201 + 6,
                                    model_file=model_file + '.4012', batch_size=batch_size, n_epochs=n_epochs,
                                    num_filters=num_filters)

        X_test, X_labels = get_data(testfile, nega, channel=2, window_size=301)
        predict43 = predict_network(model_type, np.array(X_test), channel=2, window_size=301 + 6,
                                    model_file=model_file + '.4013', batch_size=batch_size, n_epochs=n_epochs,
                                    num_filters=num_filters)

        X_test, X_labels = get_data(testfile, nega, channel=1, window_size=401)
        predict44 = predict_network(model_type, np.array(X_test), channel=1, window_size=401 + 6,
                                    model_file=model_file + '.4014', batch_size=batch_size, n_epochs=n_epochs,
                                    num_filters=num_filters)
        preA = (1 * predict11 + 2 * predict12 + 4 * predict13 + 3 * predict14) / 10.0
        preB = (1 * predict21 + 2 * predict22 + 4 * predict23 + 3 * predict24) / 10.0
        preC = (1 * predict31 + 2 * predict32 + 4 * predict33 + 3 * predict34) / 10.0
        preD = (1 * predict41 + 2 * predict42 + 4 * predict43 + 3 * predict44) / 10.0
        

        predict_sum =  (preA + preB + preC + preD) / 4.0
        p = predict_sum
        p = np.around(p, 0).astype(int)
        mcc, accuracy, recall, precision, f1 = Indicators(X_labels, p)
        ap = average_precision_score(X_labels, p)
        print('accuracy:' accuracy, 'precision:' precision, 'recall:' recall, 'mcc:' mcc, 'f1:' f1, 'ap:' ap)

        auc = roc_auc_score(X_labels, predict_sum)
        print('AUC:{:.3f}'.format(auc))
        myprob = "\n".join(map(str, predict_sum))
        fw.write(myprob)
        fw.close()
        file_out.write(str(round(float(auc), 3)) + '\n')
        file_out.close()
        end_time = timeit.default_timer()
        file_out2.write(str(round(float(end_time - start_time), 3)) + '\n')
        file_out2.close()
    else:
        print('please specify that you want to train the mdoel or predict for your own sequences')


# MCNN framework optional parameters
def parse_arguments(parser):
    parser.add_argument('--posi', type=str, metavar='<postive_sequecne_file>',
                        help='The fasta file of positive training samples')
    parser.add_argument('--nega', type=str, metavar='<negative_sequecne_file>',
                        help='The fasta file of negative training samples')
    parser.add_argument('--model_type', type=str, default='DRSN', help='The default model is DRSN')
    parser.add_argument('--out_file', type=str, default='prediction.txt',
                        help='The output file used to store the prediction probability of the testing sequences')
    parser.add_argument('--train', type=bool, default=True,
                        help='The path to the Pickled file containing durations between visits of patients. If you are not using duration information, do not use this option')
    parser.add_argument('--model_file', type=str, default='model.pkl',
                        help='The file to save model parameters. Use this option if you want to train on your sequences or predict for your sequences')
    parser.add_argument('--predict', type=bool, default=False,
                        help='Predicting the RNA-protein binding sites for your input sequences, if using train, then it will be False')
    parser.add_argument('--testfile', type=str, default='',
                        help='the test fast file for sequences you want to predict for, you need specify it when using predict')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='The size of a single mini-batch (default value: 128)')
    parser.add_argument('--num_filters', type=int, default=16,
                        help='The number of filters for DRSN (default value: 16)')
    parser.add_argument('--n_epochs', type=int, default=50, help='The number of training epochs (default value: 50)')

    args = parser.parse_args()  
    return args


parser = argparse.ArgumentParser()
args = parse_arguments(parser)
print(args)
RUN_MARSNet(args)


