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
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score
import random
import gzip
import pickle
import timeit
import argparse

import json
from contextlib import contextmanager

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For reproducibility (may slow down a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@contextmanager
def cuda_profiler(enabled: bool = True):
    """Context manager to record peak GPU memory and wall time."""
    start = timeit.default_timer()
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    try:
        yield
    finally:
        if enabled and torch.cuda.is_available():
            torch.cuda.synchronize()

def get_cuda_mem():
    if not torch.cuda.is_available():
        return {"max_allocated_mb": None, "max_reserved_mb": None}
    return {
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 3),
        "max_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024**2), 3),
    }

import matplotlib.pyplot as plt
import torch.utils.data as Data

from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from seq_motifs import get_motif

# GPU or CPU
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


# If L<W,use 'N' to pad to the length of max_len.
# If L>W,only the RNA sequence with the length of max_len is retained from front to back.
def padding_sequence(seq, max_len=501, repkey='N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len - seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq



# RNA sequence is turned into a one-hot encoding matrix with (L+6) rows and 4 columns.
# The first 3 rows and the last 3 rows are filled with N
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
            loss_list.append(loss.item())  
        return sum(loss_list) / len(loss_list)

    def fit(self, X, y, batch_size=32, nb_epoch=10, validation_data=()):

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

    def __init__(self, in_channels, out_channels, stride=1, use_shrinkage: bool = True):
        super().__init__()
        self.shrinkage = Shrinkage(out_channels, gap_size=(1, 1)) if use_shrinkage else nn.Identity()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * RSBU_CW.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * RSBU_CW.expansion),
            self.shrinkage
        )
        # shortcut
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != RSBU_CW.expansion * out_channels:
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


    
class extract_motifs(nn.Module):
    def __init__(self, nb_filter, channel=7, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3), labcounts=32,
                 window_size=12, hidden_size=256, stride=(1, 1), padding=0):
        super(extract_motifs, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(pool_size, stride=stride)
        out1_size = int((window_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1] + 1)
        maxpool_size = int((out1_size + 2 * padding - (pool_size[1] - 1) - 1) / stride[1] + 1)
        P=32
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, P, kernel_size=(1, 10), stride=stride, padding=padding),
            nn.BatchNorm2d(P),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, stride=stride))
        out2_size = int((maxpool_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1] + 1)
        maxpool2_size = int((out2_size + 2 * padding - (pool_size[1] - 1) - 1) / stride[1] + 1)
        self.drop1 = nn.Dropout(p=0.25)
        print('maxpool_size', maxpool_size)
        self.fc1 = nn.Linear(maxpool2_size * P, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        # Return logits for CrossEntropyLoss
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
        self.eval()
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
            if cuda:
                x = x.cuda()
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            temp = probs.data.cpu().numpy()
        return temp[:, 1]


class ARSNet(nn.Module):
    def __init__(self, nb_filter, channel=7, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3), labcounts=32,
                 window_size=12, hidden_size=256, stride=(1, 1), padding=0,
                 use_shrinkage: bool = True, use_eca: bool = True, use_cbam: bool = True):
        super(ARSNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, stride=stride))
        out1_size = int((window_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1] + 1)
        maxpool_size = int((out1_size + 2 * padding - (pool_size[1] - 1) - 1) / stride[1] + 1)
        self.reslayer1 = self._make_layer(partial(RSBU_CW, use_shrinkage=use_shrinkage), 16, 1, 1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 10), stride=stride, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, stride=stride))
        out2_size = int((maxpool_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1] + 1)
        maxpool2_size = int((out2_size + 2 * padding - (pool_size[1] - 1) - 1) / stride[1] + 1)
        self.reslayer2 = self._make_layer(partial(RSBU_CW, use_shrinkage=use_shrinkage), 32, 1, 1)
        self.reslayer20 = ECAlayer(32) if use_eca else nn.Identity()
        self.reslayer21 = cbamblock(32) if use_cbam else nn.Identity()
        self.drop1 = nn.Dropout(p=0.25)
        print('maxpool_size', maxpool_size)
        self.fc1 = nn.Linear(maxpool2_size * 32, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(out_channels, out_channels, stride))
            # self.in_channels kept for compatibility
            self.in_channels = out_channels * 1
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
        # Return logits for CrossEntropyLoss
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
        self.eval()
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
            if cuda:
                x = x.cuda()
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            temp = probs.data.cpu().numpy()
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


def train_network(model_type, X_train, y_train, channel = 7, window_size = 107, model_file = 'model.pkl', batch_size = 100, n_epochs = 50, num_filters = 16,
                  motif = False, motif_seqs = [], motif_outdir = 'motifs',
                  seed: int = 0, use_shrinkage: bool = True, use_eca: bool = True, use_cbam: bool = True,
                  profile: bool = False, profile_out: str = ''):
    print ('model training for ', model_type)
    seed_everything(seed)
    #nb_epos= 5
    prof = {'seed': seed, 'channel': channel, 'window_size': window_size, 'batch_size': batch_size, 'n_epochs': n_epochs,
            'num_filters': num_filters, 'use_shrinkage': use_shrinkage, 'use_eca': use_eca, 'use_cbam': use_cbam}
    if model_type == 'ARSNet':
        model = ARSNet(nb_filter =num_filters, labcounts = 4, window_size = window_size, channel = channel,
                       use_shrinkage=use_shrinkage, use_eca=use_eca, use_cbam=use_cbam)
    elif model_type == 'extract_motifs':
        model = extract_motifs(nb_filter =num_filters, labcounts = 4, window_size = window_size, channel = channel)        
    else:
        print ('only support ARSNet model')

    if cuda:
        model = model.cuda()
    clf = Estimator(model)
    clf.compile(optimizer=torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay = 0.0001),
                loss=nn.CrossEntropyLoss())
    with cuda_profiler(enabled=profile):
        t0 = timeit.default_timer()
        clf.fit(X_train, y_train, batch_size=batch_size, nb_epoch=n_epochs)
        t1 = timeit.default_timer()
    prof['train_seconds'] = round(float(t1 - t0), 6)
    prof.update(get_cuda_mem())
    if profile and profile_out:
        os.makedirs(os.path.dirname(profile_out), exist_ok=True)
        with open(profile_out, 'a') as f:
            f.write(json.dumps({'phase':'train', **prof}) + '\n')
    if motif and channel == 1:
        detect_motifs(model, motif_seqs, X_train, motif_outdir)

    torch.save(model.state_dict(), model_file)


def predict_network(model_type, X_test, channel = 7, window_size = 107, model_file = 'model.pkl', batch_size = 100, n_epochs = 50, num_filters = 16,
                    use_shrinkage: bool = True, use_eca: bool = True, use_cbam: bool = True,
                    profile: bool = False, profile_out: str = ''):
    print ('model training for ', model_type)
    seed_everything(seed)
    #nb_epos= 5
    prof = {'seed': seed, 'channel': channel, 'window_size': window_size, 'batch_size': batch_size, 'n_epochs': n_epochs,
            'num_filters': num_filters, 'use_shrinkage': use_shrinkage, 'use_eca': use_eca, 'use_cbam': use_cbam}
    if model_type == 'ARSNet':
        model = ARSNet(nb_filter =num_filters, labcounts = 4, window_size = window_size, channel = channel,
                       use_shrinkage=use_shrinkage, use_eca=use_eca, use_cbam=use_cbam)
    
    else:
        print ('only support ARSNet model')

    if cuda:
        model = model.cuda()

    # model.load_state_dict(torch.load(model_file))
    model.load_state_dict(torch.load(model_file), strict=False)

    try:
        with cuda_profiler(enabled=profile):
            t0 = timeit.default_timer()
            pred = model.predict_proba(X_test)
            t1 = timeit.default_timer()
        if profile and profile_out:
            info = {'phase':'predict', 'channel':channel, 'window_size':window_size, 'batch_size':batch_size,
                    'num_filters':num_filters, 'use_shrinkage':use_shrinkage, 'use_eca':use_eca, 'use_cbam':use_cbam,
                    'predict_seconds': round(float(t1-t0),6)}
            info.update(get_cuda_mem())
            os.makedirs(os.path.dirname(profile_out), exist_ok=True)
            with open(profile_out, 'a') as f:
                f.write(json.dumps(info) + '\n')
    except: #to handle the out-of-memory when testing
        test_batch = batch(X_test)
        pred = []
        for test in test_batch:
            pred_test1 = model.predict_proba(test)[:, 1]
            pred = np.concatenate((pred, pred_test1), axis = 0)
    return pred


def batch(tensor, batch_size = 1000):
    tensor_list = []
    length = tensor.shape[0]
    i = 0
    while True:
        if (i+1) * batch_size >= length:
            tensor_list.append(tensor[i * batch_size: length])
            return tensor_list
        tensor_list.append(tensor[i * batch_size: (i+1) * batch_size])
        i += 1

def detect_motifs(model, test_seqs, X_train, output_dir = 'motifs', channel = 1):
    if channel == 1:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for param in model.parameters():
            layer1_para =  param.data.cpu().numpy()
            break
        	#test_data = load_graphprot_data(protein, train = True)
        	#test_seqs = test_data["seq"]
        N = len(test_seqs)
        if N > 15000: # do need all sequence to generate motifs and avoid out-of-memory
        	sele = 15000
        else:
        	sele = N
        ix_all = np.arange(N)
        np.random.shuffle(ix_all)
        ix_test = ix_all[0:sele]

        X_train = X_train[ix_test, :, :, :]
        test_seq = []
        for ind in ix_test:
        	test_seq.append(test_seqs[ind])
        test_seqs = test_seq
        filter_outs = model.layer1out(X_train)[:,:, 0, :]
        get_motif(layer1_para[:,0, :, :], filter_outs, test_seqs, dir1 = output_dir)


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

def run(parser):
    posi = parser.posi
    nega = parser.nega
    out_file = parser.out_file
    train = parser.train
    model_type = parser.model_type
    model_file = parser.model_file
    predict = parser.predict
    batch_size = parser.batch_size
    n_epochs = parser.n_epochs
    num_filters = parser.num_filters
    testfile = parser.testfile
    start_time = timeit.default_timer()
    motif = parser.motif
    motif_outdir = parser.motif_dir

    #pdb.set_trace()
    if predict:
        train = False
        if testfile == '':
            print ('you need specify the fasta file for predicting when predict is True')
            return
    if train:
        if posi == '' or nega == '':
            print ('you need specify the training positive and negative fasta file for training when train is True')
            return


    if train:
        # Configurable multi-scale / multi-ensemble training
        # Default: 4 ensembles (A-D) x 4 window sizes (101/201/301/401) following original code naming.
        windows = parser.window_sizes
        channels = parser.channels
        seeds = parser.seeds
        profile_out = parser.profile_out

        assert len(windows) == len(channels), "window_sizes and channels must have same length"
        if len(seeds) == 0:
            seeds = [0, 1, 2, 3]

        # Training each ensemble member independently
        for ens_id, seed in enumerate(seeds):
            for w, ch in zip(windows, channels):
                print(f"[TRAIN] ens={ens_id} seed={seed} window={w} channel={ch}")
                X_train, y_train = get_data(posi, nega, channel=ch, window_size=w)
                out_model = f"{model_file}.ens{ens_id}.w{w}.pt"
                train_network(
                    model_type,
                    np.array(X_train),
                    np.array(y_train),
                    channel=ch,
                    window_size=w + 6,
                    model_file=out_model,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    num_filters=num_filters,
                    seed=seed,
                    use_shrinkage=(not parser.disable_shrinkage),
                    use_eca=(not parser.disable_eca),
                    use_cbam=(not parser.disable_cbam),
                    profile=parser.profile,
                    profile_out=profile_out,
                )

        # Record overall wall time (optional legacy)
        end_time = timeit.default_timer()
        if parser.log_time:
            with open(parser.time_train_file, 'a') as f:
                f.write(str(round(float(end_time - start_time), 3)) + '\n')
        # print ("Training final took: %.2f min" % float((end_time - start_time)/60))
    elif predict:
        fw = open(out_file, 'w')
        profile_out = parser.profile_out

        windows = parser.window_sizes
        channels = parser.channels
        seeds = parser.seeds
        if len(seeds) == 0:
            seeds = [0, 1, 2, 3]

        weights = parser.ensemble_weights
        if len(weights) == 0:
            # Original fixed weights for 4 scales
            weights = [1.0, 2.0, 4.0, 3.0]
        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()

        assert len(windows) == len(channels) == len(weights), "window_sizes/channels/ensemble_weights must match length"

        # Preload test labels for metrics (use the first scale to get labels; labels are same across scales)
        _, y_true = read_seq_graphprot(testfile, label=1)
        # We'll obtain y_true from get_data for each scale (safer)
        # For each ensemble member, compute weighted multi-scale prediction
        ens_preds = []
        t0_all = timeit.default_timer()
        for ens_id, seed in enumerate(seeds):
            scale_preds = []
            for (w, ch) in zip(windows, channels):
                X_test, y_labels = get_data(testfile, nega, channel=ch, window_size=w)
                y_true = np.array(y_labels)
                out_model = f"{model_file}.ens{ens_id}.w{w}.pt"
                pred = predict_network(
                    model_type,
                    np.array(X_test),
                    channel=ch,
                    window_size=w + 6,
                    model_file=out_model,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    num_filters=num_filters,
                    use_shrinkage=(not parser.disable_shrinkage),
                    use_eca=(not parser.disable_eca),
                    use_cbam=(not parser.disable_cbam),
                    profile=parser.profile,
                    profile_out=profile_out,
                )
                scale_preds.append(pred)
            scale_preds = np.vstack(scale_preds)  # [n_scales, n_samples]
            ens_pred = (weights[:, None] * scale_preds).sum(axis=0)
            ens_preds.append(ens_pred)

        ens_preds = np.vstack(ens_preds)  # [n_ensembles, n_samples]
        predict_sum = ens_preds.mean(axis=0)  # average across ensembles
        t1_all = timeit.default_timer()

        # Metrics
        y_hat = (predict_sum >= 0.5).astype(int)
        mcc, accuracy, recall, precision, f1 = Indicators(y_true, y_hat)
        ap = average_precision_score(y_true, predict_sum)  # IMPORTANT: use probabilities for AP
        auc_val = roc_auc_score(y_true, predict_sum)

        print('accuracy:', accuracy, 'precision:', precision, 'recall:', recall, 'mcc:', mcc, 'f1:', f1, 'ap:', ap)
        print('AUC:{:.3f}'.format(auc_val))

        fw.write("\n".join(map(str, predict_sum)))
        fw.close()

        if parser.log_time:
            with open(parser.pre_auc_file, 'a') as f:
                f.write(str(round(float(auc_val), 3)) + '\n')
            with open(parser.time_test_file, 'a') as f:
                f.write(str(round(float(t1_all - t0_all), 3)) + '\n')

        if parser.save_metrics_json:
            os.makedirs(os.path.dirname(parser.metrics_json), exist_ok=True)
            with open(parser.metrics_json, 'w') as f:
                json.dump({
                    'auc': float(auc_val),
                    'ap': float(ap),
                    'acc': float(accuracy),
                    'mcc': float(mcc),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'n_samples': int(len(y_true)),
                    'window_sizes': windows,
                    'channels': channels,
                    'weights': weights.tolist(),
                    'seeds': seeds,
                    'disable_shrinkage': parser.disable_shrinkage,
                    'disable_eca': parser.disable_eca,
                    'disable_cbam': parser.disable_cbam,
                    'seconds_total_predict': float(t1_all - t0_all),
                }, f, indent=2)

    elif motif:
        motif_seqs = []
        data = read_data_file(posi, nega)
        motif_seqs = data['seq']
        if posi == '' or nega == '':
            print ('To identify motifs, you need training positive and negative sequences using CNN.')
        train_bags, train_labels = get_data(posi, nega, channel = 1, window_size = 401)
        train_network("extract_motifs", np.array(train_bags), np.array(train_labels), channel = 1, window_size = 401 + 6, model_file = model_file + '.401', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, motif = motif, motif_seqs = motif_seqs, motif_outdir = motif_outdir)
    else:
        print ('please specify that you want to train the mdoel or predict for your own sequences')
#ARSNet

def parse_arguments(parser):
    parser.add_argument('--posi', type=str, metavar='<postive_sequecne_file>', help='The fasta file of positive training samples')
    parser.add_argument('--nega', type=str, metavar='<negative_sequecne_file>', help='The fasta file of negative training samples')
    parser.add_argument('--model_type', type=str, default='ARSNet', help='The default model is ARSNet')
    parser.add_argument('--out_file', type=str, default='prediction.txt', help='Store prediction probability of testing sequences')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--model_file', type=str, default='model', help='Prefix to save model parameters (per ensemble & scale)')
    parser.add_argument('--predict', action='store_true', help='Run prediction on a test fasta')
    parser.add_argument('--testfile', type=str, default='', help='Test fasta file (required when --predict)')
    parser.add_argument('--batch_size', type=int, default=128, help='Mini-batch size (default: 128)')
    parser.add_argument('--num_filters', type=int, default=16, help='Number of filters for the first CNN (default: 16)')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of training epochs (default: 50)')

    # Multi-scale ensemble settings
    parser.add_argument('--window_sizes', type=int, nargs='+', default=[101, 201, 301, 401],
                        help='Window sizes for multi-scale ensemble (default: 101 201 301 401)')
    parser.add_argument('--channels', type=int, nargs='+', default=[7, 3, 2, 1],
                        help='Channels per window size (default: 7 3 2 1)')
    parser.add_argument('--ensemble_weights', type=float, nargs='+', default=[],
                        help='Weights per scale (same length as window_sizes). If empty, use original 1 2 4 3.')
    parser.add_argument('--seeds', type=int, nargs='*', default=[],
                        help='Seeds for ensemble members. If empty, use 0 1 2 3 (4 ensembles).')

    # Ablation switches
    parser.add_argument('--disable_shrinkage', action='store_true', help='Ablation: disable residual shrinkage (soft-thresholding)')
    parser.add_argument('--disable_eca', action='store_true', help='Ablation: disable ECA attention')
    parser.add_argument('--disable_cbam', action='store_true', help='Ablation: disable CBAM attention')

    # Profiling & logging
    parser.add_argument('--profile', action='store_true', help='Record time & peak GPU memory to profile_out')
    parser.add_argument('--profile_out', type=str, default='results/profile.jsonl', help='Append JSONL profiling records here')
    parser.add_argument('--log_time', action='store_true', help='Keep legacy time/auc text logs (optional)')
    parser.add_argument('--time_train_file', type=str, default='time_train.txt')
    parser.add_argument('--time_test_file', type=str, default='time_test.txt')
    parser.add_argument('--pre_auc_file', type=str, default='pre_auc.txt')

    parser.add_argument('--save_metrics_json', action='store_true', help='Save metrics json for this run')
    parser.add_argument('--metrics_json', type=str, default='results/metrics.json')

    # Motif
    parser.add_argument('--motif', action='store_true', help='Identify motifs from sequences')
    parser.add_argument('--motif_dir', type=str, default='motifs', help='Output dir for motifs')

    args = parser.parse_args()
    return args

parser = argparse.ArgumentParser()
args = parse_arguments(parser)
print (args)
run(args)
