import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class DataLoaderS(object):

    def __init__(self, file_name, train, valid, device, horizon, window, Lagmax, normalize=2, interval = 2):
        # 预测用的滑动窗口 一般取为168
        self.P = window
        # 预测多久的未来值 目前取3
        self.h = horizon
        # 最大滞后时间
        self.Lagmax = Lagmax
        # 间隔
        self.interval = 2
        fin = open(file_name)
        # 原始数据 从文件中读取到的
        self.rawdat = np.loadtxt(fin, delimiter=',')
        # 按照原始数据得到为0的数据 对原始数据进行标准化等处理 存储到self.dat中
        self.dat = np.zeros(self.rawdat.shape)
        # n 是原始数据 总的长度 m是 变量数量
        self.n, self.m = self.dat.shape
        # 有三种对原始数据标准化的方法 0，1，2
        self.normalize = 2
        # 一维数组 变量数量
        self.scale = np.ones(self.m)
        # 执行标准化程序 将读取到的时间序列数据 和 scale 两部分的内容进行标准化
        self._normalized(normalize)
        # 将数据分割为训练集 验证集和 测试集 训练集的长度 训练+验证集的长度 总长度
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)
        # 转换为 tensor
        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)
        self.scale = self.scale.to(device)
        # Varibale包含三个属性：
        # data：存储了Tensor，是本体的数据； grad：保存了data的梯度，本事是个Variable而非Tensor，data形状一致
        # grad_fn：指向Function对象，用于反向传播的梯度计算之用
        self.scale = Variable(self.scale)
        # 标准差
        self.rse = normal_std(tmp)
        # 均方差
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device

    # 标准化
    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                # 对应时间序列 先取绝对值 再取最大值
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                # 将原始数据 除以最大值的绝对值 进行标准化  得到损失之后需要再乘以最大值的绝对值 复原回来
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):
        #  train_set 得到的是要预测的值的索引
        # 有间隔
        train_set = range(self.P + self.Lagmax * self.interval + self.h - 1, train)
        # 无间隔
        #train_set = range(self.P + self.Lagmax + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        # 分别给 test valid test 数据赋值
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)
    # idx_set 要预测的值的索引
    # 读取数据  X 为用于预测的数据   Y为预测值
    def _batchify(self, idx_set, horizon):
        # 原始数据中数据那些部分被划分为数据集
        # idx_set 是 要进行的预测点的索引
        n = len(idx_set)
        #  n 数据集中 被划分为训练集 验证集和测试集的索引   self.Lagmax 最大滞后时间
        #  self.P 滑动窗口   self.m 变量数
        X = torch.zeros((n, self.Lagmax+1, self.P, self.m))
        Y = torch.zeros((n, self.m))
        # 生成数据集
        for i in range(n):
            # 间隔
            end = idx_set[i] - self.h - self.Lagmax * self.interval + 1
            start = end - self.P
            for j in range(self.Lagmax+1):
                # 无间隔
                # X[i, j, :, :] = torch.from_numpy(self.dat[start+j:end+j, :])
                # 有间隔
                X[i, j, :, :] = torch.from_numpy(self.dat[start + j * self.interval:end + j * self.interval, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]

    # inputs 输入, targets 预测目标,  batch_size的大小 这个函数会在训练的时候用
    # 输入数据是以一个 batch_size 为范围的
    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            # randperm(n)---返回一个从 0 - n - 1 的随机整数permutation
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        # 每个 batch_size 开始的索引
        start_idx = 0
        while (start_idx < length):
            # 每个 batch_size 结束的索引
            end_idx = min(length, start_idx + batch_size)
            # batch_size 索引
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            # X 和 Y 是两个tensor
            # 生成器的状态保存 生成器函数在每次执行时都会保持其状态。
            # 这意味着它可以用于生成无限序列或大数据集，而不必将所有数据存储在内存中。
            yield Variable(X), Variable(Y)
            start_idx += batch_size

class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    return adj

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


