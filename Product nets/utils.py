#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'CLH'


# from __future__ import print_function
# from __future__ import absolute_import
# from __future__ import division

import sys

if sys.version[0] == '2':
    import cPickle as pkl
else:
    import pickle as pkl

import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix

DTYPE = tf.float32
FIELD_SIZES = [0] * 26

with open(r'./data/featindex.txt','r',encoding='utf-8') as fin:
    for line in fin.readlines():
        line = line.strip().split(':')
        if len(line) > 1:
            f_index = int(line[0]) - 1
            FIELD_SIZES[f_index] += 1
print('field sizes:',FIELD_SIZES)
FIELD_OFFSETS = [sum(FIELD_SIZES[:i]) for i in range(len(FIELD_SIZES))]
INPUT_DIM = sum(FIELD_SIZES)
OUTPUT_DIM = 1
STDDEV = 1e-3
MINVAL = -1e-3
MAXVAL = 1e-3

def read_data(file_name):
    X = []
    y = []
    D = []
    with open(file_name,'r',encoding='utf-8') as fin:
        for line in fin.readlines():
            features = line.strip().split()
            y_i = int(features[0])
            x_i = [int(x.split(':')[0]) for x in features[1:]]
            d_i = [int(x.split(':')[1]) for x in features[1:]]
            y.append(y_i), X.append(x_i), D.append(d_i)
    y = np.reshape(np.array(y),[-1])
    X = construct_coo_mat(zip(X,D),(len(X),INPUT_DIM)).tocsr()
    # print(X.shape[0])
    return X,y

def construct_coo_mat(data, shape):
    coo_rows = []
    coo_cols = []
    coo_data = []
    n = 0 # 样本的index
    for x, d in data:
        coo_rows.extend([n]*len(x))
        coo_cols.extend(x)
        coo_data.extend(d)
        n += 1
    coo_rows = np.array(coo_rows)
    coo_cols = np.array(coo_cols)
    coo_data = np.array(coo_data)

    return coo_matrix((coo_data,(coo_rows,coo_cols)),shape=shape)

def shuffle(data):
    X, y = data
    index = np.arange(X.shape[0])
    for i in range(7):
        np.random.shuffle(index)
    return X[index],y[index]

def init_var_map(init_vars, init_path=None):
    # 反序列化读取
    if init_path is not None:
        load_var_map = pkl.load(open(init_path),'rb')
        print('load variable map from',init_path,load_var_map.keys())
    var_map = {}
    for var_name, var_shape, init_method, dtype in init_vars:
        if init_method == 'zero':
            var_map[var_name] = tf.Variable(tf.zeros(var_shape,dtype=dtype),name = var_name, dtype= dtype)
        elif init_method == 'one':
            var_map[var_name] = tf.Variable(tf.ones(var_shape,dtype=dtype),name = var_name, dtype= dtype)
        elif init_method == 'normal':
            var_map[var_name] = tf.Variable(tf.random_normal(var_shape,mean=0.0,stddev=STDDEV,dtype=dtype),name = var_name, dtype= dtype)
        elif init_method == 'tnormal':
            var_map[var_name] = tf.Variable(tf.truncated_normal(var_shape,mean=0.0,stddev=STDDEV,dtype=dtype),name = var_name, dtype= dtype)
        elif init_method == 'uniform':
            var_map[var_name] = tf.Variable(tf.random_uniform(var_shape,mean=0.0,stddev=STDDEV,dtype=dtype),name = var_name, dtype= dtype)
        elif init_method == 'xavier':
            maxval = np.sqrt(6./np.sum(var_shape))
            minval = - maxval
            var_map[var_name] = tf.Variable(tf.random_uniform(var_shape,minval=minval,maxval=maxval,dtype=dtype),name = var_name, dtype= dtype)
        elif isinstance(init_method, int) or isinstance(init_method, float):
            var_map[var_name] = tf.Variable(tf.ones(var_shape,dtype=dtype)*init_method,name = var_name, dtype= dtype)
        elif init_method in load_var_map:
            if load_var_map[init_method].shape == tuple(var_shape):
                var_map[var_name] = tf.Variable(load_var_map[init_method], name=var_name, dtype=dtype)
            else:
                print('Bad param: init method', init_method, 'shape', var_shape, load_var_map[init_method].shape)
        else:
            print('Bad param: init method', init_method)

    return var_map

def get_optimizer(opt_algo,learning_rate,loss):
    if opt_algo == 'adaldeta':
        return tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adagrade':
        return tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adam':
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'ftrl':
        return tf.train.FtrlOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'padagrad':
        return tf.train.ProximalAdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'pgd':
        return tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    else:
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

def activate(weights, activation_function):
    if activation_function == 'sigmoid':
        return tf.nn.sigmoid(weights)
    elif activation_function == 'softmax':
        return tf.nn.softmax(weights)
    elif activation_function == 'relu':
        return tf.nn.relu(weights)
    elif activation_function == 'tanh':
        return tf.nn.tanh(weights)
    elif activation_function == 'elu':
        return tf.nn.elu(weights)
    elif activation_function == 'none':
        return weights
    else:
        return weights

def split_data(data, skip_empty=True):
    '''
    按field划分数据
    :param data: [X,y] data[0]=X, data[1]=y
    :param skip_empty:
    :return: 按field划分的X，y
    '''
    fields = []
    for i in range(len(FIELD_OFFSETS)-1):
        start_index = FIELD_OFFSETS[i]
        end_index = FIELD_OFFSETS[i+1]
        if skip_empty and start_index == end_index:
            continue
        field_i = data[0][:,start_index:end_index]
        fields.append(field_i)
    # 勿忘加入最后一个field
    fields.append(data[0][:,FIELD_OFFSETS[-1]:])
    return fields, data[1]

def slice(csr_data, start=0, size=-1):
    '''
    切片
    :param csr_data:
    :param start:
    :param size:
    :return:
    '''
    if not isinstance(csr_data[0],list):
        if size == -1 or start+size >= csr_data[0].shape[0]:
            slc_data = csr_data[0][start:]
            slc_labels = csr_data[1][start:]
        else:
            slc_data = csr_data[0][start:start+size]
            slc_labels = csr_data[1][start:start+size]
    else:
        if size == -1 or start+size >= csr_data[0][0].shape[0]:
            slc_data=[]
            for d_i in csr_data[0]:
                slc_data.append(d_i[start:])
            slc_labels = csr_data[1][start:]
        else:
            slc_data=[]
            # 逐个将样本加入csr_data[0]=X
            for d_i in csr_data[0]:
                slc_data.append(d_i[start:start+size])
            slc_labels = csr_data[1][start:start+size]
    return csr_2_input(slc_data),slc_labels

def csr_2_input(csr_mat):
    if not isinstance(csr_mat,list):
        coo_mat = csr_mat.tocoo()
        indices = np.vstack((coo_mat.row,coo_mat.col)).transpose()
        values = csr_mat.data
        shape = csr_mat.shape
        return indices,values,shape
    else:
        inputs = []
        for csr_i in csr_mat:
            inputs.append(csr_2_input(csr_i))
        return inputs

# if __name__ == "__main__":
#     read_data(r'./data/train.fm.txt')
