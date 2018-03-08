#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'CLH'


# from __future__ import print_function
# from __future__ import absolute_import
# from __future__ import division

import sys
sys.path.append(r"./")
if sys.version[0] == '2':
    import cPickle as pkl
else:
    import pickle as pkl
import utils
import tensorflow as tf
import numpy as np

dtype = utils.DTYPE
# print(dtype)
class Model(object):
    def __init__(self):
        self.sess = None
        self.X = None
        self.y = None
        self.layer_keeps = None
        self.vars = None
        self.keep_prob_train = None
        self.keep_prob_test = None

    def run(self, fetches, X=None, y=None, mode='train'):
        feed_dict = {}
        if type(self.X) is list:
            for i in range(len(X)):
                feed_dict[self.X[i]] = X[i]
        else:
            feed_dict[self.X] = X
        if y is not None:
            feed_dict[self.y] = y
        if self.layer_keeps is not None:
            if mode == 'train':
                feed_dict[self.layer_keeps] = self.keep_prob_train
            elif mode == 'test':
                feed_dict[self.layer_keeps] = self.keep_prob_test
        return self.sess.run(fetches,feed_dict)

    def dump(self,model_path):
        var_map = {}
        for name,var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path,'wb'))
        print('model dumped at', model_path)


class LR(Model):
    def __init__(self,input_dim = None, output_dim = 1, init_path = None, opt_algo='gd',
                 learning_rate = 1e-2, l2_weight=0, random_seed=None):
        Model.__init__(self)
        init_vars = [('w',[input_dim,output_dim], 'xavier', dtype),
                     ('b',[output_dim],'zero',dtype)]
        self.graph = tf.Graph()
        # 设置新的默认图
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars,init_path)

            w = self.vars['w']
            b = self.vars['b']
            xw = tf.sparse_tensor_dense_matmul(self.X,w)
            logits = tf.reshape(xw + b,[-1])
            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits = logits))\
                        + l2_weight*tf.nn.l2_loss(xw)
            self.optimizer = utils.get_optimizer(opt_algo,learning_rate,self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True #允许显存增长。如果设置为 True，分配器不会预先分配一定量 GPU 显存，而是先分配一小块，必要时增加显存分配
            self.sess = tf.Session(config = config)
            tf.global_variables_initializer().run(session=self.sess) # 前者为变量初始化

class FM(Model):
    def __init__(self,input_dim = None, output_dim = 1, factor_dim = 10,init_path = None, opt_algo='gd',
                 learning_rate = 1e-2, l2_weight=0, l2_v =0, random_seed=None):
        Model.__init__(self)
        init_vars = [('w',[input_dim,output_dim],'xavier',dtype),
                     ('v',[input_dim,factor_dim],'xavier',dtype),
                     ('b',[output_dim],'zero',dtype)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = tf.sparse_placeholder(dtype) # n * input_dim
            self.y = tf.placeholder(dtype) # n * 1
            self.vars = utils.init_var_map(init_vars,init_path)

            w = self.vars['w'] # input_dim * output_dim
            v = self.vars['v'] # input_dim * factor_dim
            b = self.vars['b'] # input_dim

            # n * input_dim
            X_square = tf.SparseTensor(self.X.indices,tf.square(self.X.values),tf.to_int64(tf.shape(self.X)))
            # n * input_dim * input_dim * factor_dim => n * factor_dim
            xv = tf.square(tf.sparse_tensor_dense_matmul(self.X, v))
            # 二次项 n * factor_dim-n * factor_dim , 再按factor_dim求和, n * output_dim
            p = 0.5 * tf.reshape(tf.reduce_sum(xv-tf.sparse_tensor_dense_matmul(X_square,tf.square(v)),1),[-1,output_dim])
            xw = tf.sparse_tensor_dense_matmul(self.X, w) # n * output_dim
            l = tf.reshape(xw + b + p,[-1]) # n
            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l,labels=self.y)) \
                        + l2_weight * tf.nn.l2_loss(xw) + l2_v * tf.nn.l2_loss(xv)
            self.optimizer = utils.get_optimizer(opt_algo,learning_rate,self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session = self.sess)

class FNN(Model):
    def __init__(self,field_sizes=None, embed_size=10, layer_size=None, layer_acts=None, drop_out=None,
                 embed_l2=None, layer_l2=None,init_path=None,opt_algo='gd',learning_rate=1e-2,random_seed=None):
        Model.__init__(self)
        init_vars = []
        num_input = len(field_sizes)
        # 定义神经网络的输入输出维度.......
        for i in range(num_input):
            init_vars.append(('embed_%d' % i,[field_sizes[i],embed_size],'xavier',dtype))
        node_in = num_input * embed_size
        for i in range(len(layer_size)):
            init_vars.append(('w%d' % i,[node_in,layer_size[i]],'xavier',dtype))
            init_vars.append(('b%d' % i,[layer_size[i]],'zero', dtype))
            node_in = layer_size[i]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_input)] # 每个field大小是 n * field_size
            self.y = tf.placeholder(dtype) # n
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)

            self.vars = utils.init_var_map(init_vars,init_path)
            w_0 = [self.vars['embed_%d' % i] for i in range(num_input)] # 每个field大小是 field_size * embed_size
            # n * (embed_size*num_field)
            xw = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i],w_0[i]) for i in range(num_input)],1)
            l = xw

            for i in range(len(layer_size)):
                w_i = self.vars['w%d' % i]
                b_i = self.vars['b%d' % i]
                print(l.shape, w_i.shape, b_i.shape)
                l = tf.nn.dropout(
                    utils.activate(tf.matmul(l,w_i)+b_i,layer_acts[i]),self.layer_keeps[i]) # w_0 : (n * （embed_size*num_field)) * ((embed_size*num_field)*layer_size[0]) => n*layer_size[0]
            l = tf.squeeze(l) # 数据降维，只裁剪等于1的维度
            self.y_prob = tf.sigmoid(l)
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l,labels=self.y))
            if layer_l2 is not None:
                self.loss += embed_l2 * tf.nn.l2_loss(xw)
                for i in range(len(layer_size)):
                    w_i = self.vars['w%d'%i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(w_i)
            self.optimizer = utils.get_optimizer(opt_algo,learning_rate,self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session = self.sess)


class IPNN(Model):
    def __init__(self,field_sizes=None, embed_size=10, layer_size=None, layer_acts=None, drop_out=None,
                 embed_l2=None, layer_l2=None,init_path=None,opt_algo='gd',learning_rate=1e-2,random_seed=None):
        Model.__init__(self)
        init_vars = []
        num_input = len(field_sizes) # field的个数
        for i in range(num_input):
            init_vars.append(('embed_%d' % i,[field_sizes[i],embed_size],'xavier',dtype))
        num_pairs = int(num_input * (num_input-1)/2) # field对的个数
        node_in = num_input * embed_size + num_pairs # 此处为设计的关键
        for i in range(len(layer_size)):
            init_vars.append(('w%d'%i,[node_in,layer_size[i]],'xavier',dtype))
            init_vars.append(('b%d'%i,[layer_size[i]],'zero',dtype))
            node_in = layer_size[i]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_input)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out) # 全1
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars,init_path)

            w_0 = [self.vars['embed_%d' % i] for i in range(num_input)] # 每个field大小是 field_size * embed_size
            # batch * (embed_size*num_field)
            xw = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i],w_0[i]) for i in range(num_input)],1)
            l = xw
            # batch * num_field * embed_size
            xw3d = tf.reshape(xw,[-1,num_input,embed_size])
            row = []
            col = []
            # 构造pair
            for i in range(num_input-1):
                for j in range(i+1,num_input):
                    row.append(i)
                    col.append(j)

            # batch * pair * embed_size
            p = tf.transpose(
                # pair * batch * embed_size
                tf.gather(
                    # num_field * batch * embed_size
                    tf.transpose(xw3d,[1,0,2]),row),[1,0,2])
            q = tf.transpose(
                # pair * batch * embed_size
                tf.gather(
                    # num_field * batch * embed_size
                    tf.transpose(xw3d,[1,0,2]),col),[1,0,2])
            p = tf.reshape(p, [-1,num_pairs,embed_size])
            q = tf.reshape(q, [-1,num_pairs,embed_size])
            ip = tf.reshape(tf.reduce_sum(p*q,[-1]),[-1,num_pairs]) # 按最后一维度求和
            # num_input * embed_size + num_pairs
            l = tf.concat([xw,ip,],1)

            for i in range(len(layer_size)):
                w_i = self.vars['w%d'%i]
                b_i = self.vars['b%d'%i]
                l = tf.nn.dropout(utils.activate(tf.matmul(l,w_i)+b_i,layer_acts[i]),self.layer_keeps[i])
            l = tf.squeeze(l)
            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l,labels=self.y))
            if layer_l2 is not None:
                self.loss += embed_l2 * tf.nn.l2_loss(xw)
                for i in range(len(layer_size)):
                    w_i = self.vars['w%d'%i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(w_i)
            self.optimizer = utils.get_optimizer(opt_algo,learning_rate,self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)


class OPNN(Model):
    def __init__(self,field_sizes=None, embed_size=10, layer_size=None, layer_acts=None, drop_out=None,
                 embed_l2=None, layer_l2=None,init_path=None,opt_algo='gd',learning_rate=1e-2,random_seed=None,
                 layer_norm=True):
        Model.__init__(self)
        init_vars = []
        num_input = len(field_sizes)
        for i in range(num_input):
            init_vars.append(('embed_%d' % i,[field_sizes[i],embed_size],'xavier',dtype))
        num_pairs = int(num_input * (num_input-1)/2) # field对的个数
        node_in = num_input * embed_size + num_pairs # 此处为设计的关键
        init_vars.append(('kernel',[embed_size,num_pairs,embed_size],'xavier',dtype))
        for i in range(len(layer_size)):
            init_vars.append(('w%d'%i,[node_in,layer_size[i]],'xavier',dtype))
            init_vars.append(('b%d'%i,[layer_size[i]],'zero',dtype))
            node_in = layer_size[i]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_input)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars,init_path)
            w_0 = [self.vars['embed_%d' %i] for i in range(num_input)]
            # batch * (embed_size*num_field)
            xw = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i],w_0[i]) for i in range(num_input)],1)
            xw3d = tf.reshape(xw,[-1,num_input,embed_size])

            row = []
            col = []
            # 构造pair
            for i in range(num_input-1):
                for j in range(i+1,num_input):
                    row.append(i)
                    col.append(j)

            # batch * pair * embed_size
            p = tf.transpose(
                # pair * batch * embed_size
                tf.gather(
                    # num_field * batch * embed_size
                    tf.transpose(xw3d,[1,0,2]),row),[1,0,2])
            q = tf.transpose(
                # pair * batch * embed_size
                tf.gather(
                    # num_field * batch * embed_size
                    tf.transpose(xw3d,[1,0,2]),col),[1,0,2])

            p = tf.reshape(p, [-1, num_pairs, embed_size])
            q = tf.reshape(q, [-1, num_pairs, embed_size])

            # embed_size*num_pairs*embed_size
            k = self.vars['kernel']

            p = tf.expand_dims(p,1) #  增加维度 batch * 1 *  pair * embed_size
            # batch * num_pairs
            # temp = tf.multiply(p,k)
            # temp = tf.reduce_sum(temp,-1)
            # temp = tf.transpose(temp,[0,2,1])
            # temp = tf.multiply(temp,q)
            # temp = tf.reduce_sum(temp,-1)
            kp = tf.reduce_sum(
                # batch * num_pairs * embed_size
                tf.multiply(
                    # 置换位置 batch * num_pairs * embed_size
                    tf.transpose(
                        # 按最后一个维度求和 batch * embed_size * num_pairs
                        tf.reduce_sum(
                            # 点乘 batch * embed_size*num_pairs*embed_size
                            tf.multiply(
                                p,k),
                            -1),
                        [0,2,1]),
                    q),
                -1)
            l = tf.concat([xw,kp],1)
            for i in range(len(layer_size)):
                w_i = self.vars['w%d'%i]
                b_i = self.vars['b%d'%i]
                l = tf.nn.dropout(utils.activate(tf.matmul(l,w_i)+b_i,layer_acts[i]),self.layer_keeps[i])
                l = tf.squeeze(l)
                self.y_prob = tf.sigmoid(l)

                self.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=l,labels=self.y))
                if layer_l2 is not None:
                    self.loss += embed_l2 * tf.nn.l2_loss(xw)
                    for i in range(len(layer_size)):
                        w_i = self.vars['w%d'%i]
                        self.loss += layer_l2[i] * tf.nn.l2_loss(w_i)
                self.optimizer = utils.get_optimizer(opt_algo,learning_rate,self.loss)

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(config = config)
                tf.global_variables_initializer().run(session=self.sess)

































































