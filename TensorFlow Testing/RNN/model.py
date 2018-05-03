# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import time
import os


class CharRNN(object):
    """docstring for ClassName"""

    def __init__(self, n_classes, n_seqs=64, n_steps=50,
                 state_size=128, n_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False, train_keep_prob=0.5,
                 use_embedding=False, embedding_size=128):

        if sampling is True:
            n_seqs, n_steps = 1, 1
        else:
            n_seqs, n_steps = n_seqs, n_steps

        self.n_classes = n_classes  # softmax的类别数=单词总量
        self.n_seqs = n_seqs # 序列数
        self.n_steps = n_steps # 时间步
        self.state_size = state_size # 状态向量维度
        self.n_layers = n_layers # 深度lstm层数
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob # dropout
        self.use_embedding = use_embedding # 是否使用embedding
        self.embedding_size = embedding_size # embedding向量的大小

        # 构建计算图
        tf.reset_default_graph()
        self.build_inputs()
        self.build_lstm()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=(self.n_seqs, self.n_steps), name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(self.n_seqs, self.n_steps), name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            # 对于中文，需要使用embedding层
            # 英文字母没必要使用embedding层
            if self.use_embedding is False:
                self.lstm_inputs = tf.one_hot(self.inputs, self.n_classes)
            else:
                with tf.device('/cpu:0'):
                    embedding = tf.get_variable('embedding', [self.n_classes, self.embedding_size])
                    self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)

    def build_lstm(self):
        # 创建单个cell并堆叠多层
        def get_a_cell(state_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(state_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.state_size, self.keep_prob) for _ in range(self.n_layers)]
            )
            self.initial_state = cell.zero_state(self.n_seqs, tf.float32)

            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs, initial_state=self.initial_state)

            seq_output = tf.concat(self.lstm_outputs, 1) # 拼接子序列输出成一个大序列
            x = tf.reshape(seq_output, [-1, self.state_size]) # 每个时间步输出都是state_size

            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([self.state_size, self.n_classes], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(self.n_classes))

            self.logits = tf.matmul(x, softmax_w) + softmax_b
            self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')

    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.n_classes)
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)

    def build_optimizer(self):
        tvars = tf.trainable_variables()
        # 让权重更新在一定的范围内
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars)) # 计算梯度并应用到变量上，是函数minimize()的第二部分，返回一个应用指定的梯度的操作Operation，对global_step做自增操作

    def train(self, batch_generator, max_steps, save_path, save_every_n, log_every_n):
        self.session = tf.Session() # 获取会话，Session对象在运行时负责对数据流图进行监督，并且是运行数据流图的主要接口
        with self.session as sess:
            sess.run(tf.global_variables_initializer()) # 所有变量初始化
            # Train network
            step = 0
            new_state = sess.run(self.initial_state) # 获取LSTM初始化的状态
            for x, y in batch_generator:
                step += 1
                start = time.time()
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([self.loss,
                                                     self.final_state,
                                                     self.optimizer],
                                                    feed_dict=feed)

                end = time.time()
                # control the print lines
                if step % log_every_n == 0:
                    print('step: {}/{}... '.format(step, max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                if (step % save_every_n == 0):
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                if step >= max_steps:
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)

    def sample(self, n_samples, prime, vocab_size):
        samples = [c for c in prime]
        sess = self.session
        new_state = sess.run(self.initial_state)
        preds = np.ones((vocab_size,))
        for c in prime:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state], feed_dict=feed)
        c = pick_top_n(preds, vocab_size)
        # 添加字段到samples中
        samples.append(c)

        # 不断生成字符，直到达到指定数目
        for i in range(n_samples):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state], feed_dict=feed)
            c = pick_top_n(preds, vocab_size)
            # 添加字段到samples中
            samples.append(c)

        return np.array(samples)

    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restore from: {}'.format(checkpoint))


def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds) # 把shape中为1的维度去掉
    # 将除了top_n个预测值之外的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0 # 从大到小索引值
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符
    c = np.random.choice(vocab_size, 1, p=p)[0]
    while c == vocab_size-1:
        c = np.random.choice(vocab_size, 1, p=p)[0]
    return c
