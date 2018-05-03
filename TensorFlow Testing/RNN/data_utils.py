# -*- coding:utf-8 -*-
__author__ = 'CLH'

import tensorflow as tf
import copy
import numpy as np
import pickle
import win_unicode_console
win_unicode_console.enable()

def batch_generator(data, n_seqs, n_steps):
    '''
    batch生成器
    :param data: 语料库
    :param n_seqs: 一个长序列划分为n_seqs个子序列
    :param n_steps: 每个子序列包含的单词长度
    :return:
    '''
    data = copy.copy(data)
    # y_data = copy.copy(data)
    # y_data[:-1], y_data[-1] = y_data[1:], y_data[0]
    batch_size = n_seqs * n_steps
    n_batches = int(len(data) / batch_size)
    data = data[:batch_size * n_batches]
    data = data.reshape((n_seqs,-1))
    # y_data = y_data[:batch_size * n_batches]
    # y_data = y_data.reshape((n_seqs,-1))

    while True:
        # 生成一个batch的训练数据，n_seqs * n_steps
        np.random.shuffle(data)
        for n in range(0,data.shape[1],n_steps):
            x = data[:, n:n + n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            # print(n_steps, x.shape, y.shape)
            yield x, y

class TextConverter(object):
    def __init__(self,text=None, max_vocab=5000, filename=None):
        if filename is not None:
            with open(filename,'rb') as fin:
                self.vocab = pickle.load(fin)
        else:
            # 获取高频词汇
            vocab = set(text)
            print(len(vocab))
            vocab_count = {}
            for word in vocab: # 一个字为一个word
                # print(word)
                vocab_count[word] = 0
            for word in text:
                vocab_count[word] += 1
            vocab_count_list = []
            for word in vocab_count:
                vocab_count_list.append((word,vocab_count[word]))
            vocab_count_list.sort(key=lambda x:x[1],reverse=True)
            vocab_count_list = vocab_count_list[:min(len(vocab_count_list),max_vocab)] # 截取指定数量的词汇
            vocab = [x[0] for x in vocab_count_list] # 保留word，忽略count
            self.vocab = vocab
        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)} # 编码
        self.int_to_word_table = dict(enumerate(self.vocab)) # 编码

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>' # 把没有出现的的词标记为"<unk>"
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_data(self, text):
        data = []
        for word in text:
            data.append(self.word_to_int(word))
        return np.array(data)

    def data_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)