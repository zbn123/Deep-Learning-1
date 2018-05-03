# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import time
import os
from data_utils import TextConverter, batch_generator
from model import CharRNN
import os
import codecs

FLAGS = tf.flags.FLAGS
# tf.flags是一个文件：flags.py，用于处理命令行参数的解析工作
tf.flags.DEFINE_string('name', 'default', 'name of the model')
tf.flags.DEFINE_integer('n_seqs', 100, 'number of seqs in one batch')
tf.flags.DEFINE_integer('n_steps', 128, 'length of one seq')
tf.flags.DEFINE_integer('state_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('n_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.flags.DEFINE_string('input_file', '', 'utf-8 encode text file')
tf.flags.DEFINE_integer('max_steps', 100000, 'max steps to train')
tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps')
tf.flags.DEFINE_integer('log_every_n', 10, 'log to the screen every n steps')
tf.flags.DEFINE_integer('max_vocab', 3500, 'max char number')


def main(_):
    model_path = os.path.join('model', FLAGS.name)  # 拼接路径model/'name'
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
        text = f.read()  # 读取文本
    converter = TextConverter(text, FLAGS.max_vocab)  # 文本转换为词汇且截取FLAGS.max_vocab个词
    converter.save_to_file(os.path.join(model_path, 'converter.pk1'))  # 序列化存储词汇

    data = converter.text_to_data(text) # 将文本转化为输入(word_to_int)
    g = batch_generator(data, FLAGS.n_seqs, FLAGS.n_steps) # 获取batch生成器
    print(converter.vocab_size)
    # 模型参数初始化
    model = CharRNN(converter.vocab_size,
                    n_seqs=FLAGS.n_seqs,
                    n_steps=FLAGS.n_steps,
                    state_size=FLAGS.state_size,
                    n_layers=FLAGS.n_layers,
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size)
    model.train(g, FLAGS.max_steps, model_path, FLAGS.save_every_n, FLAGS.log_every_n)

if __name__ == "__main__":
    tf.app.run() # 解析命令行参数，调用main函数 main(_)命令行输入的参数
    print("training is ok!")
