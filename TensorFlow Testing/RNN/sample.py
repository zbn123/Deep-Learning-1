# -*- coding:utf-8 -*-
import tensorflow as tf
from data_utils import TextConverter, batch_generator
from model import CharRNN
import os
import time
# import codecs
# import IPython
# import embed

FLAGS = tf.flags.FLAGS
# flags是一个文件：flags.py，用于处理命令行参数的解析工作
tf.flags.DEFINE_integer('state_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('n_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_string('converter_path', '', 'model/name/converter.pkl')
tf.flags.DEFINE_string('checkpoint_path', '', 'checkpoint path')
tf.flags.DEFINE_string('start_string', '', 'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 30, 'max length to generate')


def main(_):
    FLAGS.start_string = FLAGS.start_string
    converter = TextConverter(filename=FLAGS.converter_path)
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path = \
            tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    model = CharRNN(converter.vocab_size, sampling=True,
                    state_size=FLAGS.state_size, n_layers=FLAGS.n_layers,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size)
    model.load(FLAGS.checkpoint_path)

    start = converter.text_to_data(FLAGS.start_string)
    data = model.sample(FLAGS.max_length, start, converter.vocab_size)
    # for c in converter.data_to_text(data):
    #     for d in c:
    #         # print(d,end="")
    #         time.sleep(0.5)
    print(converter.data_to_text(data))




if __name__ == "__main__":
    tf.app.run()
# python sample.py  --use_embedding --converter_path model/poetry/converter.pkl --checkpoint_path model/poetry/  --max_length 300

