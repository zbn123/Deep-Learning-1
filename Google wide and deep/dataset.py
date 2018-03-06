#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'CLH'

'''
    下载和预处理数据
'''
import urllib.request
import argparse
import tensorflow as tf
import sys
import os

'''
    注：training_file 和 evaluation_file要与下载的文件名相同
'''
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",type=str,default='./tmp/census_data',help='Directory to download census data')
parser.add_argument("--training_file",type=str,default='adult.data',help='the name of training file ')
parser.add_argument("--evaluation_file",type=str,default='adult.test',help='the name of evaluation file ')

class Dataset(object):
    def __init__(self,FLAGS):
        self.FLAGS = FLAGS

    def download_and_clean_data(self,filename, url):
        '''
        从url上下载数据，并处理成CSV格式
        :return:
        '''
        temp_file,_ = urllib.request.urlretrieve(url)
        with tf.gfile.Open(temp_file,"r") as temp_eval_file:
            with tf.gfile.Open(filename,"w") as eval_file:
                for line in temp_eval_file:
                    line = line.strip().replace(', ',',')
                    if not line or ',' not in line:
                        continue
                    if line[-1] == '.':
                        line = line[:-1]
                    line += "\n"
                    eval_file.write(line)
        tf.gfile.Remove(temp_file)


def main(unused_argv):
    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MkDir(FLAGS.data_dir)
    global DATA_URL
    training_url = '{}/{}'.format(DATA_URL,FLAGS.training_file)
    eval_url = '{}/{}'.format(DATA_URL,FLAGS.evaluation_file)
    training_file_path = os.path.join(FLAGS.data_dir,FLAGS.training_file)
    eval_file_path = os.path.join(FLAGS.data_dir,FLAGS.evaluation_file)
    data = Dataset(FLAGS)
    data.download_and_clean_data(training_file_path,training_url)
    data.download_and_clean_data(eval_file_path,eval_url)


if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)