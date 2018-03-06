#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'CLH'

import argparse
import sys
import tensorflow as tf
import shutil

CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

CSV_COLUMNS_DEFAUTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                       [0], [0], [0], [''], ['']]

NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281
}

# 参数设置
def parse_args():
    parser = argparse.ArgumentParser(description="Run Model.")
    parser.add_argument('--model_dir', type=str, default='./tmp/census_model',
                        help='Base directory for model')
    parser.add_argument('--model_type', type=str, default='wide_deep',
                        help="Valid model types: {'wide', 'deep', 'wide_deep'}.")
    parser.add_argument('--train_epochs', type=int, default=40,
                        help='The number of training epochs')
    parser.add_argument('--epochs_per_eval',type=int, default=2,
                        help='The number of training epochs to run between evaluations.')
    parser.add_argument('--batch_size', type=int, default=40,
                        help='Number of examples per batch.')
    parser.add_argument('--train_data', type=str, default='./tmp/census_data/adult.data',
                        help='Path to the training data.')
    parser.add_argument('--test_data', type=str, default='./tmp/census_data/adult.test',
                        help='Path to the test data.')
    return parser

def build_model_columns():
    '''
    构造特征,主要为将离散的、分类特征转换为one-hot以及构造交叉特征
    :return:
    '''
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education',['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
        '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
        'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
        'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship', [
        'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
        'Other-relative'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
         'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
         'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    occupation = tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=1000)

    age_buckets = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # 模型的列
    base_columns = [
        education, marital_status, relationship, workclass, occupation,
        age_buckets,
    ]

    # 特征十字架，即人工构造的组合特征
    crossed_columns = [
        tf.feature_column.crossed_column(
            ['education', 'occupation'], hash_bucket_size=1000),
        tf.feature_column.crossed_column(
            [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
    ]

    wide_columns = base_columns + crossed_columns

    deep_columns = [
        age,education_num,capital_gain,capital_loss,hours_per_week,
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),
        tf.feature_column.embedding_column(occupation,dimension=8)
    ]

    return wide_columns, deep_columns


def build_estimate(model_dir, model_type):
    '''
    构建评测模型
    :param model_dir:
    :param model_type:
    :return:
    '''
    wide_columns, deep_columns = build_model_columns()
    hidden_units = [100, 75, 50, 25]

    run_config = tf.estimator.RunConfig().replace(
        session_config = tf.ConfigProto(device_count={'GPU':0}))

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(model_dir=model_dir,
                                             feature_columns=wide_columns,
                                             config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(model_dir=model_dir,
                                          feature_columns=deep_columns,
                                          hidden_units=hidden_units,
                                          config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(model_dir=model_dir,
                                                        linear_feature_columns=wide_columns,
                                                        dnn_feature_columns=deep_columns,
                                                        dnn_hidden_units=hidden_units,
                                                        config=run_config)


def input_fn(data_file,num_epochs, shuffle, batch_size):
    '''

    :param data_file: 数据集
    :param num_epochs: 
    :param shuffle: 是否混洗数据
    :param batch_size:
    :return:
    '''
    assert tf.gfile.Exists(data_file),('%s not found.'% data_file)

    def parse_csv(value):
        print('parsing',data_file)
        columns = tf.decode_csv(value,record_defaults=CSV_COLUMNS_DEFAUTS)
        features = dict(zip(CSV_COLUMNS,columns))
        labels = features.pop('income_bracket')
        return features,tf.equal(labels,'>50k')

    # 抽取数据
    dataset = tf.data.TextLineDataset(data_file)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=NUM_EXAMPLES['train'])
    dataset = dataset.map(parse_csv,num_parallel_calls=5) # 使用map对dataset中的每个元素进行处理
    # dataset,_ = dataset
    dataset = dataset.repeat(num_epochs) # 将dataset重复一定数目的次数用于多个epoch的训练
    dataset = dataset.batch(batch_size) # 将原来的dataset中的元素按照某个数量叠在一起，生成mini batch

    return dataset


def main(unused_argv):
    # 清空模型存储路径下的文件
    shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
    model = build_estimate(FLAGS.model_dir, FLAGS.model_type)

    # 训练模型
    for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        model.train(input_fn = lambda : input_fn(FLAGS.train_data,FLAGS.epochs_per_eval,True,FLAGS.batch_size))

        results = model.evaluate(input_fn = lambda : input_fn(FLAGS.test_data,1,False,FLAGS.batch_size))

        # 显示训练过程
        print('result at epoch',(n+1)*FLAGS.epochs_per_eval)
        print('-'*60)

        for key in sorted(results):
            print('{}:{}'.format(key,results[key]))




if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parse_args().parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
