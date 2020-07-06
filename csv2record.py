#将fer2013.csv转为tfrecord格式
from tqdm import tqdm#进度条
from time import sleep#从 time 模块中引入sleep函数，使用sleep函数可以让程序休眠（推迟调用线程的运行）。
import numpy as np
import os
import csv#csv文件格式是一种通用的电子表格和数据库导入导出格式
import tensorflow as tf

channel = 1
default_height = 48
default_width = 48
data_folder_name = '..\\Facical-Expression-Recognition'
data_path_name = 'data'
cv_path_name = 'fer2013'
csv_file_name = 'fer2013.csv'
record_name_train = 'fer2013_train.tfrecord'
record_name_test = 'fer2013_test.tfrecord'
record_name_eval = 'fer2013_eval.tfrecord'
cv_path = os.path.join(data_folder_name, data_path_name, cv_path_name)
csv_path = os.path.join(cv_path, csv_file_name)
record_path_train = os.path.join(data_folder_name, data_path_name, record_name_train)
record_path_test = os.path.join(data_folder_name, data_path_name, record_name_test)
record_path_eval = os.path.join(data_folder_name, data_path_name, record_name_eval)


with open(csv_path, 'r') as f:
    csvr = csv.reader(f)
    header = next(csvr)
    rows = [row for row in csvr]
    trn = [row[:-1] for row in rows if row[-1] == 'Training'] # row[:-1]：取出除了最后一列之外的所有列，row[-1]：取出最后一列；
    val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
    tst = [row[:-1] for row in rows if row[-1] == 'PrivateTest']


def write_binary(record_name_, labels_images_, height_=default_height, width_=default_width):
    writer_ = tf.python_io.TFRecordWriter(record_name_)#tfrecord生成器
    for label_image_ in tqdm(labels_images_):
        label_ = int(label_image_[0])
        image_ = np.asarray([int(p) for p in label_image_[-1].split()])# np.array 默认情况下将会copy该对象，而 np.asarray 除非必要，否则不会copy该对象。

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image/label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label_])),
                    "image/height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height_])),
                    "image/width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width_])),
                    "image/raw": tf.train.Feature(int64_list=tf.train.Int64List(value=image_))
                }
            )
        )
        writer_.write(example.SerializeToString())
    writer_.close()


write_binary(record_path_train, trn)
write_binary(record_path_test, tst)
write_binary(record_path_eval, val)
