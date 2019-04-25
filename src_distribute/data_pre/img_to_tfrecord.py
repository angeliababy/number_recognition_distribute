# -*- coding:UTF-8 -*-
'''
# 手写体识别（第一步）
# 该部分是数据预处理过程，将原始图片转换为tensorflow特有的tfrecord文件格式
# 其中tfrecord是tensorflow官方推荐的标准格式，是一种比较通用、高效的数据读取方法，还有一种方法是可以python产生数据，再把数据feed_dict喂给后端。
'''

from __future__ import absolute_import,division,print_function

import numpy as np 
import tensorflow as tf
from scipy.misc import imread,imresize
from os import walk
from os.path import join
import numpy
import random
from sklearn.model_selection import train_test_split

# 调用其他模块
import sys
sys.path.append('../parameter_set')
import paths,hyperparameter

# 图片存放位置
DATA_DIR = paths.DATA_DIR
# 生成tfrecord文件位置(训练集、测试集总共约10000张图片)
TF_Train_DIR = paths.TF_Train_DIR # 训练tfrecord文件路径
TF_Test_DIR = paths.TF_Test_DIR # 测试tfrecord文件路径

# 图片信息
IMG_HEIGHT = hyperparameter.IMG_HEIGHT #图片高
IMG_WIDTH = hyperparameter.IMG_HEIGHT #图片宽
IMG_CHANNELS = hyperparameter.IMG_HEIGHT #图片通道数

# 类别数
CLASS_NUM = hyperparameter.CLASS_NUM

# 读取每张图片数据及标签设置one-hot形式
def read_images(path):
    lable_forders = next(walk(path))[1] # 获取一级目录（每个文件夹代表一个图像类别，目录名就代表图像的标签），如返回目录的list[0-9]
    images = []#图像数组信息
    labels = []#图像标签信息
    for lable_forder in lable_forders:  # 循环遍历每个文件夹（每个类别）
        #定义标签数组
        lable = numpy.asarray([0.0 for i in range(0,CLASS_NUM)], dtype='float64')
        lable[int(lable_forder)] = 1.0  # 依据目录打标签
        filenames = next(walk(path+"/"+lable_forder))[2] # 返回目录下各图片路径的list
        for filename in filenames:
            filename = path+"/"+lable_forder+"/"+filename # 图片文件路径
            img = imread(filename) # 读取文件数据
            img = imresize(img,(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)) #将图片大小统一转换
            images.append(img)
            labels.append(lable)
    return images,labels

# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 将所有图片转换为tfrecord格式
def convert(images,labels,filename):
    # 创建一个TFRecordWriter对象,这个对象负责写记录到指定的文件中
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(images)): #将每一张图片转换为tfrecord格式
        # TFRecord文件中的数据都是通过tf.train.Example Protocol Buffer的格式存储的，具体类细节较复杂
        # tf.train.Features初始化Features对象,一般是传入一个字典
        record = tf.train.Example(features=tf.train.Features(feature={
            'label': _bytes_feature(labels[i].tobytes()),
            'image_raw': _bytes_feature(images[i].tobytes())}))
        # 把字符串形式的记录写到文件中
        writer.write(record.SerializeToString())
        if i % 100 == 0:
            print("完成"+str(i)+"张图片的转换")
    writer.close()

def main(argv):
    # 读取每张图片数据及标签
    imgs, labs = read_images(DATA_DIR)
    # 随机划分测试集与训练集，测试集比率占15%
    train_x, test_x, train_y, test_y = train_test_split(imgs, labs, test_size=0.15, random_state=random.randint(0, 100))
    print("完成图片读取")
    # 生成训练集数据tfrecord文件并进行存储
    convert(train_x, train_y, TF_Train_DIR)
    # 生成测试集数据tfrecord文件并进行存储
    convert(test_x, test_y, TF_Test_DIR)

if __name__ == '__main__':
    tf.app.run()
