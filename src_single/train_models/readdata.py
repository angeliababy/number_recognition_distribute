# -*- coding:UTF-8 -*-
# 训练、测试过程读取数据的操作过程

import tensorflow as tf

# 调用其他模块
import readdata
import sys
sys.path.append('../parameter_set')
import paths,hyperparameter

#图片信息
IMG_HEIGHT = hyperparameter.IMG_HEIGHT #图片高
IMG_WIDTH = hyperparameter.IMG_WIDTH #图片宽
IMG_CHANNELS = hyperparameter.IMG_CHANNELS #图片通道数
CLASS_NUM = hyperparameter.CLASS_NUM #类别数

# 之前生成的tfrecord文件的读取位置
TRAIN_FILE = paths.TF_Train_DIR
VALIDATION_FILE = paths.TF_Test_DIR

# 读取二进制数据
def read_and_decode(filename_queue): # 输入文件名队列
    reader = tf.TFRecordReader()  # create a reader from file queue
    _, serialized_example = reader.read(filename_queue) # reader从文件队列中读入一个序列化的样本
    # 解析符号化的样本
    features = tf.parse_single_example( # 解析 example
        serialized_example,
        features={ # 必须写明 features 里面的 key 的名称
        'image_raw': tf.FixedLenFeature([], tf.string), # 图片是string类型
        'label': tf.FixedLenFeature([], tf.string),
    })
    # 对于 BytesList，要重新进行解码，把 string 类型的 0 维 Tensor 变成 uint8 类型的一维 Tensor
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS]) # 改变数据形状
    image = tf.cast(image, tf.float32) * (1 / 255) #数据归一化
    label = tf.decode_raw(features['label'], tf.float64)
    label = tf.reshape(label, [CLASS_NUM]) # 改变数据形状
    label = tf.cast(label, tf.float32) # 改数据类型float32
    return image, label

# 获取批量数据及标签
def get_batch(train, batch_size,num_epochs):
    # 输入参数:
    # train: 是否为训练
    # batch_size: 训练的每一批有多少个样本
    # num_epochs: 过几遍数据，设置为 0/None 表示永远训练下去
    """
            注意 tf.train.QueueRunner 必须用 tf.train.start_queue_runners()来启动线程
    """
    if not num_epochs: num_epochs = None
    # 是训练过程还是测试过程并拿对应数据
    if train:
        filename = TRAIN_FILE
    else:
        filename = VALIDATION_FILE
    with tf.name_scope('input'):
        # num_epochs: 过几遍数据，设置为 0/None 表示永远训练下去
        filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
        # 读取数据及标签
        image, label = read_and_decode(filename_queue)
        # 随机化 example，并把它们规整成 batch_size 大小
        # tf.train.shuffle_batch 生成了 RandomShuffleQueue，并开启num_threads个线程
        images, sparse_labels = tf.train.shuffle_batch(
                                   [image, label], batch_size=batch_size, num_threads = 32,
                                   capacity=1000 + 3 * batch_size, #队列中的容量
                                   min_after_dequeue=1000) # 留下一部分队列，来保证每次有足够的数据做随机打乱
        return images, sparse_labels
