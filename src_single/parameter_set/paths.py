# -*- coding:UTF-8 -*-
# 该部分主要存储文件路径，方便程序读取路径

# 图片存放位置（共约10000张图片）
DATA_DIR = '../datas/train'
# 生成tfrecord文件位置(训练集、测试集)
TF_Train_DIR = '../datas/train.tfrecords'
TF_Test_DIR = '../datas/test.tfrecords'

# 需要预测的图片存放位置
predict_path = '../datas/predict'

# 神经网络生成模型文件存放位置
MODEL_SAVE_PATH_network = "../models/network/"
# cnn生成模型文件存放位置
MODEL_SAVE_PATH_cnn = "../models/cnn/"