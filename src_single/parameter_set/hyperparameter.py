# -*- coding:UTF-8 -*-
# 该部分主要存储一些基本参数，包括图片信息、训练超参数等

# 图片信息（图片归一化之后的尺寸大小，本例中使用的图片已经是这个大小的了）
IMG_HEIGHT = 28 #图片高
IMG_WIDTH = 28 #图片宽
IMG_CHANNELS = 1 #图片通道数
#类别数(数字0-9)
CLASS_NUM = 10

# cnn网络结构定义
# 通道数
NUM_CHANNELS = IMG_CHANNELS
# 类别数
NUM_LABELS = CLASS_NUM
#第一层卷积层的尺寸和深度
CONV1_SIZE = 5
CONV1_DEEP = 32
#第二卷积层的尺寸和深度
CONV2_SIZE =5
CONV2_DEEP = 64
#第一层全连接层节点个数
FC_SIZE = 512
# 输出(等于类别数)
OUTPUT_NODE = CLASS_NUM
# droupout层系数
keep_prob = 0.5

#神经网络层数定义
# 输入神经元（图片数据转化为一维）
INPUT_NODE = IMG_HEIGHT*IMG_WIDTH
# 隐含层神经元
LAYER1_NODE = 32
# 隐含层神经元
LAYER2_NODE = 16
# 输出神经元
OUTPUT_NODE = CLASS_NUM

# 训练中的超参数
# 批次样本数
BATCH_SIZE = 128
#基础的学习率
LEARNING_RATE = 0.01
#训练轮数
TRAINING_STEPS = 60000


