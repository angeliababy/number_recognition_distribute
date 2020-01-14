#coding=utf-8
# 本部分构造深度学习网络结构
# 采用神经网络结构：两层全连接、一层输出层

# 网络的5种优化方案：激活函数，多层隐层，指数衰减的学习率，正则化损失，滑动平均模型（供了解）

import tensorflow as tf

# 调用其他模块
import sys
sys.path.append('../parameter_set')
import hyperparameter

#神经网络层数定义
# 输入神经元
INPUT_NODE = hyperparameter.INPUT_NODE
# 中间神经元
LAYER1_NODE = hyperparameter.LAYER1_NODE
# 中间神经元
LAYER2_NODE = hyperparameter.LAYER2_NODE
# 输出神经元
OUTPUT_NODE = hyperparameter.OUTPUT_NODE

#生成权重变量，并加入L2正则化（防止过拟合）损失到losses集合里，此时不用，设置为None
def get_weight(shape,Lamada):
    # "weights"是变量的名称，shape是变量的维度，initializer是变量初始化的方式，tf.truncated_normal_initializer生成截断
    # 的正态分布中的随机值，mean为其均值，stddev为其标准偏差
    weights = tf.get_variable("weights",shape,initializer=tf.random_normal_initializer(mean=0,stddev=0.1))
    # 正则化None，不使用
    if Lamada!=None:
        tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(Lamada)(weights))
    return weights

# 对神经网络进行前向计算，有两个版本，包含滑动平均以及不包含滑动平均
# 使用了RELU激活函数实现了去线性化，函数支持传入计算参数平均的类，便于测试时使用滑动平均模型
def inference(input_tensor,avg_class,reuse,lamada):
    # 生成各层权重参数及偏置项，reuse为是否重新复用变量，一般测试时设置True，代表复用训练的变量
    with tf.variable_scope('layer1',reuse=reuse):
        weights1 = get_weight([INPUT_NODE,LAYER1_NODE],lamada) # 变量维度（INPUT_NODE,LAYER1_NODE）,lamada正则化
        biases1 = tf.get_variable("bias",[LAYER1_NODE],initializer=tf.random_normal_initializer(mean=0,stddev=0.1)) # 偏置项初始化为正态分布随机值，其均值为0，标准偏差为0.1
    with tf.variable_scope('layer2',reuse=reuse):
        weights2 = get_weight([LAYER1_NODE,LAYER2_NODE],lamada) # 变量维度（INPUT_NODE,LAYER2_NODE）,lamada正则化
        biases2 = tf.get_variable("bias",[LAYER2_NODE],initializer=tf.random_normal_initializer(mean=0,stddev=0.1)) # 偏置项初始化为正态分布随机值，其均值为0，标准偏差为0.1
    with tf.variable_scope('layer3',reuse=reuse):
        weights3 = get_weight([LAYER2_NODE,OUTPUT_NODE],lamada) # 变量维度（INPUT_NODE,LAYER2_NODE）,lamada正则化
        biases3 = tf.get_variable("bias",[OUTPUT_NODE],initializer=tf.random_normal_initializer(mean=0,stddev=0.1))  # 偏置项初始化为正态分布随机值，其均值为0，标准偏差为0.1
    if avg_class == None:
        # 第一层全连接，采用tanh进行非线性数据变换
        layer1 = tf.nn.tanh(tf.matmul(input_tensor,weights1)+biases1)
        # 第二层全连接，采用relu激活
        layer2 = tf.nn.relu(tf.matmul(layer1,weights2)+biases2)
        # 输出层
        output = tf.matmul(layer2,weights3)+biases3 # 矩阵相乘加偏置项
    return output
