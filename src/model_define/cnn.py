#coding=utf-8
# 本部分构造深度学习网络结构
# 采用cnn网络结构：包含两个卷积层、两个池化层、两个全连接层
# 卷积层、池化层作用：1.保持不变性：平移、旋转、尺度2.保留主要的特征同时减少参数达到降维作用
# 全连接作用：降维（很少使用升维），起到“分类器”的作用

# 网络的5种优化方案：激活函数，指数衰减的学习率，正则化损失，dropout,滑动平均模型（供了解）

import tensorflow as tf

# 调用其他模块 
import sys
sys.path.append('../parameter_set')
import hyperparameter

# 第一层卷积层通道数
NUM_CHANNELS = hyperparameter.NUM_CHANNELS
# 类别数
NUM_LABELS = hyperparameter.NUM_LABELS
# 第一层卷积层的尺寸和深度
CONV1_SIZE = hyperparameter.CONV1_SIZE
CONV1_DEEP = hyperparameter.CONV1_DEEP
# 第二卷积层的尺寸和深度
CONV2_SIZE =hyperparameter.CONV2_SIZE
CONV2_DEEP = hyperparameter.CONV2_DEEP
# 全连接层节点个数
FC_SIZE = hyperparameter.FC_SIZE
# 输出层
OUTPUT_NODE = hyperparameter.OUTPUT_NODE
# dropout层的系数
keep_prob_5 = hyperparameter.keep_prob

# 生成权重变量
def get_weight(shape):
    # "weights"是变量的名称，shape是变量的维度，initializer是变量初始化的方式，tf.truncated_normal_initializer生成截断的正态分布中的随机值，stddev为其标准偏差
    weights = tf.get_variable("weights",shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    return weights

# 对卷积神经网络进行前向计算
# 使用了RELU激活函数实现了去线性化
# 添加一个新的参数train,用于区分训练过程和测试过程，dropout仅在训练的时候使用，可以进一步提升模型的可靠性并防止过拟合
def inference(input_tensor,reuse,train,regularizer):
    # 定义第一层卷积层的变量并实现前向传播的过程。通过使用不同的命名空间来隔离不同层的变量，可以让每一层的变量命名只需要考虑在当前层的作用，而不需要担心重命名的问题
    with tf.variable_scope('layer1_conv1',reuse=reuse):
        conv1_weights = get_weight([CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP]) #尺寸、尺寸、通道、输出卷积深度
        conv1_biases = tf.get_variable("bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.0)) # 将偏置项b初始化为0
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME') # 卷积过程，strides=[1,1,1,1]表示滑动步长为1，padding='SAME'表示填0操作
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases)) # 激活过程（relu函数）,其中tf.nn.bias_add 是 tf.add （相加）的一个特例
    # 第二层池化层
    with tf.variable_scope('layer2_pool1',reuse=reuse):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') # 池化窗口大小为2*2，步长为2
    # 第三层卷积层
    with tf.variable_scope('layer3_conv2',reuse=reuse):
        conv2_weights = get_weight([CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP]) #尺寸、尺寸、通道、输出卷积深度
        conv2_biases = tf.get_variable('bias',[CONV2_DEEP],initializer=tf.constant_initializer(0.0)) # 将偏置项b初始化为0
        conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME') # 卷积过程，strides=[1,1,1,1]表示滑动步长为1，padding='SAME'表示填0操作
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases)) # 激活过程（relu函数）
    # 第四层池化层
    with tf.variable_scope('layer4_pool2',reuse=reuse):
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') # 池化窗口大小为2*2，步长为2
        
    # 将第四层池化层的输出转化为第五层全连接层的输入格式。第四层的输出为矩阵，然而第五层的输入格式为向量，
    # 因此需要将矩阵拉伸为向量。因为每一层神经网络的输入输出都为一个batch的矩阵，所以这里得到的维度也包含了一个batch中数据的个数
    # pool2形状转化为list
    pool_shape = pool2.get_shape().as_list()
    # pool_shape[0]是一个batch里数据的个数
    nodes = pool_shape[3]*pool_shape[1]*pool_shape[2]
    # 通过tf.reshape将第四层的输出变成一个batch向量
    reshaped = tf.reshape(pool2,[pool_shape[0],nodes])
    # 第五层全连接层
    # dropout一般在全连接层使用，防止过拟合
    with tf.variable_scope('layer5_fc1',reuse=reuse):
        fc1_weights=get_weight([nodes,FC_SIZE])  # 变量尺寸nodes*FC_SIZE
        #只有全连接层的权重需要加入正则化（暂时不用，为None）
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_bias = tf.get_variable('bias',[FC_SIZE],initializer=tf.constant_initializer(0.1)) # 将偏置项b初始化为0.1
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_bias) # 全连接（relu激活)
        # 如果是训练过程就要dropout,防止过拟合
        if train:
            fc1 = tf.nn.dropout(fc1,keep_prob_5)
    # 第六层全连接层
    with tf.variable_scope('layer6_fc2',reuse=reuse):
        fc2_weights=get_weight([FC_SIZE,NUM_LABELS]) # 变量尺寸FC_SIZE*NUM_LABELS
        # 只有全连接层的权重需要加入正则化（暂时不用，为None）
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_bias = tf.get_variable('bias',[NUM_LABELS],initializer=tf.constant_initializer(0.1)) # 将偏置项b初始化为0.1
        logit = tf.matmul(fc1,fc2_weights)+fc2_bias # 矩阵相乘加偏置项
    return logit
