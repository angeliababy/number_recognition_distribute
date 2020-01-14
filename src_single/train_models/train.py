# -*- coding:UTF-8 -*-
'''
# 手写体识别（第二步）
# 模型训练部分
'''

import tensorflow as tf
import os
import time

# 调用其他模块
import sys
sys.path.append('../model_define')
# 选择（神经网络模型结构或者cnn模型）
import network
import cnn

# 调用其他模块
import readdata
import sys
sys.path.append('../parameter_set')
import hyperparameter

# 训练超参数
# 批次样本数
BATCH_SIZE = hyperparameter.BATCH_SIZE
# 基础的学习率
LEARNING_RATE = hyperparameter.LEARNING_RATE
# 训练轮数
TRAINING_STEPS = hyperparameter.TRAINING_STEPS


# 训练模型的过程
def train(networkmodel, MODEL_SAVE_PATH, MODEL_NAME):
    # with tf.device('/gpu:0'):
    with tf.device('/cpu:0'):
        train_start =time.time()
        # 生成训练数据含标签
        x, y_ = readdata.get_batch(train=True, batch_size=BATCH_SIZE, num_epochs=None)
        # 生成测试数据含标签
        text_x, text_y = readdata.get_batch(train=False, batch_size=BATCH_SIZE, num_epochs=None)

        # 神经网络模型
        if networkmodel:
            # 调整神经网络输入为一维，-1代表未知数量
            x = tf.reshape(x, [-1, x.shape[1] * x.shape[2] * x.shape[3]])
            # 训练部分输出
            y = network.inference(x, avg_class=None, reuse=False, lamada=None)
        else:
            # 卷积模型
            # 训练部分输入、输出tensor
            y = cnn.inference(x, False, False, regularizer=None)

        # 初始化，从0开始，每batch一次，增加1
        global_step = tf.Variable(0, trainable=False)

        # 神经网络模型
        if networkmodel:
            # 测试数据转化为一维，适应神经网络输入
            text_x = tf.reshape(text_x, [-1, text_x.shape[1] * text_x.shape[2] * text_x.shape[3]])
            # 测试输出
            average_y = network.inference(text_x, avg_class=None, reuse=True, lamada=None)
        else:
            # 卷积网络模型测试输入、输出
            average_y = cnn.inference(text_x, True, False, regularizer=None)

        # 对每个batch数据结果求均值，cross_entropy是一种信息熵方法，能够预测模型对真实概率分布估计的准确程度
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        # 求平均值
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        # 损失函数
        loss = cross_entropy_mean
        # 训练操作，GradientDescentOptimizer为梯度下降算法的优化器，学习率LEARNING_RATE，minimize为最小化损失函数操作
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)
        # 设计计算图
        with tf.control_dependencies([train_step]):
            train_op = tf.no_op(name='train')
        # 预测数字类别是否为正确类别，tf.argmax找出真实类别
        correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(text_y, 1))
        # tf.reduce_mean求平均值
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 初始化tf持久化类
        saver = tf.train.Saver()
        # 初始化会话，并开始训练
        with tf.Session() as sess:
            # 初始化模型的参数
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
            threads = tf.train.start_queue_runners(sess=sess, coord=coord) # 启动所有队列线程

            # 迭代的训练神经网络
            for i in range(TRAINING_STEPS):
                start_time = time.time()
                _, loss_value, step = sess.run([train_op, loss, global_step]) # 设置好整个图后，启动计算
                end_time = time.time()
                print('Training elapsed each step time:%f s' % (end_time - start_time))
                # 打印训练损失
                if (i + 1) % 10 == 0:
                    print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                # 打印验证准确率
                if (i + 1) % 100 == 0:
                    validate_acc = sess.run(accuracy) # 设置好整个图后，启动计算accuracy
                    print("After %d training step(s),validation accuracy using average model is %g." % (step, validate_acc))
                    a=os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step) # 保存模型
            train_end = time.time()
            print('Training elapsed total time:%f s' % (train_end - train_start))

            coord.request_stop() # 要求所有线程停止
            coord.join(threads)