# -*- coding:UTF-8 -*-
'''
# 手写体识别（第二步）
# 模型训练部分
'''

import tensorflow as tf
from datetime import datetime
import os
import time
import tempfile

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

flags = tf.app.flags
# 定义分布式参数
# 参数服务器parameter server节点
flags.DEFINE_string('ps_hosts', None, 'Comma-separated list of hostname:port pairs')
# 两个worker节点
flags.DEFINE_string('worker_hosts', None,
                    'Comma-separated list of hostname:port pairs')
# 设置job name参数
flags.DEFINE_string('job_name', None, 'job name: worker or ps')
# 设置任务的索引
flags.DEFINE_integer('task_index', None, 'Index of task within the job')
# 选择异步并行，同步并行
flags.DEFINE_integer("issync", None, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")

FLAGS = flags.FLAGS


# 训练模型的过程
def train(networkmodel, MODEL_SAVE_PATH, MODEL_NAME):

    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print('job_name : %s' % FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print('task_index : %d' % FLAGS.task_index)

    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')

    # 创建集群
    # num_worker = len(worker_spec)
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == 'ps':
        server.join()

    is_chief = (FLAGS.task_index == 0)
    # worker_device = '/job:worker/task%d/cpu:0' % FLAGS.task_index
    with tf.device(tf.train.replica_device_setter(
            cluster=cluster)):

        # 生成训练数据含标签
        x, y_ = readdata.get_batch(train=True, batch_size=BATCH_SIZE, num_epochs=None)
        # 生成测试数据含标签
        text_x, text_y = readdata.get_batch(train=False, batch_size=BATCH_SIZE, num_epochs=50)

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

        # 初始化，从0开始，每batch一次，增加1,创建纪录全局训练步数变量
        global_step = tf.Variable(0, name='global_step', trainable=False)

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
        # 求损失函数
        loss = tf.reduce_mean(cross_entropy)
        # 训练操作，GradientDescentOptimizer为梯度下降算法的优化器，学习率LEARNING_RATE，minimize为最小化损失函数操作
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)
        # 预测数字类别是否为正确类别，tf.argmax找出真实类别
        correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(text_y, 1))
        # tf.reduce_mean求平均值
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # # 设计计算图
        # with tf.control_dependencies([train_step]):
        #     train_op = tf.no_op(name='train')
        # 生成本地的参数初始化操作init_op
        init_op = tf.global_variables_initializer()
        train_dir = tempfile.mkdtemp()
        sv = tf.train.Supervisor(is_chief=is_chief, logdir=train_dir, init_op=init_op, recovery_wait_secs=1,
                                 global_step=global_step)

        if is_chief:
            print('Worker %d: Initailizing session...' % FLAGS.task_index)
        else:
            print('Worker %d: Waiting for session to be initaialized...' % FLAGS.task_index)
        sess = sv.prepare_or_wait_for_session(server.target)
        print('Worker %d: Session initialization  complete.' % FLAGS.task_index)

        time_begin = time.time()
        print('Traing begins @ %f' % time_begin)
        local_step = 0
        for i in range(TRAINING_STEPS):

            coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 启动所有队列线程

            _, step, loss_value = sess.run([train_step, global_step, loss])
            local_step += 1

            now = time.time()
            print('%f: Worker %d: traing step %d dome (global step:%d)' % (now, FLAGS.task_index, local_step, step))

            # 打印验证准确率
            if (i + 1) % 100 == 0:
                validate_acc = sess.run(accuracy) # 设置好整个图后，启动计算accuracy
                print("After %d training step(s),validation accuracy using average model is %g." % (step, validate_acc))

            coord.request_stop()  # 要求所有线程停止
            coord.join(threads)

        time_end = time.time()
        print('Training ends @ %f' % time_end)
        train_time = time_end - time_begin
        print('Training elapsed time:%f s' % train_time)

    sess.close()