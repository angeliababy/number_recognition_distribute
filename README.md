## Tensorflow 集群
distributed tensorFlow一般将任务分为两类job：一类叫参数服务器，parameter server，简称为ps，用于存储tf.Variable；一类就是普通任务，称为worker，用于执行具体的计算。


**Tensorflow的分布式模型，分别是同步/异步更新，同步更新、异步更新有图间复制和图内复制。**
## 同步更新与异步更新
**同步随机梯度下降法(Sync-SGD，同步更新、同步训练)**
步骤 训练时，每个节点上工作任务读入共享参数，执行并行梯度计算，同步需要等待所有工作节点把局部梯度处好，将所有共享参数合并、累加，再一次性更新到模型参数，下一批次，所有工作节点用模型更新后参数训练。

**优势** 每个训练批次考虑所有工作节点训练情部，损失下降稳定。
**劣势** 性能瓶颈在最慢工作节点。异楹设备，工作节点性能不同，劣势明显

**异步随机梯度下降法(Async-SGD，异步更新、异步训练)**
步骤 每个工作节点任务独立计算局部梯度，异步更新到模型参数，不需执行协调、等待操作。

**优势** 性能不存在瓶颈。
**劣势** 每个工作节点计算梯度值发磅回参数服务器有参数更新冲突，影响算法收剑速度，损失下降过程抖动较大。

同步更新、异步更新实现区别于更新参数服务器参数策略。
数据量小，各节点计算能力较均衡，用同步模型。
数据量大，各机器计算性能参差不齐，用异步模式。


**图内模式与图间模式（数据并行）**
同步更新、异步更新有图内模式(in-graph pattern)和图间模式(between-graph pattern)，独立于图内(in-graph)、图间(between-graph)概念。

**图内复制(in-grasph replication)**
所有操作(operation)在同一个图中，用一个客户端来生成图，把所有操作分配到集群所有参数服务器和工作节点上。图内复制和单机多卡类似，扩展到多机多卡，数据分发还是在客户端一个节点上。

**优势**，计算节点只需要调用join()函数等待任务，客户端随时提交数据就可以训练。
**劣势**，训练数据分发在一个节点上，要分发给不同工作节点，严重影响并发训练速度。

**图间复制(between-graph replication)**
每一个工作节点创建一个图，训练参数保存在参数服务器，数据不分发，各个工作节点独立计算，计算完成把要更新参数告诉参数服务器，参数服务器更新参数。

**优势**，不需要数据分发，各个工作节点都创建图和读取数据训练。
**劣势**，工作节点既是图创建者又是计算任务执行者，某个工作节点宕机影响集群工作。大数据相关深度学习推荐使用图间模式。

## tensorflow分布式创建

 - 定义分布式参数

```
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
```

 - 创建server
 - 为ps角色添加等待函数，ps角色使用server.join()函数进行线程挂起，开始接受连接消息

```
ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')

    # 创建集群
    # num_worker = len(worker_spec)
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == 'ps':
        server.join()

    is_chief = (FLAGS.task_index == 0)
```

 - 创建网络结构，使用tf.device()函数将全部的节点都放在当前任务下
```
    with tf.device(tf.train.replica_device_setter(
            cluster=cluster)):

        # 生成训练数据含标签
        x, y_ = readdata.get_batch(train=True, batch_size=BATCH_SIZE, num_epochs=None)
        # 生成测试数据含标签
        text_x, text_y = readdata.get_batch(train=False, batch_size=BATCH_SIZE, num_epochs=50)       
```

 - 创建Supervisor，管理session
```
# 生成本地的参数初始化操作init_op
        init_op = tf.global_variables_initializer()
        train_dir = tempfile.mkdtemp()
        sv = tf.train.Supervisor(is_chief=is_chief, logdir=train_dir, init_op=init_op, recovery_wait_secs=1,
                                 global_step=global_step)

        if is_chief:
            print('Worker %d: Initailizing session...' % FLAGS.task_index)
        else:
            print('Worker %d: Waiting for session to be initaialized...' % FLAGS.task_index)
```

 - 训练
 ```
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
```

分布式对于单机训练只需改动train训练代码即可。

本项目完整源码地址：[https://github.com/angeliababy/number_recognition_distribute](https://github.com/angeliababy/number_recognition_distribute)
项目博客地址: [https://blog.csdn.net/qq_29153321/article/details/103969093](https://blog.csdn.net/qq_29153321/article/details/103969093)

参考博客：
[https://blog.csdn.net/u011026329/article/details/79190537](https://blog.csdn.net/u011026329/article/details/79190537)