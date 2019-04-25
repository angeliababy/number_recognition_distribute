#coding=utf-8
# 手写体识别（第三步）
# 预测部分(50张图片)

import tensorflow as tf
import os
import numpy as np
from scipy.misc import imread,imresize

# 调用其他模块
import sys
sys.path.append('../train_models')
import train_cnn as train
sys.path.append('../model_defin')
import cnn
sys.path.append('../parameter_set')
import paths,hyperparameter

#图片信息
IMG_HEIGHT = hyperparameter.IMG_HEIGHT #图片高
IMG_WIDTH = hyperparameter.IMG_WIDTH #图片宽
IMG_CHANNELS = hyperparameter.IMG_CHANNELS #图片通道数

# 预测图片位置
predict_path = paths.predict_path


# 预测部分
def evaluate(X_predict):
    with tf.Graph().as_default() as g:
        NUM = 1  # 预测图片数量
        # 网络输入、输出tensor
        x = tf.placeholder(tf.float32, shape=(NUM, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name='x_input')
        y = cnn.inference(x, False, False, regularizer=None)  # 预测结果

        # 实例化一个tf.train.Saver
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # 通过checkpoint文件找到模型文件名
            ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
            # ckpt.model_checkpoint_path:表示模型存储的位置
            if ckpt and ckpt.model_checkpoint_path:
                # 恢复模型
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 预测
                predict_y = sess.run(y, feed_dict={x: X_predict / 255.0})
                # 选出最大的那个
                predict_y = np.argmax(predict_y, axis=1)
                return predict_y
            else:
                print('No checkpoint file found')
                return

def main(argv=None):
    # 读取预测图片数据
    for filename in os.listdir(predict_path):
        filename = predict_path + '/' + filename
        print(filename)
        img = imread(filename)  # 读取图片数据
        img = imresize(img, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))  # 将图片大小统一转换
        X_predict = img.reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)  # 转化成模型输入形状
        predict_y = evaluate(X_predict)  # 结果预测,一张张预测
        print(predict_y)

    # 演示
    import matplotlib.pyplot as plt  # plt 用于显示图片
    import matplotlib.image as mpimg  # mpimg 用于读取图片
    # 在预测图片中选择一张
    print("请在../datas/predict/目录下选择一张图片路径,如..\datas\predict\9_1008.bmp：")
    path = input("input picture_path: ")
    img = mpimg.imread(path)  # 读取图片
    # 此时 img 就已经是一个 np.array 了，可以对它进行任意处理

    img = imresize(img, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))  # 将图片大小统一转换
    X_predict = img.reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS) # 转化成模型输入形状
    predict_y = evaluate(X_predict)  # 结果预测,一张张预测
    plt.title('predict: %d' % predict_y) # 标题
    plt.imshow(img, cmap='Greys_r') # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()

if __name__ == '__main__':
    tf.app.run()

