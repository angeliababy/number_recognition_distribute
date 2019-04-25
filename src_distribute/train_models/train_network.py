# -*- coding:UTF-8 -*-
'''
# 手写体识别（第二步）
# 调用模型训练部分
# 神经网络
'''

import tensorflow as tf

# 调用其他模块
import train
import sys
sys.path.append('../parameter_set')
import paths

# 生成神经网络神经网络模型文件存放位置及名称
MODEL_SAVE_PATH = paths.MODEL_SAVE_PATH_network
MODEL_NAME = 'model'
                
def main(argv):
    # 进入模型，True为神经网络模型
    train.train(networkmodel = True, MODEL_SAVE_PATH = MODEL_SAVE_PATH, MODEL_NAME = MODEL_NAME)

if __name__ == '__main__':
    tf.app.run()
