from keras.datasets import cifar10 #从keras.datasets导入CIFAR-10数据集
import numpy as np #导入NumPy模块,NumPy是python语言的拓展程序库,支持维度数组与矩阵运算
np.random.seed(10) # 设置seed以生成需要的随机数
(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data() # 下载或读取CIFAR-10
