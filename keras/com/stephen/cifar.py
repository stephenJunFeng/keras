from keras.datasets import cifar10 #从keras.datasets导入CIFAR-10数据集
import numpy as np #导入NumPy模块,NumPy是python语言的拓展程序库,支持维度数组与矩阵运算
np.random.seed(10) # 设置seed以生成需要的随机数
(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data() # 下载或读取CIFAR-10

#查看CIFAR-10数据
print('train: ',len(x_img_train))
print('test: ',len(x_img_test))

print(x_img_train.shape)
#查看第0项images图像的内容
print(x_img_test[0])

#定义label_dict字典
label_dict={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}

import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,labels,prediction,idx,num=10) :
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25 : num=25
    for i in range(0,num):
        ax = plt.subplot(5,5,1+i)
        ax.imshow(images[idx],cmap='binary')
        title=str(i)+','+label_dict[labels[i][0]]
        if len(prediction)>0:
            title+='=>'+label_dict[prediction[i]]
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()

plot_images_labels_prediction(x_img_train,y_label_train,[],0)

##将images进行预处理

#查看训练数据第1个图像的第一个点
print(x_img_train[0][0][0])

#将照片图像image的数字标准化
x_img_train_normalize = x_img_train.astype('float32')/255.0
x_img_test_normalize = x_img_test.astype('float32')/255.0

#查看照片图像images的数字标准化后的结果
print(x_img_train_normalize[0][0][0])

##对label进行数据预处理
#查看label原来的形状
print(y_label_train.shape)

#查看前5项数据,图像的分类
y_label_train[:5]

#将label标签字段转换为一位有效编码(One-Hot Encoding)
from keras.utils import np_utils
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)

#One-Hot Encoding 转换之后的label标签字段
y_label_train_OneHot.shape

#查看转换为One-Hot Encoding之后的结果
y_label_train_OneHot[:5]
