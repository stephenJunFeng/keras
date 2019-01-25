"""
    image.reshape(60000,784) 多层感知器因为直接送进神经元处理,所以reshape转换为60000项,每项有784个数字,作为784个神经元的输入
    image.reshape(60000,28,28,1) 卷积神经网络因为必须先进行卷积与池化运算,所以必须保持图像的维数,所以reshape转换为60000项,每
    项是28*28*1的图像,分别是28(宽)*28(高)*1(单色)
"""
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

from com.stephen.mnist import show_train_history

np.random.seed(10)
#读取mnist数据
(x_Train,y_Train),(x_Test,y_Test) = mnist.load_data()
#将features(数字图像特征值)转换为四维矩阵(以reshape转换为6000*28*28*1的四维矩阵)
x_Train4D=x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')
x_Test4D=x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32')
#将features(数字图像特征值)标准化-->可以提高模型预测的准确度,并且更快收敛
x_Train4D_normalize = x_Train4D/255
x_Test4D_normalize = x_Test4D/255
#label(数字真实值)以One-Hot Encoding 进行转换
"""
    使用np_utils.to_categorical将训练数据与测试数据的label进行One-Hot Encoding(一位有效编码)转换
"""
y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot = np_utils.to_categorical(y_Test)

##建立模型
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

model = Sequential()

#建立卷积层1与池化层1
"""
    filters = 16 建立16个滤镜
    kernel_size=(5,5) 没一个滤镜5*5大小
    padding= 'same' 此设置让卷积运算产生的卷积图像大小不变
    input_shape = (28,28,1) 第一,二维:代表输入的图像形状为28*28 第三维: 因为是单色灰度图像,所以最后维数值是1
    activation = 'relu' 设置ReLU 激活函数
"""
model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',input_shape=(28,28,1),activation='relu'))

#建立池化层1
"""pool_size=(2,2) 执行第一次缩减采样,将16个28*28的图像缩小为16个14*14的图像"""
model.add(MaxPooling2D(pool_size=(2,2)))

#建立卷积层2
"""
    将原本16个图像转换为36个图像,卷积运算不改变图像大小
"""
model.add(Conv2D(filters=36,kernel_size=(5,5),padding='same',activation='relu'))
#建立池化层2
model.add(MaxPooling2D(pool_size=(2,2)))
#加入Dropout避免过度拟合
"""加入Dropout(0.25)加入模型中:每次训练迭代时,会随机在神经网络中放弃25%的神经元,以避免过度拟合"""
model.add(Dropout(0.25))
#建立平坦层
model.add(Flatten())
#建立隐藏层
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
#建立输出层
model.add(Dense(10,activation='softmax'))
#查看模型摘要
print(model.summary())

##进行训练
#定义训练方式 使用compile方法对训练模型进行设置
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#开始训练
"""
    x=x_Train4D_normalize(features数字图像的特征值)
    y=y_Train_OneHot(label数字图像真实的值)
    validation_split=0.2 设置训练与验证数据比例
    epochs=10 执行10个训练周期
    batch_size=300 每一批次300项数据
    verbose=2 显示训练过程
"""
train_history=model.fit(x=x_Train4D_normalize,y=y_TrainOneHot,validation_split=0.2,epochs=10,batch_size=300,verbose=2)
#显示结果
show_train_history('acc','val_acc')
show_train_history('loss','val_loss')

##评估模型准确率
scores = model.evaluate(x_Test4D_normalize,y_TrainOneHot)
print(scores[1])

#进行预测
prediction=model.predict_classes(x_Test4D_normalize)
print(prediction[:10])
import pandas as pd
pd.crosstab(y_Test,prediction,rownames=['label'],colnames=['predict'])
