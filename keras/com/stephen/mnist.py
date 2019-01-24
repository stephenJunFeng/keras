##读取MNIST数据集
#导入所需模块
from keras.utils import np_utils
import numpy as np

from com.stephen.mnistpre import plot_images_labels_prediction

np.random.seed(10)

#读取MNIST数据
from keras.datasets import mnist
(X_train_image,y_train_label),(X_test_image,y_test_label) = mnist.load_data()

#将features(数字图像特征值)使用reshape转换
"""将原本28*28的数字图像以reshape转换成784个Float数"""
x_Train = X_train_image.reshape(60000,784).astype('float32')
x_Test = X_test_image.reshape(10000,784).astype('float32')

#将features(数字图像特征值)标准化
"""将features(数字图像特征值)标准化可以提高模型预测的准确度,并且更快收敛."""
x_Train_normalize = x_Train/255
x_Test_normalize = x_Test/255

#label(数字真实的值)以One-Hot Encoding进行转换
"""使用np_utils.to_categorical将训练数据与测试数据的label进行One-Hot Encoding转换"""
y_Train_OneHot = np_utils.to_categorical(y_train_label)
y_Test_OneHot = np_utils.to_categorical(y_test_label)


##建立模型
#导入所需模块
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout #导入Dropout模块避免过度拟合

#建立Sequential模型
"""建立一个线性堆模型,后续只需要使用model.add()方法将各个神经网络层加入模型即可"""
model = Sequential()
#建立输入层与隐藏层
#输入层
"""
    units=256 定义"隐藏层"神经元个数为256
    input_dim=784 设置"输入层"神经元个数为784(因为原本28*28的二维图像,以reshape转换为一维的向量,也就是784个Float数)
    kernel_initializer='normal'使用normal distribution 正态分布的随机数来初始化weight(权重)和bias(偏差)
    activation 定义激活函数为relu
"""
model.add(Dense(units=1000,input_dim=784,kernel_initializer='normal',activation='relu'))

model.add(Dropout(0.5)) # 加入Dropout功能

#可设置两个隐藏层(可选)
model.add(Dense(units=1000,kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.5))

#输出层
"""
    units=10 定义"输出层"神经元个数为10
    kernel_initializer='normal' 使用normal distribution 正态分布的随机数来初始化weight 与bias
    activation 定义激活函数为softmax
"""
model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))
#查看模型的摘要
print(model.summary())

##进行训练
#定义训练方式
"""
    loss 设置损失函数,在深度学习中使用cross_entropy训练的效果比较好
    optimizer 设置训练时,在深度学习中使用adam优化器可以让训练更快收敛,并提高准确率
    metrics 设置评估模型的方式是准确率
"""
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#开始训练
"""
    model.fit进行训练,训练过程会存储在train_history变量中,需输入下列参数.
    输入训练数据参数
        x=x_Train_normalize(features数字图像的特征值)
        y=y_Train_Onehot(label数字图像真实的值)
    设置训练与验证数据比例
        validation_split=0.2
    设置epoch(训练周期)次数与每一批项数
        epochs=10 执行10个训练周期
        batch_size=200 每一批次200项数据
    设置显示训练过程
        verbose=2 显示训练过程
"""
train_history = model.fit(x=x_Train_normalize,y=y_Train_OneHot,validation_split=0.2,epochs=10,batch_size=200,verbose=2)

#建立训练过程 show_train_history
import matplotlib.pyplot as plt #导入matplotlib.pyplot模块,后续会使用plt来应用
"""
    定义 show_train_history 函数,输入参数: 之前训练过程所产生的train_history
    训练数据的执行结果
    验证数据的执行结果
"""
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train]) #训练数据
    plt.plot(train_history.history[validation]) #验证数据
    plt.title('Train History') #图的标题
    plt.ylabel(train)   #显示y轴的标签
    plt.xlabel('Epoch') #设置x轴标签是'Epoch'
    plt.legend(['train','validation'],loc='upper left') #设置图例是显示'train''validation',位置在左上角
    plt.show()
#执行定义的函数
show_train_history(train_history,'acc','val_acc') #画出准确率执行结果
show_train_history(train_history,'loss','val_loss') #画出误差执行结果

##以测试数据评估模型准确率
#评估模型准确率
"""
    scores = model.evaluate( 使用model.evaluate 评估模型的准确率,评估后的准确率会存储在scores中
    x= x_Test_normalize 测试数据的features(数字图像的特征值)
    y= y_Test_Onehot) 测试数据的label(数字图像真实的值)
"""
scores = model.evaluate(x_Test_normalize,y_Test_OneHot)
print('accuracy=',scores[1])

##进行预测
prediction = model.predict_classes(x_Test)  #使用model.predict_classes 输入参数x_Test(测试市局的数字图像)进行预测,结果存储在prediction中
print(prediction)
#使用之前定义的plot_images_labels_prediction 函数显示结果
"""
    x_test_image(测试数据图像)
    y_test_label(测试数据真实的值)
    prediction(预测结果)
    idx=340(显示第340到349共10项)
"""
plot_images_labels_prediction(X_test_image,y_test_label,prediction,idx=340)

##显示混淆矩阵
#使用pandas crosstab 建立混淆矩阵的功能
import pandas as pd
"""
    使用pd.crosstab建立混淆矩阵,输入参数:测试数据数字图像的真实值
    测试数据数字图像的预测结果
    设置行的名称是 label
    设置列的名称是 predict
"""
pd.crosstab(y_test_label,prediction,rownames=['label'],colnames=['predict'])
#建立真实值与预测DataFrame
df = pd.DataFrame({'label':y_test_label,'predict':prediction})
print(df[:2])
"""
    Pandas DataFrame 可以很方便地查询数据.
"""
print(df[(df.label==5)&(df.predict==3)]) #找出真实值=5 预测值=3的数据
#查看第340项结果,真实值是5但预测是3
plot_images_labels_prediction(X_test_image,y_test_label,prediction,idx=340,num=1)

##隐藏层增加为1000个神经元,修改了 输入层的神经元数 原本为256

##多层感知器加入DropOut功能以避免过度拟合

##建立多层感知器模型包含两个隐藏层

