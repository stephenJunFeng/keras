from keras.datasets import cifar10
import numpy as np

from com.stephen.cifar import plot_images_labels_prediction, label_dict

np.random.seed(10)
#读取CIFAR-10数据
(x_img_train,y_label_train),(x_img_test,y_label_test) = cifar10.load_data()
#显示训练与验证数据的shape
print('train data:','images:',x_img_train.shape,'label:',y_label_train.shape)
print('test data:','images:',x_img_test.shape,'label:',y_label_test.shape)

#将features标准化
x_img_train_normalize = x_img_train.astype('float32')/255.0
x_img_test_normalize = x_img_test.astype('float32')/255.0

#label以一位有效编码进行转换
from keras.utils import np_utils
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)

#建立模型
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D

model = Sequential()
#建立卷积层1与池化层1
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

#建立卷积层2与池化层2
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

##建立3次卷积运算神经网络
model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

#建立平坦层
model.add(Flatten())
model.add(Dropout(0.3))
#建立隐藏层1
model.add(Dense(2500,activation='relu'))
model.add(Dropout(0.3))
#建立隐藏层2
model.add(Dense(1500,activation='relu'))
model.add(Dropout(0.3))
#建立输出层
model.add(Dense(10,activation='softmax'))
#查看模型摘要
print(model.summary())


##进行训练
#定义训练方式
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#训练之前加载模型权重
try:
    model.load_weights("SaveModel/cifarCnnModel.h5")
    print("加载模型成功!继续训练模型")
except:
    print("加载模型失败!开始训练一个新模型")

#开始训练
train_history = model.fit(x_img_train_normalize,y_label_train_OneHot,validation_split=0.2,epochs=5,batch_size=128,verbose=1)



import matplotlib.pyplot as plt #导入matplotlib.pyplot模块,后续会使用plt来应用
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train]) #训练数据
    plt.plot(train_history.history[validation]) #验证数据
    plt.title('Train History') #图的标题
    plt.ylabel(train)   #显示y轴的标签
    plt.xlabel('Epoch') #设置x轴标签是'Epoch'
    plt.legend(['train','validation'],loc='upper left') #设置图例是显示'train''validation',位置在左上角
    plt.show()


show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')

##评估模型准确率
scores = model.evaluate(x_img_test_normalize,y_label_test_OneHot,verbose=0)
scores[1]

##进行预测
prediction = model.predict_classes(x_img_test_normalize)
#预测结果
print(prediction[:10])

plot_images_labels_prediction(x_img_test,y_label_test,prediction,0,10)

##查看预测概率
Predicted_Probability = model.predict(x_img_test_normalize)

##建立show_Predicted_Probability函数
"""
    y(真实值)
    x_img(预测的图像)
    prediction(预测结果)
    Predicted_Probability(预测概率)
    i(开始显示的数据index)
"""
import matplotlib.pyplot as plt
def show_Predicted_Probability(y,prediction,x_img,Predicted_Probability,i) :
    print('label:',label_dict[y[i][0]],'predict',label_dict[prediction[i]])
    plt.figure(figsize=(2,2))
    plt.imshow(np.reshape(x_img_test[i],(32,32,3)))
    plt.show()
    for j in range(10):
        print(label_dict[j]+'Probability:%1.9f'%(Predicted_Probability[i][j]))


show_Predicted_Probability(y_label_test,prediction,x_img_test,Predicted_Probability,0)
show_Predicted_Probability(y_label_test,prediction,x_img_test,Predicted_Probability,3)

##显示混淆矩阵
#查看预测结果的形状
print(prediction.shape)

##模型的保存与加载
model.save_weights("SaveModel/cifarCnnModel.h5")
print("Save model to disk")


#查看y_label_test真实值的shape形状
print(y_label_test.shape)
#将y_label_test真实值转换为一维数组
y_label_test.reshape(-1)
#使用pandas crosstab建立混淆矩阵
import pandas as pd
print(label_dict)
pd.crosstab(y_label_test.reshape(-1),prediction,rownames=['label'],colnames=['predict'])

