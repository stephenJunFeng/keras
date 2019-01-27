##keras建立MLP,RNN,LSTM,模型进行IMDb情感分析
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from com.stephen.Imdb import read_files

#读取训练数据
y_train,train_text=read_files("train")
#读取测试数据
y_test,test_text=read_files("test")
#建立token
token = Tokenizer(num_words=3800)
token.fit_on_texts(train_text)
#将"影评文字"转换成"数字列表"
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)
#让所有"数字列表"的长度为100,后修改为380提高预测准确率
x_train = sequence.pad_sequences(x_train_seq,maxlen=380)
x_test = sequence.pad_sequences(x_test_seq,maxlen=380)

##加入潜入层
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
#from keras.layers.recurrent import SimpleRNN

model = Sequential()
"""
    output_dim=32       输出的维数为32,因为我们希望将"数字列表"转换为32维向量
    input_dim=2000      输入的维数是2000,因为之前建立的字典有2000个单词
    input_length=100    数字列表每一项有100个数字        
"""
model.add(Embedding(output_dim=32,input_dim=3800,input_length=380))
model.add(Dropout(0.2))

"""添加RNN模型 进行情感分析   建立16个神经元的RNN层"""
"""RNN只有短期记忆,没有长期记忆--使用LSTM模型,专门解决RNN的长期依赖问题"""
#model.add(SimpleRNN(units=16))  准确率: 0.82344
model.add(LSTM(32)) #准确率: 0.85436

#建立多层感知器模型
#model.add(Flatten())    #准确率: 0.83532
model.add(Dense(units=256,activation='relu'))   #隐藏层256个神经元,激活函数ReLU
model.add(Dropout(0.35))
"""
    输出层只有1个神经元,输出1代表正面评价,0代表负面评价
    激活函数Sigmoid
"""
model.add(Dense(units=1,activation='sigmoid'))
#查看模型摘要
model.summary()

##训练模型
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

train_history = model.fit(x_train,y_train,batch_size=100,epochs=10,verbose=2,validation_split=0.2)

##评估模型准确率
scores = model.evaluate(x_test,y_test,verbose=1)
print('准确率:',scores[1])

predict = model.predict_classes(x_test)
predict[:10]
predict_classes = predict.reshape(-1)
predict_classes
predict_classes[:10]

##查看测试数据预测结果
SentimentDict = {1:'正面的',0:'负面的'}
def display_test_Sentiment(i):
    print(test_text[i])
    print('label真实值',SentimentDict[y_test[i]],'预测结果:',SentimentDict[predict_classes[i]])

display_test_Sentiment(2)
display_test_Sentiment(12502)

##查看<美女与野兽>的影评

# input_text="""As a fan of the original cartoon, I also really enjoyed this remake. The songs are as wonderful as ever and the cgi effects really add to the film. The acting seemed very strong to me and the casting is pretty good. I'd definitely recommend seeing this. Will be a classic for future generations."""
# input_seq = token.texts_to_sequences([input_text])
# print(input_seq[0])
# print('before len:',len(input_seq[0]))
# pad_input_seq=sequence.pad_sequences(input_seq , maxlen=100)
# print('after len:',len(pad_input_seq[0]))
# predict_result=model.predict_classes(pad_input_seq)
# print('result:',predict_result)
# predict_result[0][0]
# print(SentimentDict[predict_result[0][0]])

def predict_review(input_text):
    input_seq = token.texts_to_sequences([input_text])
    pad_input_seq = sequence.pad_sequences(input_seq,maxlen=380)
    predict_result = model.predict_classes(pad_input_seq)
    print(SentimentDict[predict_result[0][0]])
print('-------------------------------测试---------------------------------')
predict_review('''
    Cast was created and worked hard. I seen the movie and enjoyed it. All I hear from ratings these days from people is hatred, spoiled brats, and bunch of stuck ups. If you don't like the movie and you have to bash people that still worked hard on set, then you should go act on the movie set and close your disgraceful mouths of outrageous childish behavior. They worked on hard on set, I like the movie, and the romance. Still stuff to be liked. You reviewers outta be ashamed of your selves with such childish review.
''')

predict_review('''
    Where do I start. This adaptation of Disney's 1991 Beauty and the Beast was an utter disappointment. Emma Watson as Belle was extremely unconvincing from the start to the end. She had the same expressions as the actress from Twilight. The animators did a terrible job with the Beast. He looked fake and lifeless. They could have used special makeup to create the beast similar to the Grinch where we get to see Jim Carrey's expressions. The side character animations were poorly executed. Overall I felt the film was rushed as there was lack of compassion and chemistry between the characters. There was a lot of CGI and green screen which could have been replaced by normal acting, because then why make an animated version of an animated film? This is by far the worst remake of an animated classic.
''')
