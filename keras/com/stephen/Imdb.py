import urllib.request
import os
import tarfile  #用于解压文件

url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath="C:/Users/Administrator/.keras/data/aclImdb_v1.tar.gz"
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url,filepath)
    print('downloaded:',result)
#解压文件
if not os.path.exists("C:/Users/Administrator/.keras/data/aclImdb"):
    tfile = tarfile.open(filepath,'r:gz')
    result = tfile.extractall('C:/Users/Administrator/.keras/data/')

##读取IMDb数据
from keras.preprocessing import sequence  #导入sequence模块,将用于截长补短让所有"数字列表"长度为100
from keras.preprocessing.text import Tokenizer  #导入Tokenizer模块,将用于建立字典
import  re
#创建rm_tag函数删除文字中的HTML标签
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('',text)

#创建read_files函数读取IMDb文件
def read_files(filetype):
    path='C:/Users/Administrator/.keras/data/aclImdb/'
    file_list=[]

    positive_path=path+filetype+"/pos/"
    for f in os.listdir(positive_path):
        file_list+=[positive_path+f]

    negative_path=path+filetype+"/neg/"
    for f in os.listdir(negative_path):
        file_list+=[negative_path+f]

    print('read',filetype,'files:',len(file_list))

    all_labels = ([1]*12500+[0]*12500)
    all_texts = []
    for fi in file_list:
        with open(fi,encoding='utf8') as file_input:    #fi读取file_list所有文件使用open(fi,encoding='utf8')打开文件file_input使用file_input.readlies读取文件,并用join链接所有文件内容
            all_texts += [rm_tags(" ".join(file_input.readlines()))]

    return all_labels,all_texts

y_train,train_text=read_files("train")
y_test,test_text=read_files("test")
#查看第0项"影评文字"
train_text[0]
#查看第0项label是1,也就是正面评价
y_train[0]
#查看第12501项影评文字
train_text[12501]
#查看第12501项label是0,也就是负面评价
y_train[12501]

##建立token
token = Tokenizer(num_words=2000)   #使用Tokenizer建立token,输入参数num_words=2000,也就是说我们建立一个有2000个单词的字典
token.fit_on_texts(train_text)      #读取所有的训练数据影评,按照每一个英文单词在影评中出现的次数进行排序,排序的前2000名英文单词列入字典中
#查看token读取多少文章
print(token.document_count)
#查看token.word_index属性
print(token.word_index)

##使用token将"影评文字"转换成"数字列表"
#使用token.texts_to_sequences将"影评文字"转换为"数字列表"
x_train_seq=token.texts_to_sequences(train_text)
x_test_seq=token.texts_to_sequences(test_text)
print(train_text[0])
print(x_train_seq[0])

##让转换后的数字长度相同
#使用sequence.pad_sequences()截长补短
x_train = sequence.pad_sequences(x_train_seq,maxlen=100)
x_test = sequence.pad_sequences(x_test_seq,maxlen=100)
#查看第0项"数字列表"
print('before pad_sequence length=',len(x_train_seq[0]))
print(x_train_seq[0])
#显示第0项"数字列表",经过pad_sequences处理后的内容
print('after pad_sequences length=',len(x_train[0]))
print(x_train[0])
#"影评文字"转换"数字列表"后,长度59,小于100,处理后补全41个0,保证长度为100
# print('before pad_sequences length=',len(x_train_seq[1]))
# print(x_train_seq[1])
# print('after pad_sequences length=',len(x_train[1]))
# print(x_train[1])