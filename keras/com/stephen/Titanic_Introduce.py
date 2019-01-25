import urllib.request  #用于下载文件
import os  #用于确认文件是否存在

#下载泰坦尼克号的旅客数据集
url="http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls"
filepath="C:/Users/Administrator/.keras/data/titanic3.xls"
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url,filepath)
    print('downloaded:',result)

import numpy
import pandas as pd
#读取titanic3.xls
all_df = pd.read_excel(filepath)
#查看前两项数据
all_df[:2]

"""
    survival    是否生存    0 否,1 是
    pclass      舱等       1 头等 2 二等 3 三等
    name
    sex
    age
    sibsp       手足或这配偶也在船上数量
    parch       双亲或者子女也在船上数量
    ticket      船票号码
    fare        旅客费用
    cabin       舱位号码
    embarked    登船港口       C=Cherbourg,Q=queenstown,S=Southampton
"""
#以上字段中的ticket(船票号码),cabin(舱位号码),我们认为与要预测的结果survived(是否生存关系不大)
cols=['survived','name','pclass','sex','age','sibsp','parch','fare','embarked']
all_df[:2]

##使用pandas DataFrame 进行数据预处理
#将name字段删除
df = all_df.drop(['name'],axis=1)
#找出含有null值的字段
all_df.isnull().sum()
#将age字段为null值 替换成平均值
age_mean = df['age'].mean()
df['age'] = df['age'].fillna(age_mean)

#将fare字段为null的数据替换为平均值
fare_mean = df['fare'].mean()
df['fare'] = df['fare'].fillna(fare_mean)

#转换性别字段为0与1
df['sex'] = df['sex'].map({'female':0,'male':1}).astype(int)

#将embarked字段进行一位有效编码转换
x_OneHot_df = pd.get_dummies(data=df,columns=["embarked"])

print(x_OneHot_df[:2])

##将DataFrame转换为Array
ndarray = x_OneHot_df.values
#查看array的shape
ndarray.shape
#查看ndarray的前两项数据
ndarray[:2]

#提取features与label
Label=ndarray[:,0]
Features=ndarray[:,1:]

#查看前两项label与features特征字段
Label[:2]
Features[:2]

##将ndarray特征字段进行标准化
#导入sklearn的数据预处理模块
from sklearn import preprocessing
#建立MinMaxScaler 标准化刻度 minmax_scale
minmax_scale=preprocessing.MinMaxScaler(feature_range=(0,1))
#使用minmax_scale.fit_transform进行标准化
scaledFeatures=minmax_scale.fit_transform(Features)
#查看标准化后的特征字段前两项
scaledFeatures[:2]

##将数据分为训练数据与测试数据
msk=numpy.random.rand(len(all_df)) < 0.8
train_df=all_df[msk]
test_df=all_df[~msk]

##创建PreprocessData函数进行数据预处理
def PreprocessData(raw_df):
    df=raw_df.drop(['name'],axis=1)
    age_mean=df['age'].mean()
    df['age']=df['age'].fillna(age_mean)
    fare_mean=df['fare'].mean()
    df['sex']=df['sex'].map({'female':0,'male':1}).astype(int)
    x_OneHot_df=pd.get_dummies(data=df,columns=["embarked"])

    ndarray=x_OneHot_df.values
    Features=ndarray[:,1:]
    Label=ndarray[:,0]

    minmax_scale=preprocessing.MinMaxScaler(feature_range=(0,1))
    scaledFeatures=minmax_scale.fit_transform(Features)
    return scaledFeatures,Label

train_Features,train_Label=PreprocessData(train_df)
test_Features,test_Label=PreprocessData(test_df)