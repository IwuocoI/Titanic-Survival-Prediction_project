import pandas as pd
from pandas import read_csv
import config

#第一轮数据处理，处理缺失值和删除无用特征
def data_clean_1(IN_PATH,OUT_PATH):
    data=pd.read_csv(IN_PATH)
    data=data.drop(columns=["PassengerId","Name","SibSp", "Parch", "Ticket", "Cabin"])
    data["Age"]=data["Age"].fillna(data["Age"].median())#中位数填充
    data["Embarked"]=data["Embarked"].fillna("S")#众数填充
    data=data[data.columns.tolist()[1:]+["Survived"]]#结果特征放到最后
    data.to_csv(OUT_PATH,index=False)#写入清洗后表格

data_clean_1(config.RAW_TRAIN_DATA,config.CLEAN_TRAIN_DATA)#第一轮数据处理

'''
你写data_clean_2前删掉这个注释，只是一个提示
后面测试集也用这两个函数处理，所以建议和我一样把读取路径和写入路径当作函数参数
路径直接用config代码里定义的路径，参考我的，用绝对路径在别的电脑上会报错
写完提交代码后改一下github自述文件，在里面写一下用的什么方式编码的
'''