import pandas as pd
from pandas import read_csv
import config

#第一轮数据处理，处理缺失值和删除无用特征
def data_clean_1(IN_PATH,OUT_PATH):
    data=pd.read_csv(IN_PATH)
    data=data.drop(columns=["Name","SibSp", "Parch", "Ticket", "Cabin"])
    data["Age"]=data["Age"].fillna(data["Age"].median())#中位数填充
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())#验证集fare缺了一个
    data["Embarked"]=data["Embarked"].fillna("S")#众数填充
    data.to_csv(OUT_PATH,index=False)#写入清洗后表格

def data_clean_2(IN_PATH, OUT_PATH):

    data = pd.read_csv(IN_PATH)#读取第一轮处理后的数据集
    
    data["Sex"] = data["Sex"].map({"female": 0, "male": 1})#对性别进行编码，female-0，male-1
    embarked_dummies = pd.get_dummies(data["Embarked"], prefix="Embarked")#生成Embarked的独热编码列，前缀为Embarked（列名变成Embarked_S/Embarked_C/Embarked_Q）
    data = pd.concat([data, embarked_dummies], axis=1)#将独热编码横向拼接到原数据集
    data = data.drop(columns=["Embarked"])#删除原始的Embarked列
    if "Survived" in data.columns:
        data = data[[data.columns[0]] + data.columns.tolist()[2:] + ["Survived"]]  # 结果特征放到最后
    
    data.to_csv(OUT_PATH, index=False)#写入编码后数据集

#第一轮数据处理
data_clean_1(config.RAW_TRAIN_DATA,config.CLEAN_TRAIN_DATA)
data_clean_1(config.RAW_TEST_DATA,config.CLEAN_TEST_DATA)

#第二轮数据处理
data_clean_2(config.CLEAN_TRAIN_DATA, config.CLEAN_TRAIN_DATA)
data_clean_2(config.CLEAN_TEST_DATA, config.CLEAN_TEST_DATA)
