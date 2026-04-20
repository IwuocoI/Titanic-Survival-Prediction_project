import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score#test
import sys
sys.path.append("..")
import config

data = pd.read_csv(config.CLEAN_TRAIN_DATA)
for i in data.columns:
     if i not in config.FEATURE_USED:
         data = data.drop(i, axis=1)
#划分特征和标签
x_train = data.drop("Survived", axis=1)
y_train = data["Survived"]
#模型定义
model = XGBClassifier(**config.XGBOOST_PARAMS)
#采用5折交叉验证
val_score=cross_val_score(model,x_train,y_train,cv=5)
print(val_score.mean())
#预测
model.fit(x_train,y_train)
test = pd.read_csv(config.CLEAN_TEST_DATA)
id = test["PassengerId"]
for i in test.columns:
    if i not in config.FEATURE_USED:
        test = test.drop(i, axis=1)

pred = pd.Series(model.predict(test), name="Survived")
result = pd.concat([id, pred], axis=1)
result.to_csv(config.XGBOOST_RESULT, index=False)
print("预测结果已存入结果文件夹")