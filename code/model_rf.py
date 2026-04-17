import pandas as pd
import config
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv(config.CLEAN_TRAIN_DATA)
for i in data.columns:#取用需要的特征
    if i not in config.FEATURE_USED:
        data=data.drop(i,axis=1)

#划分特征和标签
x_train=data.drop("Survived",axis=1)
y_train=data["Survived"]

#模型定义
rf=RandomForestClassifier(
    n_estimators=config.TREE_NUMBER,
    max_depth=config.MAX_DEPTH,
    min_samples_split=config.SPLIT_SAMPLES,
    min_samples_leaf=config.LEAF_SAMPLES,
    max_features=config.MAX_FEATURES,
    oob_score=True,
    n_jobs=-1,
    random_state=config.RANDOM_SEED
)

#训练
rf.fit(x_train,y_train)

#打印准确率
print(f"oob验证准确率：{rf.oob_score_:.4f}")

#输出测试集结果
test=pd.read_csv(config.CLEAN_TEST_DATA)
id=test["PassengerId"]#保留id
for i in test.columns:#去除不要的特征
    if i not in config.FEATURE_USED:
        test=test.drop(i,axis=1)

#对测试集进行预测
predict=pd.Series(rf.predict(test),name="Survived")
result=pd.concat([id,predict],axis=1)
result.to_csv(config.RF_RESULT,index=False)#按提交格式拼接
print("预测结果已存入结果文件夹")


