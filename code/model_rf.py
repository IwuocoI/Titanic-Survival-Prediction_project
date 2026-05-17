import pandas as pd
import sys

from seaborn.colors import xkcd_rgb

sys.path.append("..")#扩大运行环境以找到config
import config
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def model_rf(train_data,test_data,result_path,feature):
    data = pd.read_csv(train_data)
    for i in data.columns:  # 取用需要的特征
        if i not in feature:
            data = data.drop(i, axis=1)

    # 划分特征和标签
    x_train = data.drop("Survived", axis=1)
    y_train = data["Survived"]

    # 模型定义
    rf = RandomForestClassifier(**config.RF_PARAMS)

    #交叉验证
    cv_score=cross_val_score(rf,x_train,y_train,cv=5)#五折交叉验证
    cv_mean=cv_score.mean()#平均准确率
    cv_std=cv_score.std()#标准差
    print(f"随机森林模型:\n五折交叉验证平均准确率:{cv_mean:.4f}\n标准差:{cv_std:.4}")

    # 训练
    rf.fit(x_train, y_train)

    # 打印准确率
    print(f"oob验证准确率:{rf.oob_score_:.4f}")

    # 输出测试集结果
    test = pd.read_csv(test_data)
    id = test["PassengerId"]  # 保留id
    for i in test.columns:  # 去除不要的特征
        if i not in feature:
            test = test.drop(i, axis=1)

    # 对测试集进行预测
    predict = pd.Series(rf.predict(test), name="Survived")
    result = pd.concat([id, predict], axis=1)# 按提交格式拼接
    result.to_csv(result_path, index=False)
    print("预测结果已存入结果文件夹")

    #返回准确率
    return cv_mean,cv_std

#对预测集进行预测并存储结果
if __name__ == "__main__":
    model_rf(config.CLEAN_TRAIN_DATA,config.CLEAN_TEST_DATA,config.RF_RESULT,config.FEATURE_USED)


