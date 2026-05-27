import pandas as pd
import sys

sys.path.append("..")#扩大运行环境以找到config
import config
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

def model_rf(train_data,test_data,result_path,feature):
    """
    :return: 交叉验证平均准确率、F1、AUC
    """
    data = pd.read_csv(train_data)
    for i in data.columns:  # 取用需要的特征
        if i not in feature:
            data = data.drop(i, axis=1)

    # 划分特征和标签
    x_train = data.drop("Survived", axis=1)
    y_train = data["Survived"]

    # 模型定义
    rf = RandomForestClassifier(**config.RF_PARAMS)

    #交叉验证(accuracy/F1/AUC)
    scoring = {'accuracy': 'accuracy', 'f1': 'f1', 'roc_auc': 'roc_auc'}
    cv_results = cross_validate(rf, x_train, y_train, cv=5, scoring=scoring, return_train_score=False)
    cv_mean = cv_results['test_accuracy'].mean()
    cv_std = cv_results['test_accuracy'].std()
    f1_mean = cv_results['test_f1'].mean()
    auc_mean = cv_results['test_roc_auc'].mean()
    print(f"随机森林模型:\n  Accuracy: {cv_mean:.4f} (±{cv_std:.4f})\n  F1:       {f1_mean:.4f}\n  AUC:      {auc_mean:.4f}")

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

    #返回准确率、F1、AUC
    return cv_mean, f1_mean, auc_mean

#对预测集进行预测并存储结果
if __name__ == "__main__":
    model_rf(config.CLEAN_TRAIN_DATA,config.CLEAN_TEST_DATA,config.RF_RESULT,config.FEATURE_USED)


