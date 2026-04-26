import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import sys
sys.path.append("..")
import config

def model_lr(result_path, feature_cols, label_col="Survived", id_col="PassengerId"):
    """
    :param result_path: 预测结果保存路径
    :param feature_cols: 要使用的特征列列表（仅特征，不含标签/ID）
    :param label_col: 标签列名（适配Survived）
    :param id_col: ID列名（适配PassengerId）
    :return: 交叉验证平均分、标准差
    """
    #读取数据
    train_df = pd.read_csv(config.CLEAN_TRAIN_DATA)
    test_df = pd.read_csv(config.CLEAN_TEST_DATA)
    
    #分离特征和标签（训练集）：只保留特征列+标签列，删除ID列
    X_train = train_df[feature_cols].copy()  # 直接用指定的特征列，不用排除法
    y_train = train_df[label_col].copy()     # 标签列参数化
    test_passenger_id = test_df[id_col].copy()  # ID列参数化
    
    #测试集特征处理：只保留指定特征列，处理缺失值
    X_test = test_df[feature_cols].copy()
    #处理测试集Fare缺失值（针对缺失值进行的处理，若数据集在其它函数补全可注释/删除）
    if X_test["Fare"].isnull().any():
        fare_median = train_df["Fare"].median()  # 用训练集中位数填充
        X_test["Fare"].fillna(fare_median, inplace=True)
    
    #标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #模型训练+交叉验证
    model = LogisticRegression(**config.LOGISTIC_PARAMS)
    val_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_mean = val_scores.mean()
    cv_std = val_scores.std()
    print(f"逻辑回归5折交叉验证平均分: {cv_mean:.4f}, 标准差: {cv_std:.4f}")
    
    #预测并保存
    model.fit(X_train_scaled, y_train)
    pred = pd.Series(model.predict(X_test_scaled), name=label_col)
    result = pd.concat([test_passenger_id, pred], axis=1)
    result.to_csv(result_path, index=False)
    print(f"逻辑回归预测结果已存入: {result_path}")
    
    return cv_mean, cv_std
