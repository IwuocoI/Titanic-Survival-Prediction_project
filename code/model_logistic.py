import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import sys
sys.path.append("..")
import config

train_df = pd.read_csv(config.CLEAN_TRAIN_DATA)
test_df = pd.read_csv(config.CLEAN_TEST_DATA)

# 筛选特征列
feature_cols = [col for col in train_df.columns if col not in ["PassengerId", "Survived"]]
X_train = train_df[feature_cols].copy()
y_train = train_df["Survived"].copy()
X_test = test_df[feature_cols].copy()
test_passenger_id = test_df["PassengerId"].copy()

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(**config.LOGISTIC_PARAMS)

# 5折交叉验证
val_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"逻辑回归5折交叉验证平均分: {val_scores.mean():.4f}")

# 训练最终模型 + 预测测试集
model.fit(X_train_scaled, y_train)
pred = pd.Series(model.predict(X_test_scaled), name="Survived")
result = pd.concat([test_passenger_id, pred], axis=1)

result.to_csv(config.LOGISTIC_RESULT, index=False)
print("逻辑回归预测结果已存入: ", config.LOGISTIC_RESULT)
