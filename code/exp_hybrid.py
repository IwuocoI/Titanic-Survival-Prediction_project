import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import warnings
import sys
import os

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

#1.加载基础清洗数据
train = pd.read_csv(config.CLEAN_TRAIN_DATA)
test = pd.read_csv(config.CLEAN_TEST_DATA)

#只用基础特征（移除特征校验）
features = config.HYBRID_FEATURE_USED
X = train[features]
y = train['Survived']
X_test = test[features]
passenger_id = test['PassengerId']

#2.定义第一层基础模型
rf_params = config.RF_PARAMS.copy()
rf_params.pop('oob_score', None)

base_models = [
    ('lr', LogisticRegression(**config.LOGISTIC_PARAMS)),
    ('rf', RandomForestClassifier(**rf_params)),
    ('xgb', XGBClassifier(**config.XGBOOST_PARAMS))
]

#3.Stacking 核心算法（5折）
kf = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
train_meta_features = np.zeros((X.shape[0], len(base_models)))
test_meta_features = np.zeros((X_test.shape[0], len(base_models)))

for idx, (name, model) in enumerate(base_models):
    print(f"训练第一层模型：{name}")
    test_pred = np.zeros((X_test.shape[0], 5))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        #移除缺失值兜底逻辑，仅保留核心训练
        model.fit(X_train, y_train)
        train_meta_features[val_idx, idx] = model.predict_proba(X_val)[:, 1]
        test_pred[:, fold] = model.predict_proba(X_test)[:, 1]
    
    test_meta_features[:, idx] = test_pred.mean(axis=1)

#4.第二层：元学习器
meta_model = LogisticRegression(random_state=config.RANDOM_SEED)
meta_model.fit(train_meta_features, y)

#5.最终预测 & 保存结果
final_pred = meta_model.predict(test_meta_features)
final_pred = final_pred.astype(int)  # 确保符合Kaggle提交格式

sub = pd.DataFrame({
    'PassengerId': passenger_id,
    'Survived': final_pred
})

# 确保结果文件夹存在
os.makedirs(os.path.dirname(config.HYBRID_BASE_RESULT), exist_ok=True)
sub.to_csv(config.HYBRID_BASE_RESULT, index=False)

# 验证分数
train_final_pred = meta_model.predict(train_meta_features)
stacking_acc = accuracy_score(y, train_final_pred)
print(f"\nStacking混合模型训练集准确率：{stacking_acc:.4f}")
print(f"结果已保存至：{config.HYBRID_BASE_RESULT}")
