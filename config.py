#原始数据路径
RAW_TRAIN_DATA=r"../data/raw_data/train.csv"
RAW_TEST_DATA=r"../data/raw_data/test.csv"

#处理后数据路径
CLEAN_TRAIN_DATA=r"../data/processed_data/train.csv"
CLEAN_TEST_DATA=r"../data/processed_data/test.csv"

#特征工程数据路径
FEATURES_TRAIN_DATA=r"../data/feature_data/train.csv"
FEATURES_TEST_DATA=r"../data/feature_data/test.csv"

#结果输出路径
LOGISTIC_RESULT=r"../results/kaggle/model1.csv"
RF_RESULT=r"../results/kaggle/model2.csv"
XGBOOST_RESULT=r"../results/kaggle/model3.csv"

#模型输入特征列表
FEATURE_USED=["Survived","Pclass","Sex","Age","Fare","Embarked","Embarked_Q","Embarked_S","Embarked_C"]

#种子
RANDOM_SEED=54

#rf模型超参数
TREE_NUMBER=500
MAX_DEPTH=6
MAX_FEATURES="sqrt"
SPLIT_SAMPLES=4
LEAF_SAMPLES=1

# XGBoost 模型超参数
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric":"logloss",
    "n_estimators": 1000,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.4,
    "reg_alpha": 0.2,
    "reg_lambda": 5,
    "max_delta_step": 0,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

# 逻辑回归模型超参数
LOGISTIC_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "liblinear",
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}
