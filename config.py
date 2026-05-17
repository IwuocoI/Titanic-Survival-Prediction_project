#原始数据路径
RAW_TRAIN_DATA=r"../data/raw_data/train.csv"
RAW_TEST_DATA=r"../data/raw_data/test.csv"

#处理后数据路径
CLEAN_TRAIN_DATA=r"../data/processed_data/train.csv"
CLEAN_TEST_DATA=r"../data/processed_data/test.csv"

#特征工程数据路径
FEATURES_TRAIN_DATA=r"../data/feature_data/train.csv"
FEATURES_TEST_DATA=r"../data/feature_data/test.csv"

#特征工程结果路径
FEATURES_LOGISTIC_RESULT=r"../results/experiment/feature/kaggle/model1.csv"
FEATURES_RF_RESULT=r"../results/experiment/feature/kaggle/model2.csv"
FEATURES_XGBOOST_RESULT=r"../results/experiment/feature/kaggle/model3.csv"

#结果输出路径
LOGISTIC_RESULT=r"../results/kaggle/model1.csv"
RF_RESULT=r"../results/kaggle/model2.csv"
XGBOOST_RESULT=r"../results/kaggle/model3.csv"

#模型输入特征列表
FEATURE_USED=["Survived","Pclass","Sex","Age","Fare","Embarked","Embarked_Q","Embarked_S","Embarked_C"]

#种子
RANDOM_SEED=54

#rf模型超参数
RF_PARAMS={
"n_estimators":500,
"max_depth":6,
"max_features":"sqrt",
"min_samples_split":4,
"min_samples_leaf":1,
"oob_score":True,
"n_jobs":-1,
"random_state":RANDOM_SEED
}

# XGBoost 模型超参数
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric":["error", "logloss", "auc"],
    "n_estimators": 1000,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.80,
    "colsample_bytree": 0.85,
    "min_child_weight": 5,
    "gamma": 0.2,
    "reg_alpha": 0,
    "reg_lambda": 5,
    "max_delta_step": 0,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

# 逻辑回归模型超参数
LOGISTIC_PARAMS = {
    "C": 1.0,
    "solver": "liblinear",
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
}