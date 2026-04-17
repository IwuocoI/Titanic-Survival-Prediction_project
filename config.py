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
TREE_NUMBER=1000
MAX_DEPTH=5
MAX_FEATURES="sqrt"
SPLIT_SAMPLES=4
LEAF_SAMPLES=1