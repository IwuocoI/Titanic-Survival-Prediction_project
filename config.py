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
LOGISTIC_RESULT=r"../result/model1.csv"
RF_RESULT=r"../result/model2.csv"
XGBOOST_RESULT=r"../result/model3.csv"

#模型输入特征列表
use_features=["Pclass","Sex","Age","Fare","Embarked","Embarked_Q","Embarked_S","Embarked_C"]

#种子
random_seed=54
