import pandas as pd
import sys
sys.path.append("..")
import config
import model_rf as rf
import model_logistic as lo
import model_xgboost as xg
#特征工程实验
def feature(IN_PATH,OUT_PATH):
    data=pd.read_csv(IN_PATH)
    #缺失值处理
    data["Embarked"] = data["Embarked"].fillna("S")  # 众数填充
    data["Age"] = data.groupby("Pclass")["Age"].transform(lambda x: x.fillna(x.median()))  # 分舱室中位数填充
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())  # 验证集fare缺了一个
    #优化一：在姓名中取出title，分类提取特征
    data["Title"]=data["Name"].str.replace(".",",").str.split(",").str[1].str.strip()#取出title
    #分类
    married_list=["Mrs","Mme"]#已婚
    not_married_list=["Miss","Ms","Mlle"]#未婚
    unknown_list=["Mr"]#未知平民
    elite_list=["Lady","Sir","Jonkheer","the Countess","Dona","Don","Capt","Col","Major","Dr","Rev","Master"]#贵族和官员
    data["Sort"]=data["Title"].apply(lambda x:
                                     "married" if x in married_list else
                                     "not_married" if x in not_married_list else
                                     "unknown" if x in unknown_list else
                                     "elite")
    title_sort=pd.get_dummies(data["Sort"],"Title_Sort")#独热编码
    data=pd.concat([data,title_sort],axis=1)
    #优化二：整合出新特征家庭规模
    data["Familysize"]=data["SibSp"]+data["Parch"]+1
    '''
    #优化三：年龄分箱
    data["Age_sort"]=data["Age"].apply(lambda x:
                                       "kid" if x<=12 else
                                       "youth" if x<=20 else
                                       "adult" if x<=40 else
                                       "middle_age" if x<=60 else
                                       "elderly")
    data=pd.concat([data,pd.get_dummies(data["Age_sort"],"Age_sort")],axis=1)#独热编码
    '''
    #优化四：票价分箱
    data["Fare_sort"]=data["Fare"].apply(lambda x:
                                         "Low" if x<=10 else
                                         "Lower_Mid" if x<=25 else
                                         "Higher_Mid" if x<=100 else
                                         "High")
    data=pd.concat([data,pd.get_dummies(data["Fare_sort"],"Fare_sort")],axis=1)#独热编码
    #编码
    data["Sex"] = data["Sex"].map({"female": 0, "male": 1})  # 对性别进行编码，female：0，male：1
    embarked_dummies = pd.get_dummies(data["Embarked"],
                                      prefix="Embarked")  #独热编码
    data = pd.concat([data, embarked_dummies], axis=1)
    data=data.drop(["Fare_sort","Fare","Name","SibSp","Ticket","Parch","Cabin","Embarked","Title","Sort"],axis=1)
    data.to_csv(OUT_PATH,index=False)
    print("成功写入")
    return list(data.columns)
feature_list=feature(config.RAW_TRAIN_DATA,config.FEATURES_TRAIN_DATA)
feature(config.RAW_TEST_DATA,config.FEATURES_TEST_DATA)
lo_acc,lo_f1,lo_auc=lo.model_lr(config.FEATURES_TRAIN_DATA,config.FEATURES_TEST_DATA,config.FEATURES_LOGISTIC_RESULT,feature_list)
rf_acc,rf_f1,rf_auc=rf.model_rf(config.FEATURES_TRAIN_DATA,config.FEATURES_TEST_DATA,config.FEATURES_RF_RESULT,feature_list)
xg_acc,xg_f1,xg_auc=xg.model_xgboost(config.FEATURES_TRAIN_DATA,config.FEATURES_TEST_DATA,config.FEATURES_XGBOOST_RESULT,feature_list)

print(f"\n=== 特征工程后各模型 CV 指标 ===")
print(f"{'模型':<20} {'Accuracy':<12} {'F1':<12} {'AUC':<12}")
print(f"{'逻辑回归':<20} {lo_acc:<12.4f} {lo_f1:<12.4f} {lo_auc:<12.4f}")
print(f"{'随机森林':<20} {rf_acc:<12.4f} {rf_f1:<12.4f} {rf_auc:<12.4f}")
print(f"{'XGBoost':<20} {xg_acc:<12.4f} {xg_f1:<12.4f} {xg_auc:<12.4f}")