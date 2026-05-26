import pandas as pd
import sys
sys.path.append("..")
import config
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def evaluate_model(X, y, model, cv=5, scoring=None):
    """
    评估模型，返回指标'accuracy','f1'的平均值
    """
    if scoring is None:
        scoring = {'accuracy': 'accuracy', 'f1': 'f1'}
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
    metrics = {}
    for metric in scoring:
        metrics[f'{metric}_mean'] = cv_results[f'test_{metric}'].mean()
    return metrics

def ablation(IN_PATH, features_to_remove, OUT_PATH, models_dict, cv=5):
    data = pd.read_csv(IN_PATH)
    if 'Survived' not in data.columns:
        raise ValueError("数据中缺少 'Survived' 列")
    all_features = [col for col in data.columns if col not in ['PassengerId', 'Survived']]
    y = data['Survived']
    
    # 基线：使用全部特征
    X_base = data[all_features]
    baseline_metrics = {}
    for name, (model_class, params) in models_dict.items():
        model = model_class(**params)
        baseline_metrics[name] = evaluate_model(X_base, y, model, cv=cv)
    
    # 存储结果
    results_rows = []
    for removed_feat in features_to_remove:
        if removed_feat not in all_features:
            print(f"警告：特征 '{removed_feat}' 不在数据中，跳过")
            continue
        current_features = [f for f in all_features if f != removed_feat]
        X_cur = data[current_features]
        row = {'Removed_Feature': removed_feat}
        for name, (model_class, params) in models_dict.items():
            model = model_class(**params)
            cur_metrics = evaluate_model(X_cur, y, model, cv=cv)
            row[f'{name}_acc_change'] = cur_metrics['accuracy_mean'] - baseline_metrics[name]['accuracy_mean']
            row[f'{name}_f1_change'] = cur_metrics['f1_mean'] - baseline_metrics[name]['f1_mean']
        results_rows.append(row)
    
    result_df = pd.DataFrame(results_rows)
    # 添加基线行（变化量为0）
    baseline_row = {'Removed_Feature': 'Baseline'}
    for name in models_dict.keys():
        baseline_row[f'{name}_acc_change'] = 0.0
        baseline_row[f'{name}_f1_change'] = 0.0
    result_df = pd.concat([pd.DataFrame([baseline_row]), result_df], ignore_index=True)
    
    result_df.to_csv(OUT_PATH, index=False)
    print(f"消融实验结果已保存至：{OUT_PATH}")
    return result_df

if __name__ == "__main__":
    models = {
        'LogisticRegression': (LogisticRegression, config.LOGISTIC_PARAMS),
        'RandomForest': (RandomForestClassifier, config.RF_PARAMS),
        'XGBoost': (XGBClassifier, config.XGBOOST_PARAMS)
    }
    
    train_feat_path = config.IN_PATH
    # 选择4-6个重要特征进行消融
    features_to_remove = [
        'Sex',
        'Pclass',
        'Familysize',
        'Title_Sort_elite',
        'Age',
        'Fare'
    ]
    
    out_path = config.OUT_PATH
    result_table = ablation(train_feat_path, features_to_remove, out_path, models, cv=5)
    
    print("\n消融实验结果（准确率变化量）：")
    print(result_table[['Removed_Feature', 
                        'LogisticRegression_acc_change', 
                        'RandomForest_acc_change', 
                        'XGBoost_acc_change']].to_string(index=False))
    
    print("\n消融实验结果（F1 变化量）：")
    print(result_table[['Removed_Feature', 
                        'LogisticRegression_f1_change', 
                        'RandomForest_f1_change', 
                        'XGBoost_f1_change']].to_string(index=False))