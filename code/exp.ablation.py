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
    评估模型，返回指标'accuracy','f1','roc_auc'的平均值
    """
    if scoring is None:
        scoring = {'accuracy': 'accuracy', 'f1': 'f1', 'roc_auc': 'roc_auc'}
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
    metrics = {}
    for metric in scoring:
        metrics[f'{metric}_mean'] = cv_results[f'test_{metric}'].mean()
    return metrics


def ablation(IN_PATH, features_to_remove, OUT_PATH, models_dict, cv=5):
    """
    消融实验
    features_to_remove: 列表，每个元素可以是字符串（单个特征）或列表/元组（一组特征）。
    """
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

    results_rows = []
    for remove_item in features_to_remove:
        # 确定要移除的特征列名(支持字符串或列表)
        if isinstance(remove_item, str):
            removed_features = [remove_item]
            remove_label = remove_item
        elif isinstance(remove_item, (list, tuple)):
            removed_features = list(remove_item)
            # 生成组标签
            first = removed_features[0]
            if first.startswith('Title_Sort_'):
                remove_label = 'Title_Sort_group'
            elif first.startswith('Embarked_'):
                remove_label = 'Embarked_group'
            elif first.startswith('Age_sort_'):
                remove_label = 'Age_sort_group'
            elif first.startswith('Fare_sort_'):
                remove_label = 'Fare_sort_group'
            else:
                remove_label = '_'.join(removed_features) + '_group'
        else:
            raise TypeError("features_to_remove 中的元素必须是字符串或列表")

        # 检查所有特征是否存在
        missing = [f for f in removed_features if f not in all_features]
        if missing:
            print(f"警告：特征 {missing} 不在数据中，跳过")
            continue

        current_features = [f for f in all_features if f not in removed_features]
        X_cur = data[current_features]
        row = {'Removed_Feature': remove_label}
        for name, (model_class, params) in models_dict.items():
            model = model_class(**params)
            cur_metrics = evaluate_model(X_cur, y, model, cv=cv)
            row[f'{name}_acc_change'] = cur_metrics['accuracy_mean'] - baseline_metrics[name]['accuracy_mean']
            row[f'{name}_f1_change'] = cur_metrics['f1_mean'] - baseline_metrics[name]['f1_mean']
            row[f'{name}_auc_change'] = cur_metrics['roc_auc_mean'] - baseline_metrics[name]['roc_auc_mean']
        results_rows.append(row)

    # 添加基线行（变化量为0）
    baseline_row = {'Removed_Feature': 'Baseline'}
    for name in models_dict.keys():
        baseline_row[f'{name}_acc_change'] = 0.0
        baseline_row[f'{name}_f1_change'] = 0.0
        baseline_row[f'{name}_auc_change'] = 0.0
    result_df = pd.concat([pd.DataFrame([baseline_row]), pd.DataFrame(results_rows)], ignore_index=True)

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
    # 定义要移除的特征：单个特征用字符串，一组特征用列表
    features_to_remove = [
        'Age',
        'Fare',
        'Sex',
        'Pclass',
        'Familysize',
        ['Age_sort_adult', 'Age_sort_elderly', 'Age_sort_kid', 'Age_sort_middle_age', 'Age_sort_youth'],
        ['Fare_sort_High', 'Fare_sort_Higher_Mid', 'Fare_sort_Low', 'Fare_sort_Lower_Mid'],

        # Title_Sort 组
        ['Title_Sort_elite', 'Title_Sort_married', 'Title_Sort_not_married', 'Title_Sort_unknown'],
        # Embarked 组
        ['Embarked_C', 'Embarked_Q', 'Embarked_S']
    ]

    out_path = config.OUT_PATH
    result_table = ablation(train_feat_path, features_to_remove, out_path, models, cv=5)

    print("\n消融实验结果(准确率变化量):")
    print(result_table[['Removed_Feature',
                        'LogisticRegression_acc_change',
                        'RandomForest_acc_change',
                        'XGBoost_acc_change']].to_string(index=False))

    print("\n消融实验结果(F1 变化量):")
    print(result_table[['Removed_Feature',
                        'LogisticRegression_f1_change',
                        'RandomForest_f1_change',
                        'XGBoost_f1_change']].to_string(index=False))

    print("\n消融实验结果(AUC 变化量):")
    print(result_table[['Removed_Feature',
                        'LogisticRegression_auc_change',
                        'RandomForest_auc_change',
                        'XGBoost_auc_change']].to_string(index=False))