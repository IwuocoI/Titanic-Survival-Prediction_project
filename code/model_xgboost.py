import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
import sys
sys.path.append("..")
import config


def model_xgboost(result_path, features):
    # 数据处理
    data = pd.read_csv(config.CLEAN_TRAIN_DATA)
    for i in data.columns:
        if i not in features:
            data = data.drop(i, axis=1)

    x_train = data.drop("Survived", axis=1)
    y_train = data["Survived"]

    # 多指标交叉验证
    print("开始 5 折交叉验证（多指标）")
    cv_model = XGBClassifier(**config.XGBOOST_PARAMS)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    cv_results = cross_validate(
        cv_model, x_train, y_train, cv=5,
        scoring=scoring, return_train_score=False
    )
    for metric in scoring:
        scores = cv_results[f'test_{metric}']
        print(f"{metric:>10}: {scores.mean():.4f} (±{scores.std():.4f})")
    mean_accuracy = cv_results['test_accuracy'].mean()
    print(f"交叉验证平均准确率: {mean_accuracy:.4f}")

    #训练最终模型
    print("\n开始训练最终模型(全量训练集)")
    final_model = XGBClassifier(**config.XGBOOST_PARAMS)
    final_model.fit(x_train, y_train)

    # 预测
    test = pd.read_csv(config.CLEAN_TEST_DATA)
    id = test["PassengerId"]
    for i in test.columns:
        if i not in features:
            test = test.drop(i, axis=1)

    pred = pd.Series(final_model.predict(test), name="Survived")
    result = pd.concat([id, pred], axis=1)
    result.to_csv(result_path, index=False)
    print("预测结果已存入结果文件夹")

    return mean_accuracy


if __name__ == "__main__":
    model_xgboost(config.XGBOOST_RESULT, config.FEATURE_USED)