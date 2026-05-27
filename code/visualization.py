import sys
sys.path.append("..")
import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 设置中文字体，防乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 通用颜色（存活绿/死亡红，模型分类色）
CLRS = ['#C44E52', '#55A868', '#4C72B0', '#DD8452', '#8172B3']
SAVE_DIR = "../results/vis"

# ============================================================
# 分组一：数据探索 EDA
# ============================================================

def vis_survived_pie(data_path, save_path):
    """生存分布饼图"""
    data = pd.read_csv(data_path)
    counts = data['Survived'].value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=['死亡', '存活'], autopct='%1.1f%%',
            colors=['#C44E52', '#55A868'], startangle=90)
    plt.title('泰坦尼克号乘客生存分布')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"生存分布饼图已保存: {save_path}")

def vis_age_dist(data_path, save_path):
    """年龄分布直方图（按生存分色+核密度）"""
    data = pd.read_csv(data_path)

    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='Age', hue='Survived', kde=True,
                 palette=['#C44E52', '#55A868'], bins=30, alpha=0.6)
    plt.title('乘客年龄分布（按生存情况）')
    plt.xlabel('年龄')
    plt.ylabel('人数')
    plt.legend(['存活', '死亡'])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"年龄分布图已保存: {save_path}")

def vis_fare_dist(data_path, save_path):
    """票价分布直方图（按生存分色，限300以内避免长尾干扰）"""
    data = pd.read_csv(data_path)

    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='Fare', hue='Survived', kde=True,
                 palette=['#C44E52', '#55A868'], bins=50, alpha=0.6)
    plt.title('票价分布（按生存情况）')
    plt.xlabel('票价')
    plt.ylabel('人数')
    plt.xlim(0, 300)
    plt.legend(['存活', '死亡'])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"票价分布图已保存: {save_path}")

def vis_corr_heatmap(data_path, save_path):
    """特征相关性热力图"""
    data = pd.read_csv(data_path)
    numeric_data = data.select_dtypes(include=[np.number])
    corr = numeric_data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5)
    plt.title('特征相关性热力图')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"相关性热力图已保存: {save_path}")

# ============================================================
# 分组二：单特征 vs 生存率
# ============================================================

def vis_pclass_survived(data_path, save_path):
    """船舱等级 vs 生存分组柱状图"""
    data = pd.read_csv(data_path)
    cross = pd.crosstab(data['Pclass'], data['Survived'])

    plt.figure(figsize=(8, 6))
    cross.plot(kind='bar', stacked=False, color=['#C44E52', '#55A868'], ax=plt.gca())
    plt.title('船舱等级与生存率')
    plt.xlabel('船舱等级')
    plt.ylabel('人数')
    plt.xticks(rotation=0)
    plt.legend(['死亡', '存活'])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"船舱等级-生存图已保存: {save_path}")

def vis_sex_survived(data_path, save_path):
    """性别 vs 生存分组柱状图"""
    data = pd.read_csv(data_path)
    sex_map = {0: '女性', 1: '男性'}
    data['Sex_label'] = data['Sex'].map(sex_map)
    cross = pd.crosstab(data['Sex_label'], data['Survived'])

    plt.figure(figsize=(7, 6))
    cross.plot(kind='bar', stacked=False, color=['#C44E52', '#55A868'], ax=plt.gca())
    plt.title('性别与生存率')
    plt.xlabel('性别')
    plt.ylabel('人数')
    plt.xticks(rotation=0)
    plt.legend(['死亡', '存活'])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"性别-生存图已保存: {save_path}")

def vis_embarked_survived(data_path, save_path):
    """登船港口 vs 生存分组柱状图"""
    data = pd.read_csv(data_path)
    # 从独热编码还原Embarked列
    port_map = {col.replace('Embarked_', ''): col
                for col in data.columns if col.startswith('Embarked_')}
    def decode_embarked(row):
        for port, col in port_map.items():
            if row[col] == 1:
                return port
        return 'Unknown'
    data['Embarked_label'] = data.apply(decode_embarked, axis=1)
    cross = pd.crosstab(data['Embarked_label'], data['Survived'])

    plt.figure(figsize=(7, 6))
    cross.plot(kind='bar', stacked=False, color=['#C44E52', '#55A868'], ax=plt.gca())
    plt.title('登船港口与生存率')
    plt.xlabel('登船港口')
    plt.ylabel('人数')
    plt.xticks(rotation=0)
    plt.legend(['死亡', '存活'])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"登船港口-生存图已保存: {save_path}")

def vis_age_boxplot(data_path, save_path):
    """年龄箱线图（按生存分组对比）"""
    data = pd.read_csv(data_path)

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data, x='Survived', y='Age',
                palette=['#C44E52', '#55A868'])
    plt.title('年龄分布箱线图（按生存情况）')
    plt.xlabel('生存情况（0=死亡, 1=存活）')
    plt.ylabel('年龄')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"年龄箱线图已保存: {save_path}")

def vis_fare_boxplot(data_path, save_path):
    """票价箱线图（按生存分组对比）"""
    data = pd.read_csv(data_path)

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data, x='Survived', y='Fare',
                palette=['#C44E52', '#55A868'])
    plt.title('票价分布箱线图（按生存情况）')
    plt.xlabel('生存情况（0=死亡, 1=存活）')
    plt.ylabel('票价')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"票价箱线图已保存: {save_path}")

# ============================================================
# 分组三：自造特征展示
# ============================================================

def vis_title_survived(data_path, save_path):
    """Title分类 vs 生存分组柱状图"""
    data = pd.read_csv(data_path)
    # 从独热编码还原Sort列
    sort_map = {col.replace('Title_Sort_', ''): col
                for col in data.columns if col.startswith('Title_Sort_')}
    # 中文标签映射
    title_label_map = {
        'married': '已婚女性',
        'not_married': '未婚女性',
        'unknown': '平民男性',
        'elite': '精英',
    }
    def decode_sort(row):
        for sort_val, col in sort_map.items():
            if row[col] == 1:
                return title_label_map.get(sort_val, sort_val)
        return 'Unknown'
    data['Sort_label'] = data.apply(decode_sort, axis=1)
    cross = pd.crosstab(data['Sort_label'], data['Survived'])

    plt.figure(figsize=(9, 6))
    cross.plot(kind='bar', stacked=False, color=['#C44E52', '#55A868'], ax=plt.gca())
    plt.title('Title 分类与生存率')
    plt.xlabel('Title 分类')
    plt.ylabel('人数')
    plt.xticks(rotation=0)
    plt.legend(['死亡', '存活'])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Title分类-生存图已保存: {save_path}")

def vis_familysize_survived(data_path, save_path):
    """家庭规模 vs 生存分组柱状图"""
    data = pd.read_csv(data_path)
    cross = pd.crosstab(data['Familysize'], data['Survived'])

    plt.figure(figsize=(9, 6))
    cross.plot(kind='bar', stacked=False, color=['#C44E52', '#55A868'], ax=plt.gca())
    plt.title('家庭规模与生存率')
    plt.xlabel('家庭规模')
    plt.ylabel('人数')
    plt.xticks(rotation=0)
    plt.legend(['死亡', '存活'])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"家庭规模-生存图已保存: {save_path}")

def vis_faresort_survived(data_path, save_path):
    """票价分箱 vs 生存分组柱状图"""
    data = pd.read_csv(data_path)
    # 从独热编码还原Fare_sort列
    fare_map = {col.replace('Fare_sort_', ''): col
                for col in data.columns if col.startswith('Fare_sort_')}
    # 正确排序：Low → Lower_Mid → Higher_Mid → High
    fare_order = ['Low', 'Lower_Mid', 'Higher_Mid', 'High']
    fare_label_map = {
        'Low': '低价',
        'Lower_Mid': '中低价',
        'Higher_Mid': '中高价',
        'High': '高价',
    }
    def decode_fare_sort(row):
        for val, col in fare_map.items():
            if row[col] == 1:
                return val
        return 'Unknown'
    data['Fare_sort_label'] = data.apply(decode_fare_sort, axis=1)
    # 按指定顺序排序
    data['Fare_sort_label'] = pd.Categorical(data['Fare_sort_label'],
                                              categories=fare_order, ordered=True)
    cross = pd.crosstab(data['Fare_sort_label'], data['Survived'])

    plt.figure(figsize=(9, 6))
    cross.plot(kind='bar', stacked=False, color=['#C44E52', '#55A868'], ax=plt.gca())
    # x轴标签替换为中文
    labels = [fare_label_map.get(l.get_text(), l.get_text()) for l in plt.gca().get_xticklabels()]
    plt.gca().set_xticklabels(labels)
    plt.title('票价分箱与生存率')
    plt.xlabel('票价分箱')
    plt.ylabel('人数')
    plt.xticks(rotation=0)
    plt.legend(['死亡', '存活'])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"票价分箱-生存图已保存: {save_path}")

# ============================================================
# 分组四：模型性能对比
# ============================================================

def vis_model_comparison_6groups(baseline_metrics, feat_metrics, basline_kaggle, feat_kaggle, save_path,
                                  hybrid_metrics=None, hybrid_kaggle=None):
    """
    模型性能 + Kaggle分数对比图（含混合模型）
    baseline_metrics: [{'name':'逻辑回归', 'acc':a, 'f1':f, 'auc':u}, ...]  (3个基线)
    feat_metrics:     [同上, 特征工程后] (3个特征)
    basline_kaggle:   [lr_kaggle, rf_kaggle, xgb_kaggle]  (基线Kaggle分)
    feat_kaggle:      [lr_kaggle, rf_kaggle, xgb_kaggle]  (特征工程后Kaggle分)
    hybrid_metrics:   {'acc':a, 'f1':f, 'auc':u}  (混合模型, 可选)
    hybrid_kaggle:    float (混合模型Kaggle分, 可选)
    """
    model_names = ['逻辑回归', '随机森林', 'XGBoost']
    group_labels = []
    for name in model_names:
        group_labels.append(f'{name}\n基线')
        group_labels.append(f'{name}\n特征工程')
    
    all_acc = []
    all_f1 = []
    all_auc = []
    all_kaggle = []
    for i in range(3):
        all_acc.append(baseline_metrics[i]['acc'])
        all_f1.append(baseline_metrics[i]['f1'])
        all_auc.append(baseline_metrics[i]['auc'])
        all_kaggle.append(baseline_kaggle[i])
        all_acc.append(feat_metrics[i]['acc'])
        all_f1.append(feat_metrics[i]['f1'])
        all_auc.append(feat_metrics[i]['auc'])
        all_kaggle.append(feat_kaggle[i])

    # 添加混合模型（如果有）
    has_hybrid = hybrid_metrics is not None
    if has_hybrid:
        group_labels.append('混合模型\nStacking')
        all_acc.append(hybrid_metrics['acc'])
        all_f1.append(hybrid_metrics['f1'])
        all_auc.append(hybrid_metrics['auc'])
        all_kaggle.append(hybrid_kaggle if hybrid_kaggle is not None else 0)

    x = np.arange(len(group_labels))
    w = 0.18

    fig, ax = plt.subplots(figsize=(15, 7))

    bars_acc = ax.bar(x - 1.5*w, all_acc, w, label='Accuracy', color='#4C72B0')
    bars_f1  = ax.bar(x - 0.5*w, all_f1,  w, label='F1',        color='#55A868')
    bars_auc = ax.bar(x + 0.5*w, all_auc, w, label='AUC',       color='#DD8452')
    bars_kgl = ax.bar(x + 1.5*w, all_kaggle, w, label='Kaggle',    color='#8172B3')

    all_vals = all_acc + all_f1 + all_auc + [v for v in all_kaggle if v > 0]
    y_min = max(0.45, min(all_vals) - 0.03)
    y_max = max(all_vals) + 0.03
    ax.set_ylim(y_min, y_max)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=10)
    ax.set_ylabel('得分', fontsize=12)
    ax.set_title('模型性能指标与Kaggle分数对比', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')

    for bars in [bars_acc, bars_f1, bars_auc, bars_kgl]:
        for bar in bars:
            height = bar.get_height()
            if height == 0:
                continue
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=7, rotation=30)

    # 分隔线
    for i in range(1, 3):
        sep_x = i * 2 - 0.5
        ax.axvline(x=sep_x, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"模型性能对比图已保存: {save_path}")

def vis_feature_importance(data_path, save_path):
    """RF特征重要性水平柱状图"""
    data = pd.read_csv(data_path)
    # 取特征列（去掉PassengerId和Survived）
    feature_cols = [c for c in data.columns if c not in ['PassengerId', 'Survived']]
    X = data[feature_cols]
    y = data['Survived']

    # 训练RF获取特征重要性
    rf = RandomForestClassifier(**config.RF_PARAMS)
    rf.fit(X, y)
    imp = pd.DataFrame({'feature': feature_cols, 'importance': rf.feature_importances_})
    imp = imp.sort_values('importance', ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(imp['feature'], imp['importance'], color='#4C72B0')
    plt.title('随机森林特征重要性')
    plt.xlabel('重要性')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"RF特征重要性图已保存: {save_path}")

# ============================================================
# 分组五：消融实验可视化
# ============================================================

def vis_ablation_per_model(ablation_path, model_key, model_label, save_path):
    """
    单模型消融实验：每个特征显示acc/f1/auc变化量
    model_key: 'LogisticRegression' / 'RandomForest' / 'XGBoost'
    """
    df = pd.read_csv(ablation_path)
    df_plot = df[df['Removed_Feature'] != 'Baseline'].copy()

    features = df_plot['Removed_Feature'].values
    n = len(features)
    x = np.arange(n)
    w = 0.22

    acc_vals = df_plot[f'{model_key}_acc_change'].values
    f1_vals  = df_plot[f'{model_key}_f1_change'].values
    auc_vals = df_plot[f'{model_key}_auc_change'].values

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_acc = ax.bar(x - w, acc_vals, w, label='Accuracy', color='#4C72B0')
    bars_f1  = ax.bar(x,     f1_vals,  w, label='F1',        color='#55A868')
    bars_auc = ax.bar(x + w, auc_vals, w, label='AUC',       color='#DD8452')

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=10, rotation=30, ha='right')
    ax.set_ylabel('变化量', fontsize=12)
    ax.set_title(f'消融实验：{model_label} — 移除特征后指标变化', fontsize=14)
    ax.legend(fontsize=10)

    # 柱上标数值
    for bars in [bars_acc, bars_f1, bars_auc]:
        for bar in bars:
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 0.0005 if height >= 0 else -0.0005
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                    f'{height:.4f}', ha='center', va=va, fontsize=7, rotation=30)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"消融实验图已保存: {save_path}")


# ============================================================
# 主函数：一键生成全部可视化
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("开始生成全部可视化图片")
    print("=" * 50)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- 分组一：数据探索（基于清洗后数据，有数值编码但无缺失值） ---
    clean_train = config.CLEAN_TRAIN_DATA
    vis_survived_pie(clean_train, f"{SAVE_DIR}/01_survived_pie.png")
    vis_age_dist(clean_train, f"{SAVE_DIR}/02_age_dist.png")
    vis_fare_dist(clean_train, f"{SAVE_DIR}/03_fare_dist.png")
    vis_corr_heatmap(clean_train, f"{SAVE_DIR}/04_corr_heatmap.png")

    # --- 分组二：单特征 vs 生存率（清理后数据已有Sex编码，但自己还原标签） ---
    vis_pclass_survived(clean_train, f"{SAVE_DIR}/05_pclass_survived.png")
    vis_sex_survived(clean_train, f"{SAVE_DIR}/06_sex_survived.png")
    vis_embarked_survived(clean_train, f"{SAVE_DIR}/07_embarked_survived.png")
    # vis_age_boxplot(clean_train, f"{SAVE_DIR}/08_age_boxplot.png")  # 已删除
    # vis_fare_boxplot(clean_train, f"{SAVE_DIR}/09_fare_boxplot.png")  # 已删除

    # --- 分组三：自造特征（基于特征工程后数据） ---
    feat_train = config.FEATURES_TRAIN_DATA
    vis_title_survived(feat_train, f"{SAVE_DIR}/10_title_survived.png")
    vis_familysize_survived(feat_train, f"{SAVE_DIR}/11_familysize_survived.png")
    vis_faresort_survived(feat_train, f"{SAVE_DIR}/12_faresort_survived.png")

    # --- 分组四：模型性能对比（基线与特征工程后，含Kaggle分数） ---
    from model_logistic import model_lr
    from model_rf import model_rf
    from model_xgboost import model_xgboost

    # --- 基线模型（特征工程前，清洗数据） ---
    cv_feat_clean = config.FEATURE_USED  # 包含Survived，模型内会自己drop
    b_lr_acc, b_lr_f1, b_lr_auc = model_lr(clean_train, config.CLEAN_TEST_DATA,
                                             f"{SAVE_DIR}/_tmp_lr.csv", cv_feat_clean)
    b_rf_acc, b_rf_f1, b_rf_auc = model_rf(clean_train, config.CLEAN_TEST_DATA,
                                             f"{SAVE_DIR}/_tmp_rf.csv", cv_feat_clean)
    b_xg_acc, b_xg_f1, b_xg_auc = model_xgboost(clean_train, config.CLEAN_TEST_DATA,
                                                  f"{SAVE_DIR}/_tmp_xgb.csv", cv_feat_clean)

    # --- 特征工程后模型（用feature_data） ---
    # 加载特征工程后数据，获取所有特征列（去掉PassengerId和Survived）
    feat_df_cols = pd.read_csv(config.FEATURES_TRAIN_DATA).columns.tolist()
    cv_feat_feat = [c for c in feat_df_cols if c != 'PassengerId']  # 模型内会drop Survived

    f_lr_acc, f_lr_f1, f_lr_auc = model_lr(config.FEATURES_TRAIN_DATA, config.FEATURES_TEST_DATA,
                                             f"{SAVE_DIR}/_tmp_flr.csv", cv_feat_feat)
    f_rf_acc, f_rf_f1, f_rf_auc = model_rf(config.FEATURES_TRAIN_DATA, config.FEATURES_TEST_DATA,
                                             f"{SAVE_DIR}/_tmp_frf.csv", cv_feat_feat)
    f_xg_acc, f_xg_f1, f_xg_auc = model_xgboost(config.FEATURES_TRAIN_DATA, config.FEATURES_TEST_DATA,
                                                  f"{SAVE_DIR}/_tmp_fxgb.csv", cv_feat_feat)

    # 清理临时预测文件
    for tmp in ['_tmp_lr.csv', '_tmp_rf.csv', '_tmp_xgb.csv',
                '_tmp_flr.csv', '_tmp_frf.csv', '_tmp_fxgb.csv']:
        tmp_path = f"{SAVE_DIR}/{tmp}"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    baseline_metrics = [
        {'name': '逻辑回归', 'acc': b_lr_acc, 'f1': b_lr_f1, 'auc': b_lr_auc},
        {'name': '随机森林', 'acc': b_rf_acc, 'f1': b_rf_f1, 'auc': b_rf_auc},
        {'name': 'XGBoost',  'acc': b_xg_acc, 'f1': b_xg_f1, 'auc': b_xg_auc},
    ]
    feat_metrics = [
        {'name': '逻辑回归', 'acc': f_lr_acc, 'f1': f_lr_f1, 'auc': f_lr_auc},
        {'name': '随机森林', 'acc': f_rf_acc, 'f1': f_rf_f1, 'auc': f_rf_auc},
        {'name': 'XGBoost',  'acc': f_xg_acc, 'f1': f_xg_f1, 'auc': f_xg_auc},
    ]
    # Kaggle分数（从刚提交的结果）
    baseline_kaggle = [0.74641, 0.79665, 0.77272]
    feat_kaggle =     [0.77511, 0.77751, 0.75837]

    # --- 混合模型 ---
    hybrid_base_path = config.HYBRID_BASE_RESULT
    if os.path.exists(hybrid_base_path):
        hybrid_metrics = {'acc': 0.8227, 'f1': 0.7460, 'auc': 0.8644}
        hybrid_kaggle_val = 0.78708
    else:
        hybrid_metrics = None
        hybrid_kaggle_val = None

    vis_model_comparison_6groups(baseline_metrics, feat_metrics, baseline_kaggle, feat_kaggle,
                                  f"{SAVE_DIR}/13_model_comparison_6groups.png",
                                  hybrid_metrics=hybrid_metrics, hybrid_kaggle=hybrid_kaggle_val)
    vis_feature_importance(feat_train, f"{SAVE_DIR}/14_feature_importance.png")

    # --- 分组五：消融实验（每模型一张图，特征×指标） ---
    ab_path = config.OUT_PATH
    vis_ablation_per_model(ab_path, 'LogisticRegression', '逻辑回归',
                            f"{SAVE_DIR}/15_ablation_lr.png")
    vis_ablation_per_model(ab_path, 'RandomForest', '随机森林',
                            f"{SAVE_DIR}/16_ablation_rf.png")
    vis_ablation_per_model(ab_path, 'XGBoost', 'XGBoost',
                            f"{SAVE_DIR}/17_ablation_xgb.png")

    print("\n" + "=" * 50)
    print("全部可视化生成完毕！文件保存在:", SAVE_DIR)
    print("=" * 50)
