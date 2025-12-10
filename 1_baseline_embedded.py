# 1_baseline_embedded.py
import pandas as pd
from xgboost import XGBClassifier
from data_loader import load_and_preprocess_nslkdd
from config import XGB_PARAMS, TOP_K_FEATURES
from utils import evaluate_performance
from sklearn.preprocessing import LabelEncoder


import warnings
warnings.filterwarnings("ignore")

def run_embedded_baseline():
    """
    运行纯嵌入式 XGBoost 特征选择基线
    """
    
    print("--- 阶段 1：开始运行纯嵌入式 XGBoost 基线 ---")
    
    # 1. 加载和预处理数据
    X_train_full, X_test_full, y_train, y_test, num_classes = load_and_preprocess_nslkdd()
    
    # 标签重新编码为从 0 开始的连续整数（合并后 fit，保证所有标签都被编码）
    label_encoder = LabelEncoder()
    label_encoder.fit(pd.concat([pd.Series(y_train), pd.Series(y_test)], ignore_index=True))
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)
    
    # 调整 num_class 参数以匹配实际的标签类别数 (例如 5 个或 40 个，取决于数据)
    XGB_PARAMS['num_class'] = num_classes 

    # 2. 训练完整特征集模型 (获取原始 Gain)
    print("\n[A] 训练完整特征集 (基线性能)")
    xgb_model_full = XGBClassifier(**XGB_PARAMS)
    xgb_model_full.fit(X_train_full, y_train)
    
    y_pred_full = xgb_model_full.predict(X_test_full)
    evaluate_performance(y_test, y_pred_full, subset_size=X_train_full.shape[1])
    
    # 3. 提取 Gain 特征重要性
    gain_scores = xgb_model_full.get_booster().get_score(importance_type='gain')
    
    # 将结果转换为 DataFrame 并按 Gain 排序
    importance_df = pd.DataFrame(list(gain_scores.items()), columns=['Feature', 'Gain'])
    importance_df = importance_df.sort_values(by='Gain', ascending=False)
    
    # 4. 选取 Top K 特征 (嵌入式特征选择)
    top_k_features = importance_df['Feature'].head(TOP_K_FEATURES).tolist()
    
    print(f"\n[B] 使用原始 Gain 选取的 Top {TOP_K_FEATURES} 特征：")
    # print(top_k_features)
    
    X_train_reduced = X_train_full[top_k_features]
    X_test_reduced = X_test_full[top_k_features]
    
    # 5. 在 Top K 特征上重新训练和评估 (嵌入式基线性能)
    print(f"\n[C] 在 Top {TOP_K_FEATURES} 特征上重新训练模型 (嵌入式基线)")
    
    # 重新初始化模型，保证公平性
    xgb_model_reduced = XGBClassifier(**XGB_PARAMS)
    xgb_model_reduced.fit(X_train_reduced, y_train)
    
    y_pred_reduced = xgb_model_reduced.predict(X_test_reduced)
    acc_baseline, f1_baseline = evaluate_performance(y_test, y_pred_reduced, subset_size=TOP_K_FEATURES)
    
    print(f"\n*** 嵌入式基线 (Top {TOP_K_FEATURES}) 性能: F1-Score {f1_baseline:.4f} ***")

if __name__ == '__main__':
    run_embedded_baseline()