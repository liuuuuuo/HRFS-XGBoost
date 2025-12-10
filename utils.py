# utils.py (修正版本 - 增加 Precision 和 Recall 指标)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

def evaluate_performance(y_true, y_pred, subset_size=None, verbose=True):
    """评估性能，返回准确率、宏平均 F1-Score, Precision 和 Recall"""
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # 获取所有可能出现的标签，确保计算稳定性
    all_unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    
    # 使用 average='macro' 计算 F1-Score, Precision 和 Recall，确保对少数类别的公平评估
    # zero_division=0 确保在某些类别没有被预测到时不会报错
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0, labels=all_unique_labels)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0, labels=all_unique_labels)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0, labels=all_unique_labels)
    
    if verbose:
        print("-" * 40)
        if subset_size is not None:
            print(f"当前特征子集大小: {subset_size}")
        print(f"Accuracy (准确率): {accuracy:.4f}")
        print(f"F1-Score (宏平均 F1): {f1_macro:.4f}")
        print(f"Precision (宏平均精度): {precision_macro:.4f}")
        print(f"Recall (宏平均召回率): {recall_macro:.4f}")
        print("-" * 40)
        
    return accuracy, f1_macro, precision_macro, recall_macro