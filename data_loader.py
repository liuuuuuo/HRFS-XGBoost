# data_loader.py (使用 factorize 的稳定版本)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import FEATURE_NAMES, CATEGORICAL_FEATURES, LABEL_COLUMN
import numpy as np

def load_and_preprocess_nslkdd(train_path='data/KDDTrain+.txt', test_path='data/KDDTest+.txt'):
    
    # 1. 加载数据
    df_train = pd.read_csv(train_path, header=None, names=FEATURE_NAMES).drop('level', axis=1, errors='ignore')
    df_test = pd.read_csv(test_path, header=None, names=FEATURE_NAMES).drop('level', axis=1, errors='ignore')
    
    # --- 标签处理 (使用 factorize 确保标签从 0 开始且全局连续) ---
    all_labels_series = pd.concat([df_train[LABEL_COLUMN], df_test[LABEL_COLUMN]]).astype(str)
    
    # pd.factorize 保证从 0 开始编码，且序列是连续的
    all_labels_encoded, unique_labels_str = pd.factorize(all_labels_series)
    
    num_classes = len(unique_labels_str) # 获得总类别数

    # 将编码后的标签重新分回训练集和测试集
    y_train = pd.Series(all_labels_encoded[:len(df_train)])
    y_test = pd.Series(all_labels_encoded[len(df_train):])
    
    # 将标签列从原始 DataFrame 中移除
    X_train = df_train.drop(LABEL_COLUMN, axis=1)
    X_test = df_test.drop(LABEL_COLUMN, axis=1)
    
    # --- 特征处理 ---
    combined_X = pd.concat([X_train, X_test], ignore_index=True)
    combined_X = pd.get_dummies(combined_X, columns=CATEGORICAL_FEATURES, drop_first=False)
    
    X_train = combined_X.iloc[:len(df_train)]
    X_test = combined_X.iloc[len(df_train):]
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    print(f"数据加载完成。训练集特征数: {X_train_scaled.shape[1]}")
    return X_train_scaled, X_test_scaled, y_train, y_test, num_classes