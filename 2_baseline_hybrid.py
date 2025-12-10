# 2_baseline_hybrid.py (ACO-XGBoost 混合基线 - 包含计时和所有指标)

import pandas as pd
from xgboost import XGBClassifier
from data_loader import load_and_preprocess_nslkdd
from config import XGB_PARAMS, TOP_K_FEATURES
from utils import evaluate_performance
from aco_fs import ACO_XGBoost_FS # 依赖于 aco_fs.py 文件

import warnings
import time 
import numpy as np
warnings.filterwarnings("ignore")

def run_hybrid_aco_baseline():
    
    total_start_time = time.time() # <<< 计时开始
    
    print("=" * 70)
    print("--- 阶段 2：开始运行 ACO-XGBoost 混合基线 ---")
    print("=" * 70)
    
    # 1. 加载和预处理数据
    data_start = time.time()
    X_train_full, X_test_full, y_train, y_test, num_classes = load_and_preprocess_nslkdd()
    print(f"数据加载完成。训练集特征数: {X_train_full.shape[1]}")
    data_load_time = time.time() - data_start

    # 调整 XGBoost 参数，设置正确的类别数
    current_xgb_params = XGB_PARAMS.copy()
    current_xgb_params.update({
        'num_class': num_classes, 
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'verbosity': 0
    })
    
    # 2. 初始化并运行 ACO 特征选择
    print(f"\n[A] 开始运行 ACO 优化（目标特征数 K={TOP_K_FEATURES}）...")
    aco_start_time = time.time()
    
    # 初始化 ACO Solver
    aco_solver = ACO_XGBoost_FS(
        X=X_train_full, 
        y=y_train,
        xgb_params=current_xgb_params,
        n_ants=10, 
        max_iter=10, 
        alpha=1.0, 
        beta=1.0, 
        rho=0.1, 
        target_feature_count=TOP_K_FEATURES
    )
    
    best_features, best_cv_f1 = aco_solver.run()
    aco_time = time.time() - aco_start_time
    
    print("\n[B] ACO 选出的最优特征子集：")
    print(f"最优特征子集大小: {len(best_features)}")
    print(f"ACO 交叉验证得到的最佳 F1-Score: {best_cv_f1:.4f}")

    # 3. 在测试集上评估最终性能
    print("\n[C] 在测试集上评估最终性能")
    eval_start_time = time.time()
    
    X_train_final = X_train_full[best_features]
    X_test_final = X_test_full[best_features]
    
    # 训练最终模型 (使用更强的参数)
    final_params = current_xgb_params.copy()
    final_params['n_estimators'] = 200 # 最终评估使用更多树
    final_params['max_depth'] = 8       # 最终评估使用更深树
    
    xgb_model_final = XGBClassifier(**final_params)
    xgb_model_final.fit(X_train_final, y_train)
    
    y_pred_final = xgb_model_final.predict(X_test_final)
    
    # 接收所有 4 个指标 (Acc, F1, Prec, Rec)
    acc_final, f1_final, prec_final, rec_final = evaluate_performance(
        y_test, y_pred_final, subset_size=len(best_features)
    )
    eval_time = time.time() - eval_start_time

    total_time = time.time() - total_start_time # <<< 计时结束

    # 打印最终报告
    print("\n" + "=" * 70)
    print("ACO-XGBoost 混合基线 - 性能报告")
    print("=" * 70)
    print(f"时间统计:")
    print(f"  数据加载: {data_load_time:.2f}秒")
    print(f"  ACO 特征选择: {aco_time:.2f}秒")
    print(f"  最终评估: {eval_time:.2f}秒")
    print(f"  总耗时: {total_time:.2f}秒")
    print("\n特征统计:")
    print(f"  原始特征数: {X_train_full.shape[1]}")
    print(f"  最终特征数: {len(best_features)}")
    print(f"  特征压缩率: {(1 - len(best_features)/X_train_full.shape[1])*100:.1f}%")
    print("\n性能指标:")
    print(f"  最终测试准确率: {acc_final:.4f}")
    print(f"  最终测试F1-Score: {f1_final:.4f}")
    print(f"  最终测试Precision: {prec_final:.4f}")
    print(f"  最终测试Recall: {rec_final:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    run_hybrid_aco_baseline()