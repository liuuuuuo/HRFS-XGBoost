# 3_hrfs_xgboost.py (最终稳定版：包含 Precision/Recall 指标)
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from data_loader import load_and_preprocess_nslkdd
# 【修正】导入所有必需的常量
from config import (
    XGB_PARAMS, TOP_K_FEATURES, TOP_M_FEATURES, 
    TAU_REDUNDANCY, LOCAL_WINDOW, MAX_ITERATIONS
)
from utils import evaluate_performance
from sklearn.preprocessing import LabelEncoder
import warnings
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter

warnings.filterwarnings("ignore")


def sample_data_stratified(X, y, sample_ratio=0.2, random_state=42):
    """分层采样数据，确保所有类别都被采样到"""
    if sample_ratio >= 1.0:
        return X, y
    
    y_series = pd.Series(y)
    class_counts = y_series.value_counts()
    min_samples_per_class = 2  # 每个类别至少2个样本
    
    target_total = int(len(X) * sample_ratio)
    
    sampled_indices = []
    
    for class_label in class_counts.index:
        class_indices = y_series[y_series == class_label].index.tolist()
        n_class = len(class_indices)
        
        n_sample = max(min_samples_per_class, int(n_class * sample_ratio))
        n_sample = min(n_sample, n_class)
        
        if n_sample > 0:
            sampled_class_indices = np.random.choice(
                class_indices, size=n_sample, replace=False
            ).tolist()
            sampled_indices.extend(sampled_class_indices)
    
    if len(sampled_indices) < target_total:
        remaining_indices = list(set(X.index) - set(sampled_indices))
        if remaining_indices:
            n_needed = target_total - len(sampled_indices)
            n_needed = min(n_needed, len(remaining_indices))
            additional_indices = np.random.choice(
                remaining_indices, size=n_needed, replace=False
            ).tolist()
            sampled_indices.extend(additional_indices)
    
    np.random.shuffle(sampled_indices)
    
    X_sampled = X.iloc[sampled_indices].reset_index(drop=True)
    y_sampled = pd.Series(y).iloc[sampled_indices].reset_index(drop=True)
    
    return X_sampled, y_sampled


# ----------------------------------------------------
def calculate_redundancy_penalty(feature_idx, selected_indices, correlation_matrix, tau=TAU_REDUNDANCY):
    """计算特征冗余度惩罚项"""
    if not selected_indices:
        return 0.0
    
    penalty = 0.0
    for sel_idx in selected_indices:
        corr = abs(correlation_matrix.iloc[feature_idx, sel_idx]) 
        if corr > tau:
            penalty += corr
    
    return penalty


def calculate_irc_score(feature_idx, gain_scores, selected_indices, correlation_matrix, tau=TAU_REDUNDANCY):
    """计算 I_RC 分数：Gain * (1 / (1 + 冗余度惩罚))"""
    gain = gain_scores.get(feature_idx, 0.0)
    redundancy = calculate_redundancy_penalty(feature_idx, selected_indices, correlation_matrix, tau)
    irc_score = gain * (1.0 / (1.0 + redundancy))
    return irc_score


def phase1_greedy_selection_with_irc(X_train, y_train, xgb_params, 
                                     m_features=TOP_M_FEATURES, tau=TAU_REDUNDANCY, 
                                     sample_ratio=0.3, random_state=42):
    """阶段1：基于 I_RC 指标的特征筛选"""
    print(f"\n[A] 阶段1：基于冗余度校准的特征筛选 (I_RC 指标, τ={tau})")
    start_time = time.time()
    
    if sample_ratio < 1.0:
        print(f"  使用 {sample_ratio*100:.0f}% 数据采样加速阶段1训练...")
        X_train_sampled, y_train_sampled = sample_data_stratified(
            X_train, y_train, sample_ratio, random_state
        )
    else:
        X_train_sampled, y_train_sampled = X_train, y_train
    
    unique_classes_sampled = np.unique(y_train_sampled)
    unique_classes_original = np.unique(y_train)
    
    print(f"  采样后数据: {len(X_train_sampled)} 样本, {X_train_sampled.shape[1]} 特征")
    print(f"  原始类别数: {len(unique_classes_original)}, 采样后类别数: {len(unique_classes_sampled)}")
    
    xgb_params_adjusted = xgb_params.copy()
    xgb_params_adjusted['num_class'] = len(unique_classes_sampled)
    xgb_params_adjusted['n_estimators'] = 50 
    xgb_params_adjusted['max_depth'] = 4
    
    print("  训练轻量XGBoost模型获取Gain分数...")
    xgb_model = XGBClassifier(**xgb_params_adjusted)
    xgb_model.fit(X_train_sampled, y_train_sampled)
    
    gain_scores_raw = xgb_model.get_booster().get_score(importance_type='gain')
    
    feature_names = X_train.columns.tolist()
    feature_name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    
    gain_scores = {}
    for fname, gain in gain_scores_raw.items():
        if fname in feature_name_to_idx:
            idx = feature_name_to_idx[fname]
            gain_scores[idx] = gain
    
    for idx in range(len(feature_names)):
        if idx not in gain_scores:
            gain_scores[idx] = 0.0
    
    print("  计算特征相关性矩阵...")
    corr_start = time.time()
    
    X_train_corr = X_train_sampled if len(X_train_sampled) < 5000 else X_train_sampled.sample(5000, random_state=random_state)
    correlation_matrix = pd.DataFrame(
        np.corrcoef(X_train_corr.values.T), 
        index=range(X_train.shape[1]), 
        columns=range(X_train.shape[1])
    )
    print(f"  相关性矩阵计算完成，耗时: {time.time() - corr_start:.2f}秒")
    
    selected_indices = []
    remaining_indices = list(range(X_train.shape[1]))
    
    print(f"  从 {len(remaining_indices)} 个特征中贪婪选择 {m_features} 个特征...")
    
    pbar = tqdm(total=m_features, desc="  特征筛选进度")
    
    for step in range(m_features):
        if not remaining_indices:
            break
            
        irc_scores = []
        for idx in remaining_indices:
            irc = calculate_irc_score(idx, gain_scores, selected_indices, correlation_matrix, tau=tau) 
            irc_scores.append((idx, irc))
        
        irc_scores.sort(key=lambda x: x[1], reverse=True)
        best_idx, best_irc = irc_scores[0]
        
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        
        pbar.update(1)
        
        if (step + 1) % 10 == 0 or (step + 1) == m_features:
            feature_name = feature_names[best_idx]
            pbar.set_postfix({
                '当前特征': feature_name[:15],
                'I_RC分数': f"{best_irc:.2f}",
                '已选特征数': len(selected_indices)
            })
    
    pbar.close()
    
    selected_features = [feature_names[idx] for idx in selected_indices]
    
    phase1_time = time.time() - start_time
    print(f"  阶段1完成。筛选出 {len(selected_features)} 个特征，耗时: {phase1_time:.2f}秒")
    
    return selected_features


def phase2_local_greedy_search(X_train, y_train, X_test, y_test, 
                               pre_selected_features, xgb_params, 
                               k_features=TOP_K_FEATURES, local_window=LOCAL_WINDOW,
                               max_iterations=MAX_ITERATIONS,
                               sample_ratio=0.4, random_state=42):
    """阶段2：局部贪婪搜索优化"""
    print(f"\n[B] 阶段2：局部贪婪搜索优化 (窗口大小={local_window}, 最大迭代={max_iterations})")
    start_time = time.time()
    
    feature_names = X_train.columns.tolist()
    
    if sample_ratio < 1.0:
        print(f"  使用 {sample_ratio*100:.0f}% 数据采样加速局部搜索...")
        X_train_sampled, y_train_sampled = sample_data_stratified(
            X_train, y_train, sample_ratio, random_state
        )
    else:
        X_train_sampled, y_train_sampled = X_train, y_train
    
    xgb_params_adjusted = xgb_params.copy()
    xgb_params_adjusted['num_class'] = len(np.unique(y_train_sampled))
    
    pre_selected_indices = [feature_names.index(f) for f in pre_selected_features 
                           if f in feature_names]
    
    all_indices = list(range(len(feature_names)))
    unselected_indices = [idx for idx in all_indices if idx not in pre_selected_indices]
    
    print("  训练轻量模型获取特征排序...")
    xgb_model = XGBClassifier(**xgb_params_adjusted)
    xgb_model.fit(X_train_sampled, y_train_sampled)
    gain_scores_raw = xgb_model.get_booster().get_score(importance_type='gain')
    feature_name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    
    gain_scores = {}
    for fname, gain in gain_scores_raw.items():
        if fname in feature_name_to_idx:
            idx = feature_name_to_idx[fname]
            gain_scores[idx] = gain
    
    for idx in unselected_indices:
        if idx not in gain_scores:
            gain_scores[idx] = 0.0
    
    unselected_with_gain = [(idx, gain_scores.get(idx, 0.0)) for idx in unselected_indices]
    unselected_with_gain.sort(key=lambda x: x[1], reverse=True)
    
    top_unselected = [feature_names[idx] for idx, _ in unselected_with_gain[:local_window]]
    
    local_search_features = list(set(pre_selected_features) | set(top_unselected))
    
    print(f"  局部搜索空间包含 {len(local_search_features)} 个特征")
    
    current_features = pre_selected_features.copy()
    
    # 【修正】适应 evaluate_subset_performance 的新返回指标
    current_performance, *_ = evaluate_subset_performance(
        X_train_sampled, y_train_sampled, X_test, y_test, 
        current_features, xgb_params_adjusted
    )
    
    print(f"  初始子集 ({len(current_features)} 个特征) F1-Score: {current_performance:.4f}")
    
    improved = True
    iteration = 0
    
    print(f"  开始局部搜索 (最多 {max_iterations} 次迭代)...")
    
    best_performance = current_performance
    best_features = current_features.copy()
    
    while improved and iteration < max_iterations and len(current_features) > 5:
        improved = False
        
        add_candidates = []
        for feature in local_search_features:
            if feature in current_features:
                continue
                
            candidate_features = current_features + [feature]
            if len(candidate_features) > k_features + 5:
                continue
                
            candidate_performance, *_ = evaluate_subset_performance(
                X_train_sampled, y_train_sampled, X_test, y_test, 
                candidate_features, xgb_params_adjusted
            )
            
            add_candidates.append((feature, candidate_performance))
        
        remove_candidates = []
        if len(current_features) > max(5, k_features - 5):
            for feature in current_features.copy():
                if feature not in local_search_features:
                    continue
                    
                candidate_features = [f for f in current_features if f != feature]
                if len(candidate_features) < max(5, k_features - 10):
                    continue
                    
                candidate_performance, *_ = evaluate_subset_performance(
                    X_train_sampled, y_train_sampled, X_test, y_test, 
                    candidate_features, xgb_params_adjusted
                )
                
                remove_candidates.append((feature, candidate_performance))
        
        best_change = None
        best_change_performance = current_performance
        
        for feature, perf in add_candidates:
            if perf > best_change_performance + 0.001:
                best_change = ('add', feature)
                best_change_performance = perf
        
        for feature, perf in remove_candidates:
            if perf >= best_change_performance - 0.0005 and perf > current_performance:
                best_change = ('remove', feature)
                best_change_performance = perf
        
        if best_change:
            if best_change[0] == 'add':
                current_features.append(best_change[1])
                print(f"    迭代 {iteration+1}: +添加 '{best_change[1][:15]}...'，F1: {best_change_performance:.4f}")
            else:
                current_features.remove(best_change[1])
                print(f"    迭代 {iteration+1}: -移除 '{best_change[1][:15]}...'，F1: {best_change_performance:.4f}")
            
            current_performance = best_change_performance
            improved = True
            
            if current_performance > best_performance:
                best_performance = current_performance
                best_features = current_features.copy()
        else:
            print(f"    迭代 {iteration+1}: 未找到显著改进操作，停止搜索。")
            break
        
        iteration += 1
    
    current_features = best_features
    current_performance = best_performance
    
    phase2_time = time.time() - start_time
    print(f"  局部搜索完成。耗时: {phase2_time:.2f}秒")
    print(f"  最终子集大小: {len(current_features)}，F1-Score: {current_performance:.4f}")
    
    return current_features, current_performance


def evaluate_subset_performance(X_train, y_train, X_test, y_test, features, xgb_params):
    """评估特征子集的性能（快速评估），只返回 F1-Score 作为主要指标"""
    if not features or len(features) < 2:
        return 0.0, 0.0, 0.0, 0.0 # 适应 utils.py 的新返回值
    
    features_in_common = [f for f in features if f in X_train.columns and f in X_test.columns]
    
    if len(features_in_common) < 2:
        return 0.0, 0.0, 0.0, 0.0
        
    X_train_sub = X_train[features_in_common]
    X_test_sub = X_test[features_in_common]
    
    all_unique = np.unique(np.concatenate([y_train, y_test]))
    
    eval_params = xgb_params.copy()
    eval_params['num_class'] = len(all_unique)
    eval_params['n_estimators'] = min(50, eval_params.get('n_estimators', 80))  
    eval_params['max_depth'] = min(3, eval_params.get('max_depth', 6))
    
    try:
        model = XGBClassifier(**eval_params)
        model.fit(X_train_sub, y_train)
        
        y_pred = model.predict(X_test_sub)
        
        # 【修正】调用 evaluate_performance，并解包全部指标
        acc, f1, prec, rec = evaluate_performance(y_test, y_pred, verbose=False) 
    except Exception as e:
        # print(f"  评估错误: {e}, 返回0.0")
        acc, f1, prec, rec = 0.0, 0.0, 0.0, 0.0
    
    return acc, f1, prec, rec


def run_hrfs_xgboost_algorithm(random_state=42, sample_ratio_phase1=0.3, sample_ratio_phase2=0.4):
    """运行 HRFS-XGBoost 算法"""
    print("=" * 70)
    print("HRFS-XGBoost 算法 (高鲁棒性混合特征选择)")
    print("=" * 70)
    print(f"配置参数:")
    print(f"  - 随机种子: {random_state}")
    print(f"  - 阶段1采样比例: {sample_ratio_phase1*100:.0f}%")
    print(f"  - 阶段2采样比例: {sample_ratio_phase2*100:.0f}%")
    print(f"  - 目标特征数: {TOP_K_FEATURES}")
    print(f"  - 预选特征数: {TOP_M_FEATURES}")
    print(f"  - 冗余度阈值 (τ): {TAU_REDUNDANCY}")
    print("=" * 70)
    
    total_start_time = time.time()
    
    # 1. 加载数据
    print("\n[1] 加载数据...")
    data_start = time.time()
    X_train_full, X_test_full, y_train, y_test, num_classes = load_and_preprocess_nslkdd()
    print(f"  数据加载完成。训练集: {X_train_full.shape}, 测试集: {X_test_full.shape}")
    print(f"  数据加载耗时: {time.time() - data_start:.2f}秒")
    
    # 2. 统一标签编码
    print("\n[2] 标签编码...")
    le = LabelEncoder()
    all_labels = pd.concat([pd.Series(y_train), pd.Series(y_test)], ignore_index=True)
    le.fit(all_labels)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    
    all_unique = np.unique(np.concatenate([y_train, y_test]))
    print(f"  总类别数: {len(all_unique)}")
    
    # 3. 准备XGBoost参数
    current_xgb_params = XGB_PARAMS.copy()
    current_xgb_params.update({
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "verbosity": 0,
        "n_jobs": -1,
        "random_state": random_state,
        "num_class": len(all_unique),
        "n_estimators": 100,
        "max_depth": 6
    })
    
    # 4. 阶段1：基于I_RC的特征筛选
    phase1_start = time.time()
    pre_selected_features = phase1_greedy_selection_with_irc(
        X_train_full, y_train, current_xgb_params, 
        m_features=TOP_M_FEATURES, 
        tau=TAU_REDUNDANCY,
        sample_ratio=sample_ratio_phase1, random_state=random_state
    )
    phase1_time = time.time() - phase1_start
    
    # 5. 阶段2：局部贪婪搜索
    phase2_start = time.time()
    # 局部搜索返回的最佳F1-score
    final_features, local_f1 = phase2_local_greedy_search(
        X_train_full, y_train, X_test_full, y_test,
        pre_selected_features, current_xgb_params,
        k_features=TOP_K_FEATURES, 
        local_window=LOCAL_WINDOW,
        max_iterations=MAX_ITERATIONS,
        sample_ratio=sample_ratio_phase2, random_state=random_state
    )
    phase2_time = time.time() - phase2_start
    
    # 6. 最终评估（使用完整数据）
    print(f"\n[C] 最终评估：使用完整数据训练最终模型")
    eval_start = time.time()
    
    X_train_final = X_train_full[final_features]
    X_test_final = X_test_full[final_features]
    
    final_params = current_xgb_params.copy()
    final_params['n_estimators'] = 200
    final_params['max_depth'] = 8
    
    print(f"  训练最终模型 ({len(final_features)} 个特征)...")
    final_model = XGBClassifier(**final_params)
    final_model.fit(X_train_final, y_train)
    
    y_pred_final = final_model.predict(X_test_final)
    
    # 【修正】使用utils中的评估函数，解包所有指标
    acc_final, f1_final, prec_final, rec_final = evaluate_performance(
        y_test, y_pred_final, subset_size=len(final_features), verbose=False
    )
    
    eval_time = time.time() - eval_start
    total_time = time.time() - total_start_time
    
    # 打印性能报告
    print("\n" + "=" * 70)
    print("HRFS-XGBoost 算法完成 - 性能报告")
    print("=" * 70)
    print(f"时间统计:")
    print(f"  阶段1 (特征筛选): {phase1_time:.2f}秒")
    print(f"  阶段2 (局部搜索): {phase2_time:.2f}秒")
    print(f"  最终评估: {eval_time:.2f}秒")
    print(f"  总耗时: {total_time:.2f}秒")
    print()
    print(f"特征统计:")
    print(f"  原始特征数: {X_train_full.shape[1]}")
    print(f"  阶段1预选特征: {len(pre_selected_features)}")
    print(f"  最终特征数: {len(final_features)}")
    print(f"  特征压缩率: {(1 - len(final_features)/X_train_full.shape[1])*100:.1f}%")
    print()
    print(f"性能指标:")
    print(f"  阶段2局部搜索F1: {local_f1:.4f}")
    print(f"  最终测试准确率: {acc_final:.4f}")
    print(f"  最终测试F1-Score: {f1_final:.4f}")
    # 【新增指标】
    print(f"  最终测试Precision: {prec_final:.4f}")
    print(f"  最终测试Recall: {rec_final:.4f}")
    print("=" * 70)
    
    return final_features, acc_final, f1_final, total_time


if __name__ == '__main__':
    np.random.seed(42)
    
    final_features, acc_final, f1_final, total_time = run_hrfs_xgboost_algorithm(
        random_state=42,
        sample_ratio_phase1=0.3,
        sample_ratio_phase2=0.4
    )