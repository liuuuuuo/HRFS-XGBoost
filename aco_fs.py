# aco_fs.py (ACO-XGBoost 特征选择类结构模板)

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm

class ACO_XGBoost_FS:
    """
    蚁群优化算法结合 XGBoost 的特征选择器。
    """
    def __init__(self, X, y, xgb_params, n_ants=10, max_iter=10, alpha=1.0, beta=1.0, rho=0.1, target_feature_count=20):
        self.X = X
        # 确保 y 是 pd.Series
        self.y = pd.Series(y).reset_index(drop=True)
        self.xgb_params = xgb_params
        self.n_features = X.shape[1]
        self.feature_names = X.columns.tolist()
        
        # ACO 参数
        self.n_ants = n_ants
        self.max_iter = max_iter
        self.alpha = alpha  # 信息素重要性
        self.beta = beta    # 启发式信息重要性
        self.rho = rho      # 信息素挥发率
        self.target_k = target_feature_count
        
        # 初始化信息素矩阵和启发式信息
        self.pheromone = np.ones(self.n_features) * 0.1
        self.heuristic = self._calculate_initial_heuristic()


    def _calculate_initial_heuristic(self):
        """计算初始启发式信息 (XGBoost Gain)"""
        
        lite_params = self.xgb_params.copy()
        lite_params['n_estimators'] = 50
        lite_params['max_depth'] = 4
        
        xgb_model = XGBClassifier(**lite_params)
        xgb_model.fit(self.X, self.y)
        
        gain_scores = xgb_model.get_booster().get_score(importance_type='gain')
        
        heuristic = np.zeros(self.n_features)
        
        for i, name in enumerate(self.feature_names):
            heuristic[i] = gain_scores.get(name, 0)
            
        max_h = np.max(heuristic)
        if max_h > 0:
            heuristic = heuristic / max_h
            
        # 避免启发式信息为 0
        heuristic[heuristic == 0] = 0.01 
        
        return heuristic
        

    def _evaluate_subset(self, features):
        """
        使用 3 折交叉验证评估特征子集的性能 (F1-Score)。
        """
        if len(features) < 2:
            return 0.0

        X_sub = self.X[features]
        
        # 使用 3-Fold 交叉验证
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        f1_scores = []
        
        # 使用轻量级 XGBoost 参数进行评估加速
        eval_params = self.xgb_params.copy()
        eval_params['n_estimators'] = 50
        eval_params['max_depth'] = 4
        
        for train_index, val_index in kf.split(X_sub, self.y):
            X_train, X_val = X_sub.iloc[train_index], X_sub.iloc[val_index]
            y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]
            
            try:
                model = XGBClassifier(**eval_params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
                f1_scores.append(f1)
            except Exception as e:
                f1_scores.append(0.0)

        return np.mean(f1_scores) if f1_scores else 0.0


    def _select_features_by_ant(self, start_feature_index):
        """
        蚂蚁选择 K 个特征。
        """
        selected_indices = [start_feature_index]
        
        for _ in range(self.target_k - 1):
            
            unselected_indices = [i for i in range(self.n_features) if i not in selected_indices]
            if not unselected_indices:
                break
                
            probabilities = np.zeros(self.n_features)
            
            for i in unselected_indices:
                tau_i = self.pheromone[i]
                eta_i = self.heuristic[i]
                # ACO 核心公式
                probabilities[i] = (tau_i ** self.alpha) * (eta_i ** self.beta)
                
            prob_sum = np.sum(probabilities[unselected_indices])
            
            if prob_sum == 0:
                next_idx = np.random.choice(unselected_indices)
            else:
                normalized_probs = probabilities[unselected_indices] / prob_sum
                
                # 轮盘赌选择
                next_idx = np.random.choice(unselected_indices, p=normalized_probs)
                
            selected_indices.append(next_idx)
            
        selected_features = [self.feature_names[i] for i in selected_indices]
        return selected_features, selected_indices


    def _update_pheromone(self, best_features_indices, best_f1):
        """
        信息素挥发和沉积。
        """
        # 1. 信息素挥发
        self.pheromone = (1 - self.rho) * self.pheromone
        
        # 2. 信息素沉积 (在最佳路径上沉积)
        Q = best_f1 * 10 
        
        for idx in best_features_indices:
            self.pheromone[idx] += Q
            
        # 信息素限制
        self.pheromone[self.pheromone < 0.001] = 0.001
        self.pheromone[self.pheromone > 100] = 100


    def run(self):
        """
        运行 ACO 特征选择主循环。
        """
        best_features_global = []
        best_f1_global = 0.0
        
        pbar = tqdm(total=self.max_iter, desc="  ACO 迭代优化")
        
        for iteration in range(self.max_iter):
            ant_features_list = []
            ant_f1_list = []
            
            # 1. 每只蚂蚁选择路径
            start_features = np.random.choice(self.n_features, size=self.n_ants, replace=False)
            
            for ant_id, start_idx in enumerate(start_features):
                features, indices = self._select_features_by_ant(start_idx)
                f1 = self._evaluate_subset(features)
                
                ant_features_list.append((features, indices))
                ant_f1_list.append(f1)
                
                if f1 > best_f1_global:
                    best_f1_global = f1
                    best_features_global = features
                    
            # 2. 更新信息素
            best_ant_idx = np.argmax(ant_f1_list)
            _, best_ant_indices = ant_features_list[best_ant_idx]
            best_ant_f1 = ant_f1_list[best_ant_idx]
            
            self._update_pheromone(best_ant_indices, best_ant_f1)
            
            pbar.update(1)
            pbar.set_postfix({'最佳 F1': f"{best_f1_global:.4f}", '当前迭代最佳 F1': f"{best_ant_f1:.4f}"})
            
        pbar.close()
        return best_features_global, best_f1_global