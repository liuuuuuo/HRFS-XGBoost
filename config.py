# config.py

# NSL-KDD 原始的 41 个特征名 + 标签
FEATURE_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 
    'attack_type', 'level' # 标签和冗余列
]

# NSL-KDD 中的类别特征（需要 One-Hot 编码）
CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']

# 标签列名
LABEL_COLUMN = 'attack_type'

# XGBoost 模型超参数 (用于最终评估的初始参数)
XGB_PARAMS = {
    'objective': 'multi:softmax',
    'eval_metric': 'mlogloss',
    'n_estimators': 200,           # 增加树的数量，提供更强的学习能力
    'max_depth': 8,                # 增加深度，捕获更复杂模式
    'learning_rate': 0.1,          # 适中学习率
    'subsample': 0.8,              # 行采样，防止过拟合
    'colsample_bytree': 0.8,       # 列采样，增加多样性
    'min_child_weight': 3,         # 控制过拟合，对不平衡数据很重要
    'gamma': 0.2,                  # 分裂的最小损失减少，防止过拟合
    'reg_alpha': 0.1,              # L1正则化
    'reg_lambda': 1.0,             # L2正则化
    'random_state': 42,
    'verbosity': 0,
    'use_label_encoder': False,
    'n_jobs': -1                   # 使用所有CPU核心加速
}

# ==================== HRFS 算法参数 (新增和修正) ====================
# 特征选择基线设置
TOP_K_FEATURES = 20  # 最终选取的特征数 K
TOP_M_FEATURES = 40  # HRFS的 Gain 预筛选池大小 M 

# 阶段1 冗余度惩罚阈值 (Corr > TAU_REDUNDANCY 视为高度相关)
TAU_REDUNDANCY = 0.7 

# 阶段2 局部搜索窗口 (S_Pre + 额外纳入考虑的 Gain 排名靠前的未选特征数)
LOCAL_WINDOW = 5

# 阶段2 局部搜索最大迭代次数 (限制速度的关键)
MAX_ITERATIONS = 5