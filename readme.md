# NSL-KDD Intrusion Detection with HRFS-XGBoost

This project implements a **Hybrid Robust Feature Selection (HRFS) algorithm** combined with XGBoost for network intrusion detection using the NSL-KDD dataset. The system includes multiple baseline methods (embedded feature selection and ACO-XGBoost) and compares them against the proposed HRFS-XGBoost approach.

## 📁 Project Structure

```
HRFS_XGBoost_Project/
│
├── data/
│   ├── KDDTrain+.txt        # NSL-KDD training set
│   └── KDDTest+.txt         # NSL-KDD test set
│
├── config.py                # Configuration: feature names, paths, hyperparameters
├── data_loader.py           # Data processing: loading, cleaning, encoding, normalization
├── utils.py                 # Utility functions: performance evaluation, visualization
├── aco_fs.py                # ACO feature selection implementation
├── display_fresh.py         # Visualization module with fresh color schemes
├── 1_baseline_embedded.py   # Phase 1: Pure embedded XGBoost baseline
├── 2_baseline_hybrid.py     # Phase 2: ACO-XGBoost hybrid baseline
├── 3_hrfs_xgboost.py        # Phase 3: HRFS-XGBoost algorithm (main)
└── README.md                # This file
```

## 📊 Dataset

The project uses the **NSL-KDD** dataset for network intrusion detection. You can download it from:

[NSL-KDD Dataset Download Link](https://gitcode.com/Premium-Resources/b4986/?utm_source=article_gitcode_universal&index=top&type=card&uuid_tt_dd=10_21307064330-1752202919931-952831&from_id=147697006&from_link=439f753f3a18277b4d3b256ea48b2176)

Place the dataset files (`KDDTrain+.txt` and `KDDTest+.txt`) in the `data/` directory.

## 🚀 Installation & Requirements

### Python Dependencies

```bash
pip install pandas numpy scikit-learn xgboost matplotlib tqdm
```

### Required Libraries
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- tqdm

## 🧩 Module Descriptions

### 1. **config.py**
Configuration file containing:
- Feature names for NSL-KDD (41 features)
- Categorical feature definitions
- XGBoost hyperparameters
- HRFS algorithm parameters (τ, K, M, window sizes, etc.)

### 2. **data_loader.py**
Data loading and preprocessing module:
- Loads NSL-KDD training and test sets
- Handles categorical feature encoding (One-Hot)
- Applies feature scaling (StandardScaler)
- Manages label encoding with factorize for consistent class indices

### 3. **utils.py**
Utility functions:
- Performance evaluation (Accuracy, F1-Score, Precision, Recall)
- Macro-averaged metrics calculation
- Verbose output formatting

### 4. **aco_fs.py**
Ant Colony Optimization (ACO) feature selection:
- Implements ACO algorithm for feature selection
- Uses pheromone trails and heuristic information
- Evaluates feature subsets with cross-validation

### 5. **1_baseline_embedded.py**
**Baseline 1**: Pure embedded XGBoost feature selection
- Trains XGBoost on full feature set
- Selects top K features based on Gain importance
- Evaluates performance on reduced feature set

### 6. **2_baseline_hybrid.py**
**Baseline 2**: ACO-XGBoost hybrid feature selection
- Combines ACO optimization with XGBoost
- Uses ACO to search for optimal feature subsets
- Includes timing and performance metrics

### 7. **3_hrfs_xgboost.py**
**Main Algorithm**: HRFS-XGBoost (Hybrid Robust Feature Selection)
- **Phase 1**: Greedy selection with I_RC metric (redundancy-calibrated)
- **Phase 2**: Local greedy search optimization
- Features data sampling for computational efficiency
- Returns all performance metrics (Accuracy, F1, Precision, Recall)

### 8. **display_fresh.py**
Visualization module:
- Creates comparison charts for different algorithms
- Uses fresh, academic-friendly color schemes
- Generates performance metrics and timing comparison plots

## ⚙️ Key Algorithm Parameters (config.py)

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `TOP_K_FEATURES` | Final number of features to select | 20 |
| `TOP_M_FEATURES` | Phase 1 pre-selection pool size | 40 |
| `TAU_REDUNDANCY` | Redundancy threshold (correlation > τ) | 0.7 |
| `LOCAL_WINDOW` | Local search window size | 5 |
| `MAX_ITERATIONS` | Maximum local search iterations | 5 |
| `XGB_PARAMS` | XGBoost hyperparameters | See config.py |

## 🏃‍♂️ How to Run

### 1. Data Preparation
Ensure the NSL-KDD dataset files are in the `data/` directory.

### 2. Run Baseline 1 (Embedded)
```bash
python 1_baseline_embedded.py
```

### 3. Run Baseline 2 (ACO-XGBoost)
```bash
python 2_baseline_hybrid.py
```

### 4. Run Main Algorithm (HRFS-XGBoost)
```bash
python 3_hrfs_xgboost.py
```

### 5. Generate Visualizations
```bash
python display_fresh.py
```

## 📈 Performance Metrics

All algorithms evaluate the following metrics:
- **Accuracy** (Acc)
- **F1-Score** (macro-averaged)
- **Precision** (macro-averaged)
- **Recall** (macro-averaged)
- **Feature compression rate**
- **Execution time**

## 🔧 Algorithm Details

### HRFS-XGBoost (Proposed Method)

**Phase 1: Redundancy-Calibrated Feature Screening**
- Uses XGBoost Gain scores as relevance measure
- Applies redundancy penalty based on feature correlations
- Selects M features using I_RC score: `Gain × (1 / (1 + Redundancy))`
- Implements data sampling for computational efficiency

**Phase 2: Local Greedy Search Optimization**
- Creates local search space from pre-selected features + top Gain features
- Performs iterative add/remove operations
- Uses early stopping based on performance improvement
- Evaluates feature subsets with lightweight XGBoost models

### Key Features:
- **Robustness**: Handles class imbalance in intrusion detection
- **Efficiency**: Implements data sampling strategies
- **Interpretability**: Uses XGBoost feature importance
- **Flexibility**: Configurable parameters for different scenarios

## 📊 Expected Results

Based on the implementation, HRFS-XGBoost should demonstrate:
1. **Higher F1-Score** compared to baseline methods
2. **Better handling of feature redundancy**
3. **Reasonable computational time** compared to ACO-based methods
4. **Improved precision and recall** for minority attack classes

## 🔍 Troubleshooting

1. **FileNotFoundError**: Ensure NSL-KDD files are in `data/` directory
2. **Memory Issues**: Reduce `sample_ratio` parameters in `3_hrfs_xgboost.py`
3. **XGBoost Warnings**: Ignore label encoder warnings (handled in code)
4. **Visualization Issues**: Install Chinese fonts or modify `display_fresh.py`

## 📚 References

1. Tavallaee, M., et al. (2009). "A detailed analysis of the KDD CUP 99 data set"
2. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
3. Dorigo, M., et al. (1996). "Ant system: optimization by a colony of cooperating agents"

## 👥 Authors

- Liu
- Jiang
- Wang

## 📄 License


This project is for academic research purposes.
