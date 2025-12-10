# display_fresh.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.font_manager as fm

# -------------- 1. 中文字体 --------------
try:
    font_path = fm.findfont('SimHei', fallback_to_default=True)
    plt.rcParams['font.sans-serif'] = [os.path.basename(font_path).split('.')[0]]
except:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# -------------- 2. 数据 --------------
PERFORMANCE_DATA = {
    '模型': ['纯嵌入式 (Top 20)', 'ACO-XGBoost (K=20)', 'HRFS-XGBoost (K=38)'],
    '特征数': [20, 20, 38],
    'F1-Score': [0.2047, 0.2099, 0.2436],
    '准确率 (Acc)': [0.7102, 0.7152, 0.7221],
    '精确率 (Prec)': [0.1865, 0.2203, 0.2535],
    '召回率 (Rec)': [0.2624, 0.2656, 0.2813]
}
TIME_DATA = {
    '模型': ['纯嵌入式 (Top 20)', 'ACO-XGBoost', 'HRFS-XGBoost'],
    '总耗时 (秒)': [25.0, 1235.62, 602.15]
}

# -------------- 3. 清新莫兰迪配色 --------------
# 低饱和 + 高明度，适合学术图
FRESH = {
    'f1':   '#A8D8EA',   # 雾蓝
    'prec': '#FFAAA7',   # 雾粉
    'rec':  '#B0E0A8',   # 雾绿
    'acc1': '#D8BFD8',   # 雾紫
    'acc2': '#FFE4C4',   # 雾杏
    'acc3': '#FFF9C4',   # 雾柠檬
    'time1':'#FFCCAA',   # 雾珊瑚
    'time2':'#AAE6D8'    # 雾青
}

# -------------- 4. 绘图函数 --------------
def plot_performance_metrics(data, save='performance_f1_prec_rec.png'):
    df = pd.DataFrame(data)
    models = df['模型']
    metrics = ['F1-Score', '精确率 (Prec)', '召回率 (Rec)']
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, df[metrics[0]], width, label=metrics[0], color=FRESH['f1'],  edgecolor='white', lw=1.2)
    rects2 = ax.bar(x,        df[metrics[1]], width, label=metrics[1], color=FRESH['prec'], edgecolor='white', lw=1.2)
    rects3 = ax.bar(x + width, df[metrics[2]], width, label=metrics[2], color=FRESH['rec'],  edgecolor='white', lw=1.2)

    ax.set_ylabel('宏平均指标值', fontsize=12)
    ax.set_title('不同特征选择算法的宏平均性能指标对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(framealpha=0)
    ax.set_facecolor('#FAFBFC')
    fig.patch.set_facecolor('#FAFBFC')
    plt.grid(axis='y', ls='--', lw=0.8, color='#E0E0E0')

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, h+0.002, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=9, color='#555555')
    autolabel(rects1); autolabel(rects2); autolabel(rects3)
    fig.tight_layout()
    plt.savefig(save, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f'已保存 → {save}')

def plot_accuracy_comparison(data, save='accuracy_comparison.png'):
    df = pd.DataFrame(data)
    models = df['模型']
    acc = df['准确率 (Acc)']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, acc, color=[FRESH['acc1'], FRESH['acc2'], FRESH['acc3']], edgecolor='white', lw=1.2)
    ax.set_ylabel('准确率 (Accuracy)', fontsize=12)
    ax.set_title('不同特征选择算法的准确率对比', fontsize=14, fontweight='bold')
    ax.set_ylim(0.705, 0.725)
    ax.set_facecolor('#FAFBFC')
    fig.patch.set_facecolor('#FAFBFC')
    plt.grid(axis='y', ls='--', lw=0.8, color='#E0E0E0')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h+0.0005, f'{h:.4f}',
                ha='center', va='bottom', fontsize=10, color='#555555')
    fig.tight_layout()
    plt.savefig(save, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f'已保存 → {save}')

def plot_time_comparison(data, save='time_comparison.png'):
    df = pd.DataFrame(data)
    df_plot = df[df['总耗时 (秒)'] > 100].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df_plot['模型'], df_plot['总耗时 (秒)'],
                  color=[FRESH['time1'], FRESH['time2']], edgecolor='white', lw=1.2)
    ax.set_ylabel('总耗时 (秒)', fontsize=12)
    ax.set_title('特征选择算法总运行时间对比', fontsize=14, fontweight='bold')
    ax.set_facecolor('#FAFBFC')
    fig.patch.set_facecolor('#FAFBFC')
    plt.grid(axis='y', ls='--', lw=0.8, color='#E0E0E0')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h+20, f'{h:.2f}',
                ha='center', va='bottom', fontsize=10, color='#555555')
    embedded_time = df[df['模型'] == '纯嵌入式 (Top 20)']['总耗时 (秒)'].iloc[0]
    
    fig.tight_layout()
    plt.savefig(save, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f'已保存 → {save}')

# -------------- 5. 一键重绘 --------------
if __name__ == '__main__':
    print('正在重新绘制清新配色图表...')
    plot_performance_metrics(PERFORMANCE_DATA)
    plot_accuracy_comparison(PERFORMANCE_DATA)
    plot_time_comparison(TIME_DATA)
    print('\n全部完成！快去文件夹里验收清新感吧~')