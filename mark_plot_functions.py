import matplotlib.pyplot as plt
import numpy as np

def plot_snr_subgraphs(groups, data, title="SNR Measurements", y_range=(5, 25), ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_ylim(y_range)

    for i, (group, values) in enumerate(zip(groups, data)):
        max_val = max(values)
        min_val = min(values)
        median_val = np.median(values)  # 计算中值

        # 使用竖线连接最大值和最小值
        ax.vlines(i, min_val, max_val, color='blue', linewidth=2)

        # 绘制最大值和最小值的横线
        line_width = 0.1  # 根据需要可以调整线的长度
        ax.hlines(max_val, i - line_width, i + line_width, color='blue', linewidth=2)
        ax.hlines(min_val, i - line_width, i + line_width, color='blue', linewidth=2)

        # 标注最大值和最小值以及中值
        ax.text(i + line_width, max_val, f'{max_val:.2f}', color='blue', fontsize=8, ha='left')
        ax.text(i + line_width, min_val, f'{min_val:.2f}', color='blue', fontsize=8, ha='left')
        ax.text(i + 0.05, median_val, f'{median_val:.2f}', color='red', fontsize=8, ha='left')

        # 使用叉号标记中值
        ax.scatter(i, median_val, color='red', marker='x', s=100)

        # 绘制其余的数据点，但不标注
        for value in values:
            if value not in [max_val, min_val, median_val]:
                ax.scatter(i, value, color='red', marker='x', s=50)  # 使用灰色点表示其余数据点

    ax.set_ylabel('SNR (dB)')
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha='right')
    
    if ax is None:
        plt.show()

'''
#只画叉号的
import matplotlib.pyplot as plt

from module_reloader import reload_module
reload_module('mark_plot_functions')
from mark_plot_functions import plot_snr_subgraphs

# 使用示例
fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1行2列的子图布局

# 对不同的数据集调用函数
groups = ['ResBiasing_Ch1', 'ResBiasing_Ch', 'ResBiasing_Ch3', 'ResBiasing_Ch4']
SNR_old_board = [[19.85, 20.68, 21.14], [16.01, 18.58, 18.71], [11.43, 12.63, 13.26], [20.66, 22.15, 22.1]]
SNR_new_board = [[21.79, 21.3, 18.88], [18.95, 18.64, 15.27], [17.1, 14.88, 14.66], [23.49, 22.6, 21.45]]

plot_snr_subgraphs(groups, SNR_old_board, "Old board -- resBiasing", y_range=(5, 25), ax=axs[0])
plot_snr_subgraphs(groups, SNR_new_board, "New board -- resBiasing", y_range=(5, 25), ax=axs[1])

plt.tight_layout()  # 调整子图布局
plt.show()
'''




'''
support single image plot,example:
groups = ['EarBiasing', 'ShortDisWithBaising', 'NoBiasing', 'ResBiasing']
data = [[13.37, 9.54, 16.16], [12.34, 8.79, 13.04], [11.66, 8.98, 6.60], [7.58, 11.83, 10.43]]
title = "Measurement of Old Board (Version 2) on Jan Head"
plot_snr_data(groups, data, title, y_range=(5, 25))
'''


def plot_snr_single(groups, data, title="SNR Measurements", y_range=(5, 25)):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_ylim(y_range)

    # 定义x坐标，使组之间更接近
    x_positions = np.linspace(0, len(groups)-1.5, len(groups))

    for i, (group, values) in enumerate(zip(groups, data)):
        max_val = max(values)
        min_val = min(values)
        median_val = np.median(values)

        # 使用自定义的x坐标
        x = x_positions[i]

        # 使用竖线连接最大值和最小值
        ax.vlines(x, min_val, max_val, color='blue', linewidth=2)

        # 绘制最大值和最小值的横线
        line_width = 0.05  # 根据需要可以调整线的长度
        ax.hlines(max_val, x - line_width, x + line_width, color='blue', linewidth=2)
        ax.hlines(min_val, x - line_width, x + line_width, color='blue', linewidth=2)

        # 标注最大值和最小值
        ax.text(x + line_width, max_val, f'{max_val:.2f}', color='blue', fontsize=8, ha='left')
        ax.text(x + line_width, min_val, f'{min_val:.2f}', color='blue', fontsize=8, ha='left')
        # 使用叉号标记中间值
        ax.scatter(x, median_val, color='red', marker='x', s=100)
        ax.text(x+0.05, median_val, f'{median_val:.2f}', color='red', fontsize=8, ha='left')
        
        # 绘制其余数据点，不标注
        for value in values:
            if value not in [max_val, min_val, median_val]:
                ax.scatter(x, value, color='red', marker='x', s=50)

    # 设置轴标签
    ax.set_ylabel('SNR (dB)')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(groups, rotation=45, ha='right')

    plt.show()



def plot_SSVEP_50Hz_ratio_single(groups, data, title="SSVEP_50Hz_ratio", y_range=(-70, -20)):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_ylim(y_range)

    # 定义x坐标，使组之间更接近
    x_positions = np.linspace(0, len(groups)-1.5, len(groups))

    for i, (group, values) in enumerate(zip(groups, data)):
        max_val = max(values)
        min_val = min(values)
        median_val = np.median(values)

        # 使用自定义的x坐标
        x = x_positions[i]

        # 使用竖线连接最大值和最小值
        ax.vlines(x, min_val, max_val, color='blue', linewidth=2)

        # 绘制最大值和最小值的横线
        line_width = 0.05  # 根据需要可以调整线的长度
        ax.hlines(max_val, x - line_width, x + line_width, color='blue', linewidth=2)
        ax.hlines(min_val, x - line_width, x + line_width, color='blue', linewidth=2)

        # 标注最大值和最小值
        ax.text(x + line_width, max_val, f'{max_val:.2f}', color='blue', fontsize=8, ha='left')
        ax.text(x + line_width, min_val, f'{min_val:.2f}', color='blue', fontsize=8, ha='left')
        # 使用叉号标记中间值
        ax.scatter(x, median_val, color='red', marker='x', s=100)
        ax.text(x+0.05, median_val, f'{median_val:.2f}', color='red', fontsize=8, ha='left')
        
        # 绘制其余数据点，不标注
        for value in values:
            if value not in [max_val, min_val, median_val]:
                ax.scatter(x, value, color='red', marker='x', s=50)

    # 设置轴标签
    ax.set_ylabel('SSVEP/50Hz ratio (dB)')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(groups, rotation=45, ha='right')

    plt.show()