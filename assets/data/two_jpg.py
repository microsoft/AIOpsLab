import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_bar_graph(file_path):
    # 默认参数设置
    spacing_factor = 0.5 # 间距因子
    bar_height = 0.2 # 条形图高度
    vline_text = "Overall Average" # 垂直线文字
    text_size = 12 # 虚线文字大小
    font_size = 12 # ylabel 字体大小

    data = pd.read_csv(file_path)
    metrics = ['Detection', 'Localization', 'Diagnosis', 'Mitigation']
    for col in metrics:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    data['Average'] = data[metrics].mean(axis=1)
    
    data_sorted = data.sort_values(by='Average', ascending=True)
    agents = data_sorted['Agent Name']
    avg_values = data_sorted['Average']
    N = len(data_sorted)
    
    ind = np.arange(N) * spacing_factor
    
    plt.figure(figsize=(8, 8)) # 设置图形大小
    
    overall_avg = data_sorted['Average'].mean()
    max_val = data_sorted['Average'].max()
    
    plt.axvline(x=overall_avg, color='red', linestyle='--', linewidth=2)
    plt.text(overall_avg + 0.02 * max_val, 0, f'{vline_text}= {overall_avg:.2f}', 
             color='red', va='center', ha='left', fontsize=text_size)
    
    plt.barh(ind, avg_values, height=bar_height, color='skyblue', edgecolor='blue')
    
    plt.yticks(ind, agents)
    plt.xlabel("Average Score", fontsize=font_size)
    plt.tight_layout()
    plt.savefig("bar.png", dpi=800, bbox_inches='tight')


def plot_rabar_graph(file_path):
    data = pd.read_csv(file_path)
    agent_col = "Agent Name"
    cols = ['Detection', 'Localization', 'Diagnosis', 'Mitigation']
    for col in cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    # 最大值用于设置雷达图的径向坐标
    max_value = data[cols].values.max()
    if max_value == 0:
        max_value = 1

    num_vars = len(cols)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  
    plt.figure(figsize=(8, 8)) # 设置图形大小
    ax = plt.subplot(111, polar=True)
    ax.set_xticklabels([])  
    ax.tick_params(axis='y', pad=0)
    colors = plt.get_cmap('tab10', len(data))
    # 绘制每个 Agent 的雷达图
    for idx, (index, row) in enumerate(data.iterrows()):
        values = row[cols].tolist()
        values += values[:1]  
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row[agent_col], color=colors(idx))
        ax.fill(angles, values, alpha=0.1, color=colors(idx))

    ax.set_rlabel_position(45)
    yticks = np.linspace(0, max_value, num=5)
    plt.yticks(yticks, [str(round(x, 2)) for x in yticks], color="grey", size=7)
    plt.ylim(0, max_value)

    # 自定义 xtick 文本，设置与雷达图的距离
    offsets = {
        'Detection': 1.2,
        'Localization': 1.05,  
        'Diagnosis': 1.2,
        'Mitigation': 1.05     
    }
    for angle, label in zip(angles[:-1], cols): # 设置 xtick 文本，放在雷达图外侧，与雷达图的距离由 offsets 决定，字体大小为 15
        ax.text(angle, max_value * offsets[label], label,
                size=15, horizontalalignment='center', verticalalignment='center') 

    plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize='larger') # 设置图例，放在雷达图下方，分三列，字体大小为 large
    plt.subplots_adjust(bottom=0.2)
    plt.savefig("radar.png", dpi=800, bbox_inches='tight')


if __name__ == "__main__":
    plot_bar_graph("data.csv")
    plot_rabar_graph("data.csv")

