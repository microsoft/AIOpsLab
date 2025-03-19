import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_combined_graphs(file_path, out_path):
    data = pd.read_csv(file_path)
    metrics = ['Localization', 'Detection', 'Mitigation', 'Diagnosis']
    for col in metrics:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    data['Average'] = data[metrics].mean(axis=1)
    
    # 雷达图需要的最大值
    max_value = data[metrics].values.max()
    if max_value == 0:
        max_value = 1

    fig = plt.figure(figsize=(16, 5))
    ax1 = fig.add_subplot(121, polar=True)
    ax2 = fig.add_subplot(122)
    
    # 绘制雷达图
    scale_size = 12   # 刻度文字大小
    text_size  = 18   # 四个指标文字大小
    num_vars = len(metrics)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles.append(angles[0])  # 闭合数据
    ax1.set_xticklabels([])  
    ax1.tick_params(axis='y', pad=0)
    
    colors = plt.get_cmap('tab10', len(data))
    agent_col = "Agent Name"
    for idx, (index, row) in enumerate(data.iterrows()):
        values = row[metrics].tolist()
        values.append(values[0])
        ax1.plot(angles, values, linewidth=2, linestyle='solid', label=row[agent_col], color=colors(idx))
        ax1.fill(angles, values, alpha=0.1, color=colors(idx))
    
    ax1.set_rlabel_position(45)
    yticks = np.linspace(0, max_value, num=5)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels([str(round(x, 2)) for x in yticks], color="grey", size=scale_size)
    ax1.set_ylim(0, max_value)
    
    # 设置雷达图外侧指标文字位置
    offsets = {
        'Detection': 1.12,
        'Localization': 1.53,  
        'Diagnosis': 1.12,
        'Mitigation': 1.45     
    }
    for angle, label in zip(angles[:-1], metrics):
        ax1.text(angle, max_value * offsets[label], label,
                 size=text_size, horizontalalignment='center', verticalalignment='center')
    
    ax1.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.1), fontsize='x-large') # legend文字大小设置
    
    # 绘制条形图
    # 默认参数设置
    spacing_factor = 0.5  # 间距因子
    bar_height = 0.2      # 条形高度
    vline_text = "Average"  # 虚线文字
    text_size = 14       # 虚线文字大小
    font_size = 18        # x坐标轴字体大小
    xy_size = 15        # x,y轴标签字体大小

    data_sorted = data.sort_values(by='Average', ascending=True)
    agents = data_sorted['Agent Name']
    avg_values = data_sorted['Average']
    N = len(data_sorted)
    ind = np.arange(N) * spacing_factor
    
    overall_avg = data_sorted['Average'].mean()
    max_val = data_sorted['Average'].max()
    
    ax2.axvline(x=overall_avg, color='black', linestyle='--', linewidth=2)
    ax2.text(overall_avg + 0.02 * max_val, 0, f'{vline_text}= {overall_avg:.2f}',
             color='black', va='center', ha='left', fontsize=text_size)
    
    # 自定义agent颜色设置
    custom_colors = {
        "FLASH (GPT-4)": "#10A37F",
        "REACT (GPT-4)": "#10A37F",
        "DeepSeek-R1": "#4E6CFE",
        "GPT-4 w Shell": "#10A37F",
        "GPT-3.5 w Shell": "#FEA443",
        "FLASH (Llama3-8b)": "#7C2AE8",
        "REACT (Llama3-8b)": "#7C2AE8",
        "LocaleXpert (Llama3-8b)": "#7C2AE8"
    }
    bar_colors = [custom_colors.get(agent, "skyblue") for agent in agents]
    ax2.barh(ind, avg_values, height=bar_height, color=bar_colors, edgecolor=bar_colors)
    
    ax2.set_yticks(ind)
    ax2.set_yticklabels(agents)
    ax2.set_xlabel("Average Score", fontsize=font_size)
    ax2.tick_params(labelsize=xy_size)
    # 调整整体布局
    fig.tight_layout()

    plt.savefig(out_path + "/combined.jpg", dpi=800, bbox_inches='tight')

if __name__ == "__main__":
    plot_combined_graphs("data.csv", "../../pages")