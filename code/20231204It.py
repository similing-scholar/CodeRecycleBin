import matplotlib.pyplot as plt
import glob


# 获取目录中所有的 .txt 文件列表
file_list = glob.glob('D:/BIT课题研究/微型光谱成像仪/【数据】导电聚合物数据/20230712-kei/*.txt')


for file_name in file_list:
    # 获取数据
    time = []
    voltage = []
    current = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            values = line.strip().split('\t')
            time.append(float(values[0]))
            voltage.append(float(values[1]))
            current.append(float(values[2]))

    # 创建一个带有 gridspec 布局的图形
    fig = plt.figure(figsize=(4, 5))  # 根据需要调整 figsize
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3])  # 子图高度比例

    # 在第一个子图中绘制电压随时间变化的图形
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time, voltage)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Voltage vs Time')

    # 在第二个子图中绘制电流随时间变化的图形
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(time, current)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Current (mA)')
    ax2.set_title('Current vs Time')

    # 在图中添加文件名标签
    plt.text(0.5, 0.02, file_name.split("\\")[1], ha='center', va='bottom', transform=fig.transFigure)

    # 调整子图之间的间距
    plt.subplots_adjust(hspace=0.3)  # 根据需要调整间距

    plt.tight_layout()
    plt.rcParams['font.sans-serif'] = ['simhei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.savefig(file_name.replace('.txt', '_plot.png'))
    # plt.show(block=True)
    plt.close()