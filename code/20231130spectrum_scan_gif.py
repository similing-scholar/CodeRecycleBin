"""将时间序列的光谱先画成gif图片，但是只能在jupyter运行？？？"""
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def get_gif(file_path):
    """读取新的excel，并画gif图"""
    df = pd.read_excel(file_path, index_col=0)

    # 初始化图形
    fig, ax = plt.subplots()
    line, = ax.plot(df.index, df.iloc[:, 0], label='0s')  # 初始化第一个时间点

    # 添加标题和标签
    ax.set_title('Dynamic Time Series Spectral Changes')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Transmittance')
    ax.legend(title='Time')

    # 更新函数，用于每个动画帧
    def update(frame):
        line.set_ydata(df.iloc[:, frame])
        line.set_label(f'{frame * 0.5}s')  # 更新标签
        legend = ax.legend(title='Time')
        return line, legend

    # 创建动画
    animation = FuncAnimation(fig, update, frames=df.shape[1], interval=50)
    # 保存为 GIF
    animation.save(file_path.replace('xlsx', 'gif'), writer='pillow', fps=30)
    # 显示动画
    plt.show()
    return print("GIF saved")


if __name__ == '__main__':
    # 读取新的Excel文件
    file_path = 'C:/Users/JiaPeng/Desktop/test/ava-excel/时间序列透过率/spectrum_20231030a-2000r-cv-Na.xlsx'
    get_gif(file_path)