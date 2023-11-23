"""把ava时间序列扫描的光谱原始数据转换为一个新的excel，并画动图"""

import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def transmittance_calculation(data_path, new_data_path):
    """计算透过率，并存为新的Excel文件"""
    # 读取原始数据
    df = pd.read_excel(data_path, skiprows=131)  # 【可修改，目前舍去299.5nm以前的数据行】
    # 提取波长列
    wavelength_column = df.iloc[:, 0]
    # 获取背景数据和参考数据
    background = df.iloc[:, 1]
    reference = df.iloc[:, 2]

    # 扣除背景
    transmittance_df = df.iloc[:, 3:].sub(background, axis=0)  # 使用sub将每一列都减去新的列，0表示按行操作
    reference = reference - background
    # 计算透过率
    transmittance_df = transmittance_df.div(reference, axis=0)  # 使用div将每一列都除以新的列，0表示按行操作
    # 创建列名
    transmittance_df_columns = [f'{i * 0.5}s' for i in range(transmittance_df.shape[1])]  # 【可修改，时间间隔】
    transmittance_df.columns = transmittance_df_columns

    # 合并数据
    result_df = pd.concat([wavelength_column, transmittance_df], axis=1)
    # 将第一列重命名为 'Wavelength [nm]'
    result_df.columns.values[0] = 'Wavelength [nm]'

    # 将结果保存到新的Excel文件
    result_df.to_excel(new_data_path, sheet_name='transmittance', index=False, engine='xlsxwriter')
    return print(f"Transmittance data saved to {new_data_path}")


def get_gif(file_path):
    """读取新的excel，并画gif图"""
    df = pd.read_excel(file_path, index_col=0)

    # 初始化图形
    fig, ax = plt.subplots()
    line, = ax.plot(df.index, df.iloc[:, 0], label='0s')  # 初始化第一个时间点

    # 添加标题和标签
    ax.set_title('Dynamic Time Series Spectral Changes')
    ax.set_xlabel('Wavelength (nm)')
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
    data_fold = 'C:/Users/JiaPeng/Desktop/test/后处理'  # 【改地址】
    new_fold = 'C:/Users/JiaPeng/Desktop/test/后处理/时间序列透过率'  # 【改地址】
    # old_name = '20231030a-2000r-cv.xlsx'  # 【改文件名】

    for old_name in os.listdir(data_fold):
        if old_name.endswith('.xlsx'):
            # 生成新的光谱透过率文件
            new_name = 'spectrum_' + old_name
            transmittance_calculation(os.path.join(data_fold, old_name), os.path.join(new_fold, new_name))
            # 生成gif图
            # get_gif(os.path.join(new_fold, new_name))

