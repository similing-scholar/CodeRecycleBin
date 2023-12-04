'''
数据预处理：
将lsv方式测的电阻数据画图
'''
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ------将同一个文件夹下的csv文件转化为一个总的excel文件------ #
def merge_excel(data_folder, mergedfile_name):
    # 获取数据文件列表
    data_files = [file for file in os.listdir(data_folder) if file.endswith('.csv')]
    # 创建一个空的DataFrame来存储合并的数据
    merged_data = pd.DataFrame()

    # 循环读取每个数据文件，并将电流值合并到DataFrame中
    for file in data_files:
        file_path = os.path.join(data_folder, file)
        df = pd.read_csv(file_path, delimiter=',', skiprows=9)  # 【可修改，数据行前面的都舍去】
        # 将第一列重命名，并赋值为电压列名
        df.rename(columns={df.columns[0]: "Voltage(V)"}, inplace=True)
        voltage_column = df.columns[0]
        # 使用文件名将电流列重命名
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        df.rename(columns={df.columns[1]: f'{file_name}'}, inplace=True)
        # 合并数据到主DataFrame中，基于电压列
        if merged_data.empty:
            merged_data = df
        else:
            merged_data = pd.merge(merged_data, df, on=voltage_column, how='outer')

    # 将合并的数据写入Excel文件
    merged_data.to_excel(os.path.join(data_folder, mergedfile_name + '-Resistance_merged.xlsx'),
                         sheet_name='Resistance', index=False, engine='openpyxl')

    return print(f"Merged data saved to {data_folder}")


# -----绘制每个子文件夹的IV曲线,和每一个片子的IV曲线，并将线性拟合参数保存到父文件夹中-----
def IV_resistance_curve(data_folder, mergedfile_name, resistance_path):
    # 读取Excel文件数据
    df_merged = pd.read_excel(resistance_path)
    # 获取列名，即光谱曲线的标签
    curve_labels = df_merged.columns[1:]

    # 提前设置图形属性，避免重复
    plt.rcParams['font.sans-serif'] = ['simhei']
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制电流-电压曲线
    plt.figure()
    plt.grid(True)  # 辅助网格样式
    plt.title('IV Curve')
    plt.xlabel('Potential (V)')
    plt.ylabel('Current (A)')
    # 使用科学计数法表示纵轴坐标
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    for curve_label in curve_labels:
        # 获取对应IV曲线的数据
        Potential = df_merged.iloc[:, 0]  # 提取第一列数据作为波长数据
        Current = df_merged[curve_label]
        plt.plot(Potential, Current, label=curve_label)

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_folder, mergedfile_name + '-Resistance_merged.png'), dpi=300)
    # plt.show(block=True)
    plt.close()
    print(f"Merged Resistance PNG saved to {data_folder}")

    # 创建一个空的DataFrame来存储相关系数和斜率
    results_df = pd.DataFrame(columns=['Curve Label', 'Correlation Coefficient', 'Slope', 'Intercept'])
    # 遍历每个IV曲线并绘图，并保存为独立图像文件
    for curve_label in curve_labels:
        # 获取对应IV曲线的数据
        Potential = df_merged.iloc[:, 0]  # 提取第一列数据作为波长数据
        Current = df_merged[curve_label]

        # 删除包含NaN值的行：电流列比电压列少的地方由NaN值填充
        non_nan_indices = Current.notna()
        Potential = Potential[non_nan_indices]
        Current = Current[non_nan_indices]

        # 在给定电压范围内进行线性拟合并计算相关系数
        voltage1 = Potential.iloc[0]  # 使用电流列的第一个值对应的电压
        voltage2 = Potential.iloc[-1]  # 使用电流列的最后一个值对应的电压
        voltage_range_mask = (Potential >= voltage1) & (Potential <= voltage2)
        fit_potential = Potential[voltage_range_mask]
        fit_current = Current[voltage_range_mask]
        # 计算线性拟合的系数
        coeffs = np.polyfit(fit_potential, fit_current, 1)
        # 计算相关系数
        correlation = np.corrcoef(fit_potential, fit_current)[0, 1]
        fit_line = np.poly1d(coeffs)
        # 将结果添加到DataFrame
        results_df = results_df.append({'Curve Label': curve_label,
                                        'Correlation Coefficient': correlation,
                                        'Slope': coeffs[0],
                                        'Intercept': coeffs[1]},
                                       ignore_index=True)

        # 创建新的图表
        plt.figure()
        plt.grid(True)  # 辅助网格样式
        # 在标题中添加相关系数,线性拟合的斜率与截距(科学计数法表示)
        title = (f'IV Curve with Linear Fit Coefficients\n'
                 f'Correlation Coefficient: {correlation:.4f}, '
                 f'Slope: {"{:.2e}".format(coeffs[0])}')
        plt.title(title)
        plt.xlabel('Potential (V)')
        plt.ylabel('Current (A)')
        # 使用科学计数法表示纵轴坐标
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        # 绘制直线
        plt.plot(Potential, Current, marker='o', linestyle='-', label='IV Curve')
        plt.plot(fit_potential, fit_line(fit_potential), linestyle='--', label='Linear Fit')

        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(data_folder, f"Resistance_{curve_label}.png"), dpi=300)
        # plt.show(block=True)
        plt.close()
        # 线性拟合的斜率与截距
        print(f"{curve_label} Resistance PNG saved to {data_folder}")

    # 保存结果到Excel文件
    results_df.to_excel(os.path.join(data_folder, mergedfile_name + '-LinearFit_Coefficients.xlsx'), index=False)
    print(f"Linear Fit Results saved to {data_folder}")

    return None


if __name__ == "__main__":
    # 设置处理方法和对应的地址
    Folder_processing_method = 'single'  # 【改模式】'single'或 'multiple',
    parent_folder = r'D:/BIT课题研究/微型光谱成像仪/【数据】导电聚合物数据/郁海凡登记/20230828数据/film/电阻'  # 【改地址】
    data_folder = r'D:/BIT课题研究/微型光谱成像仪/【数据】导电聚合物数据/郁海凡登记/20230828数据/film/电阻/20220926-hy'  # 【改地址】

    # 封装步骤：对单个文件夹进行数据预处理操作
    def process_data_folder(data_folder):
        mergedfile_name = os.path.basename(data_folder).split('-')[0]
        resistance_path = os.path.join(data_folder, mergedfile_name + '-Resistance_merged.xlsx')
        # 先合并excel
        if not os.path.exists(resistance_path):
            merge_excel(data_folder, mergedfile_name)
        # 画UV光谱图
        IV_resistance_curve(data_folder, mergedfile_name, resistance_path)
        print(f'已处理{data_folder}')

    if Folder_processing_method == 'single':
        process_data_folder(data_folder)

    if Folder_processing_method == 'multiple':
        # 从上一级文件夹开始对所有子文件夹进行数据预处理操作
        subfolders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
        for subfolder in subfolders:
            data_folder = os.path.join(parent_folder, subfolder)
            process_data_folder(data_folder)
