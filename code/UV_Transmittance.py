'''
数据预处理：
将UV测的吸光度sca数据转化为透过率excel数据,并画图
'''
import os
import pandas as pd
import matplotlib.pyplot as plt


# ------吸光度转化为透过率函数------ #
def absorbance_to_transmittance(absorbance):
    return 10 ** (-absorbance)


# ------将同一个文件夹下的sca文件转化为一个总的透过率excel文件------ #
def sca2excel(datafolder, mergedfile_name):
    # 获取所有txt文件的路径
    txt_files = [f for f in os.listdir(data_folder) if f.endswith('.sca')]

    # 读取所有txt文件的数据
    dfs = []
    for txt_file in txt_files:
        with open(data_folder + '/' + txt_file, 'r') as f:
            lines = f.readlines()

        # 获取数据开始的行数
        for i, line in enumerate(lines):
            if line.startswith('Filter:10'):
                start_row = i + 1
                break

        # 从数据开始的行数开始读取数据
        data = []
        for line in lines[start_row:]:
            if line.startswith('[Extended]'):
                break
            data.append(line.strip().split(' '))  # 空格分列

        # 将数据转换为DataFrame，并添加文件名作为一列
        col_name = txt_file.split('.sca')[0]
        df = pd.DataFrame(data, columns=['Wavelengths[nm]', col_name])
        df[col_name] = df[col_name].astype(float)
        dfs.append(df)

    # 将所有DataFrame合并为一个DataFrame，共享第一列Wavelengths
    df_merged = pd.concat([df.set_index('Wavelengths[nm]') for df in dfs], axis=1).reset_index()
    # 将DataFrame保存为excel文件
    df_merged.to_excel(os.path.join(data_folder, mergedfile_name + '-Absorbance_merged.xlsx'),
                       sheet_name='Absorbance', index=False)

    # 从第二列开始，转换为透过率
    df_merged.iloc[:, 1:] = df_merged.iloc[:, 1:].apply(absorbance_to_transmittance)
    # 将DataFrame保存为excel文件
    df_merged.to_excel(os.path.join(data_folder, mergedfile_name + '-Transmittance_merged.xlsx'),
                       sheet_name='Transmittance', index=False)

    return print(f"Merged data saved to {data_folder}")


# ------画出UV曲线并保存------ #
def Transmittance_curve(data_folder, mergedfile_name, transmittance_path):
    # 读取excel文件数据
    df_merged = pd.read_excel(transmittance_path)
    # 获取列名，即光谱曲线的标签
    curve_labels = df_merged.columns[1:]

    # 提前设置图形属性，避免重复
    plt.rcParams['font.sans-serif'] = ['simhei']
    plt.rcParams['axes.unicode_minus'] = False

    # 遍历每个光谱曲线并绘总图
    plt.figure()
    plt.xlim(300, 1100)
    plt.ylim(-0.1, 1.2)
    plt.title("Spectral Curves")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmittance")
    # 循环内只进行绘图设置提高效率
    for curve_label in curve_labels:
        # 获取对应光谱曲线的数据
        wavelength = df_merged.iloc[:, 0]  # 提取第一列数据作为波长数据
        intensity = df_merged[curve_label]
        # 绘制光谱曲线
        plt.plot(wavelength, intensity, label=curve_label)

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_folder, mergedfile_name + '-Transmittance_merged.png'), dpi=300)
    # plt.show(block=True)
    plt.close()
    print(f"Merged Transmittance PNG saved to {data_folder}")

    # 遍历每个光谱曲线并绘图，并保存为独立图像文件
    for curve_label in curve_labels:
        # 创建新的图表
        plt.figure()
        # 获取对应光谱曲线的数据
        wavelength = df_merged.iloc[:, 0]  # 提取第一列数据作为波长数据
        intensity = df_merged[curve_label]
        # 绘制光谱曲线
        plt.plot(wavelength, intensity)
        plt.xlim(300, 1100)
        plt.ylim(-0.1, 1.2)
        plt.title(f"Spectral Curve: {curve_label}")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Transmittance")

        plt.tight_layout()
        # 保存图像，文件名以光谱曲线的标签命名
        plt.savefig(data_folder + f"/Transmittance_{curve_label}.png", dpi=300)
        # plt.show(block=True)
        plt.close()
        print(f"{curve_label} Transmittance PNG saved to {data_folder}")

    return None


if __name__ == "__main__":
    # 设置处理方法和对应的地址
    Folder_processing_method = 'single'  # 【改模式】'single'或 'multiple'
    parent_folder = 'D:/BIT课题研究/微型光谱成像仪/【数据】导电聚合物数据/郁海凡登记/20230828数据/solution/透过-UV'  # 【改地址】
    data_folder = 'D:/BIT课题研究/微型光谱成像仪/【数据】导电聚合物数据/郁海凡登记/20230828数据/film/透过/20230528-UV'  # 【改地址】

    # 封装步骤：对单个文件夹进行数据预处理操作
    def process_data_folder(data_folder):
        mergedfile_name = os.path.basename(data_folder).split('-')[0]
        transmittance_path = os.path.join(data_folder, mergedfile_name + '-Transmittance_merged.xlsx')
        # 先合并excel
        if not os.path.exists(transmittance_path):
            sca2excel(data_folder, mergedfile_name)
        # 画UV光谱图
        Transmittance_curve(data_folder, mergedfile_name, transmittance_path)
        print(f'已处理{data_folder}')

    if Folder_processing_method == 'single':
        process_data_folder(data_folder)

    if Folder_processing_method == 'multiple':
        # 从上一级文件夹开始对所有子文件夹进行数据预处理操作
        subfolders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
        for subfolder in subfolders:
            data_folder = os.path.join(parent_folder, subfolder)
            process_data_folder(data_folder)