"""仿真计算观测矩阵和正交的稀疏矩阵的互相关性"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import entropy

from Measurement_matrix import CommonMatrixs
from Measurement_matrix import ExperimentMatrix_AVAdata


def get_DCT(n):
    """构建一个(n, n)DCT矩阵"""
    DCT_matrix = np.zeros((n, n))
    for row in range(n):
        # 第一行系数
        if row == 0:
            c = np.sqrt(1 / n)
        else:
            c = np.sqrt(2 / n)
        for col in range(n):
            DCT_matrix[row, col] = c * (np.cos((np.pi * (2 * col + 1) * row) / (2 * n)))

    return DCT_matrix


def RowCorrelation(matrix):
    """计算一个(m,n)矩阵的行相关性"""
    return np.corrcoef(matrix)


def entropy_1D(matrix):
    # 将矩阵展平成一维数组
    flattened_matrix = matrix.flatten()
    # 计算灰度熵
    gray_entropy = entropy(flattened_matrix)
    return gray_entropy


def MatrixCorrelation(matrix_A, matrix_B):
    """计算一个(m,n)矩阵和一个(n,n)正交矩阵的互相关性"""
    matrix_C = np.dot(matrix_A, matrix_B)
    # 使用corrcoef函数计算两个矩阵的列相关性
    correlation_matrix = np.corrcoef(matrix_A, matrix_C, rowvar=False)  # (m+n,m+n)的矩阵
    # 提取互相关系数部分
    m, n = matrix_A.shape
    cross_correlation = correlation_matrix[:m, n:]

    return cross_correlation


def show_result_split(spectra_matrix):
    """绘制观测矩阵的分析结果"""
    m, n = spectra_matrix.shape

    # ---绘制观测矩阵的所有光谱---
    # 映射一个x轴
    x = np.linspace(400, 700, n)
    # 绘制光谱曲线
    colors = plt.cm.copper(np.linspace(0, 1, m))  # 颜色数目为曲线数量m
    for i in range(m):
        plt.plot(x, spectra_matrix[i, :], '-', color=colors[i])

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('Gaussian Distribution Spectra Matrix')
    plt.grid(True)
    plt.show()

    # ---绘制观测矩阵的二维热力图---
    plt.figure(figsize=(9, 4))
    plt.imshow(spectra_matrix)
    plt.colorbar()
    # 计算矩阵的秩
    # spectra_matrix = np.dot(spectra_matrix, get_DCT(n))
    rank = np.linalg.matrix_rank(spectra_matrix)
    plt.title(f"Shape:{spectra_matrix.shape}, Sampling Rate:{m / n}, Rank: {rank}")
    plt.tight_layout()
    plt.show()

    # ---计算观测矩阵的行相关性---
    row_correlation = RowCorrelation(spectra_matrix)
    plt.figure()
    plt.imshow(row_correlation, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Row Correlation of Observation matrix')
    plt.tight_layout()
    plt.show()

    # ---计算观测矩阵和正交稀疏基矩阵的相关性---
    cross_correlation = MatrixCorrelation(spectra_matrix, get_DCT(n))
    plt.figure(figsize=(9, 4))
    plt.imshow(cross_correlation, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation between Observation and Sparse matrix')
    plt.tight_layout()
    plt.show()

    return None


def show_result(spectra_matrix):
    """绘制观测矩阵的分析结果"""
    m, n = spectra_matrix.shape

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # --- 绘制观测矩阵的所有光谱 ---
    x = np.linspace(400, 700, n)
    colors = plt.cm.copper(np.linspace(0, 1, m))
    for i in range(m):
        axs[0, 0].plot(x, spectra_matrix[i, :], '-', color=colors[i])

    axs[0, 0].set_xlabel('Wavelength (nm)')
    axs[0, 0].set_ylabel('Transmittance')
    axs[0, 0].set_title('Spectral Measurement Matrix')
    axs[0, 0].grid(True)

    # --- 绘制观测矩阵的二维热力图 ---
    heatmap = axs[0, 1].imshow(spectra_matrix)
    rank = np.linalg.matrix_rank(spectra_matrix)
    axs[0, 1].set_title(f"Shape:{spectra_matrix.shape}, Sampling Rate:{m / n}, Rank: {rank}")
    fig.colorbar(heatmap, ax=axs[0, 1])

    # --- 计算观测矩阵的行相关性 ---
    row_correlation = np.corrcoef(spectra_matrix)
    im2 = axs[1, 0].imshow(row_correlation, cmap='coolwarm', vmin=-1, vmax=1)
    axs[1, 0].set_title('Row Correlation of Observation matrix')
    fig.colorbar(im2, ax=axs[1, 0])

    # --- 计算观测矩阵和正交稀疏基矩阵的相关性 ---
    cross_correlation = MatrixCorrelation(spectra_matrix, get_DCT(n))
    im3 = axs[1, 1].imshow(cross_correlation, cmap='coolwarm', vmin=-1, vmax=1)
    axs[1, 1].set_title('Correlation between \n Observation and Sparse matrix')
    fig.colorbar(im3, ax=axs[1, 1])

    plt.tight_layout()
    plt.show()

    return None



class GenerateSingleGaussianMatrix:
    """生成观测矩阵
    由一系列单峰高斯分布光谱组成
    """

    def __init__(self, m, n):
        self.m = m
        self.n = n
        # 创建一个波谱 x 轴
        self.x = np.linspace(0, 100, n)
        # 创建一个空的NumPy矩阵，用于存储光谱数据
        self.spectra_matrix = np.zeros((m, n))

    def change_positions(self, positions_start, positions_end, amplitude, peak_width):
        # 峰位
        center_positions = np.linspace(positions_start, positions_end, self.m)  # 修改峰位定义域
        # 峰高
        amplitude = amplitude
        # 峰宽
        peak_width = peak_width
        # 生成并存储每条光谱
        for i, center_position in enumerate(center_positions):
            gaussian_curve = amplitude * np.exp(-((self.x - center_position) / peak_width) ** 2)
            self.spectra_matrix[i, :] = gaussian_curve
        return self.spectra_matrix

    def change_amplitude(self, amplitude_start, amplitude_end, center_position, peak_width):
        # 峰位
        center_position = center_position
        # 峰高
        amplitudes = np.linspace(amplitude_start, amplitude_end, self.m)  # 修改峰高定义域
        # 峰宽
        peak_width = peak_width
        # 生成并存储每条光谱
        for i, amplitude in enumerate(amplitudes):
            gaussian_curve = amplitude * np.exp(-((self.x - center_position) / peak_width) ** 2)
            self.spectra_matrix[i, :] = gaussian_curve
        return self.spectra_matrix

    def change_width(self, peak_width_start, peak_width_end, center_position, amplitude):
        # 峰位
        center_position = center_position
        # 峰高
        amplitude = amplitude
        # 峰宽
        peak_widths = np.linspace(peak_width_start, peak_width_end, self.m)  # 修改峰高定义域
        # 生成并存储每条光谱
        for i, peak_width in enumerate(peak_widths):
            gaussian_curve = amplitude * np.exp(-((self.x - center_position) / peak_width) ** 2)
            self.spectra_matrix[i, :] = gaussian_curve
        return self.spectra_matrix

    def change_positions_width(self, positions_start, positions_end, peak_width_start, peak_width_end, amplitude):
        # 峰位
        center_positions = np.linspace(positions_start, positions_end, self.m)  # 修改峰位定义域
        # 峰高
        amplitude = amplitude
        # 峰宽
        peak_widths = np.linspace(peak_width_start, peak_width_end, self.m)  # 修改峰高定义域
        # 生成并存储每条光谱
        for i, (center_position, peak_width) in enumerate(zip(center_positions, peak_widths)):
            gaussian_curve = amplitude * np.exp(-((self.x - center_position) / peak_width) ** 2)
            self.spectra_matrix[i, :] = gaussian_curve
        return self.spectra_matrix

    def chang_potions_width_amplitude(self, positions_start, positions_end, peak_width_start, peak_width_end, amplitude_start, amplitude_end):
        # 峰位
        center_positions = np.linspace(positions_start, positions_end, self.m)
        # 峰高
        amplitudes = np.linspace(amplitude_start, amplitude_end, self.m)
        # 峰宽
        peak_widths = np.linspace(peak_width_start, peak_width_end, self.m)
        # 生成并存储每条光谱
        for i, (center_position, peak_width, amplitude) in enumerate(zip(center_positions, peak_widths, amplitudes)):
            gaussian_curve = amplitude * np.exp(-((self.x - center_position) / peak_width) ** 2)
            self.spectra_matrix[i, :] = gaussian_curve
        return self.spectra_matrix


class GenerateMultiGaussianMatrix:
    """生成观测矩阵
    由一系列多峰高斯分布光谱组成
    """

    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.combination_spectra_matrix = np.zeros((m, n))

    def get_array_filters(self):
        """一个实测的数据集
        """
        df = pd.read_excel(
            'D:/BIT课题研究/微型光谱成像仪/【数据】相机标定/透过法标定/20230703-ava-滤光片&光源/滤光片透过率.xlsx',
            header=0)  # 滤光片光谱数据【改文件地址参数】
        # 使用drop方法删除第一列
        df = df.drop(df.columns[0], axis=1)
        # 定义要进行插值的行
        desired_wavelengths = np.linspace(0, 300, self.n)  # 全闭区间包括0和300
        # 进行插值，以获取所需波长处的光谱数值
        for i in range(self.m):
            self.combination_spectra_matrix[i, :] = np.interp(desired_wavelengths, df.index,
                                                              df.iloc[:, i])  # 插值位置，x值，y值
        return self.combination_spectra_matrix

    def generate_gaussian_vectors(self, max_gaussians):
        """生成随机峰个数叠加的高斯分布光谱
        max_gaussians：每个向量叠加的高斯分布数量
        """
        np.random.seed(0)  # 设置随机种子以确保结果可重复
        # 定义x轴的取值范围和分辨率
        x = np.linspace(-5, 5, self.n)
        for i in range(self.m):
            # 随机确定每个向量叠加的高斯分布数量
            num_gaussians = np.random.randint(1, max_gaussians + 1)
            # 生成多个高斯分布的叠加曲线
            combined_curve = np.zeros(self.n)
            for _ in range(num_gaussians):
                mean = np.random.uniform(-5, 5)  # 随机均值
                std_dev = np.random.uniform(0.1, 3)  # 随机标准差
                gaussian = (1.0 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2))
                weight = np.random.uniform(0.1, 0.5)  # 随机权重
                combined_curve += weight * gaussian
            # 将生成的曲线保存到数组中
            self.combination_spectra_matrix[i, :] = combined_curve
        return self.combination_spectra_matrix

    def multi_peaks(self, *spectra_matrixs):
        """任意组合单个峰的光谱"""
        for spectra_matrix in spectra_matrixs:
            self.combination_spectra_matrix += spectra_matrix
        return self.combination_spectra_matrix

    def double_peak_1(self):
        """两个峰的中心均不变，一个峰高减小，一个峰高增加"""
        single_gaussian_a = GenerateSingleGaussianMatrix(m=self.m, n=self.n)
        a = single_gaussian_a.change_amplitude(amplitude_start=0, amplitude_end=70, center_position=25,
                                                            peak_width=25)
        single_gaussian_b = GenerateSingleGaussianMatrix(m=self.m, n=self.n)
        b = single_gaussian_b.change_amplitude(amplitude_start=100, amplitude_end=0, center_position=75,
                                                            peak_width=50)
        return a + b

    def double_peak_2(self):
        """一个峰中心不变，展宽；一个峰高度不变，中心移动"""
        single_gaussian_a = GenerateSingleGaussianMatrix(m=self.m, n=self.n)
        a = single_gaussian_a.change_width(peak_width_start=25, peak_width_end=50, center_position=30,
                                                        amplitude=100)
        single_gaussian_b = GenerateSingleGaussianMatrix(m=self.m, n=self.n)
        b = single_gaussian_b.change_positions(positions_start=50, positions_end=80, amplitude=100,
                                                            peak_width=50)
        return a + b

    def double_peak_3(self):
        """两个峰的中心、宽度、高度均改变"""
        single_gaussian_a = GenerateSingleGaussianMatrix(m=self.m, n=self.n)
        a = single_gaussian_a.chang_potions_width_amplitude(positions_start=35, positions_end=5, peak_width_start=60,
                                                            peak_width_end=20, amplitude_start=80, amplitude_end=50)
        single_gaussian_b = GenerateSingleGaussianMatrix(m=self.m, n=self.n)
        b = single_gaussian_b.chang_potions_width_amplitude(positions_start=90, positions_end=60, peak_width_start=10,
                                                            peak_width_end=50, amplitude_start=50, amplitude_end=60)
        return a + b


if __name__ == '__main__':
    # 常见高斯矩阵
    # Phi = CommonMatrixs.gaussian_matrix(m=32, n=64)

    # 单峰高斯分布光谱矩阵
    single_gaussian = GenerateSingleGaussianMatrix(m=32, n=64)
    # Phi = single_gaussian.change_positions(positions_start=0, positions_end=100, amplitude=1, peak_width=25)
    # Phi = single_gaussian.change_amplitude(amplitude_start=0, amplitude_end=1, center_position=25, peak_width=25)
    # Phi = single_gaussian.change_width(peak_width_start=0, peak_width_end=100, center_position=1, amplitude=1)
    # Phi = single_gaussian.change_positions_width(positions_start=30, positions_end=70, peak_width_start=20, peak_width_end=50, amplitude=1)
    # Phi = single_gaussian.chang_potions_width_amplitude(positions_start=0, positions_end=30, peak_width_start=20,
    #                                                     peak_width_end=50, amplitude_start=100, amplitude_end=80)

    # 双峰
    multi_gaussian = GenerateMultiGaussianMatrix(m=32, n=64)
    # Phi = multi_gaussian.get_array_filters()
    # Phi = multi_gaussian.generate_gaussian_vectors(20)
    # Phi = multi_gaussian.double_peak_1()
    # Phi = multi_gaussian.double_peak_2()
    Phi = multi_gaussian.double_peak_3()

    # 画图
    print(Phi)
    show_result(Phi)
