import random
import numpy as np


# 定义一条高斯曲线
def gaussian(x, a, mu, sigma):
    '''
    :param x: 长度为l的一维数组
    :param a: 峰高
    :param mu: 波长
    :param sigma: 半峰宽
    :return:
    '''
    return a*np.exp(-(x-mu)**2/(2*sigma**2))


# 模拟电致变色透过光谱矩阵
def ECDMatrix(m, n):
    # 初始化一个(m, n)矩阵，保存m个长度为n的光谱数据
    ECDMatrix = np.zeros((m, n))
    Phi = np.zeros((m, n))
    L = np.arange(0, n)
    # CCD的量子效率(m,)
    CCD = gaussian(L, a=1, mu=int(n / 2), sigma=int(n / 2))
    # 随机生成m个光谱
    for i in range(m):
        # 初始化光谱数据
        spec = np.zeros(n)
        # 随机指定光谱峰的个数
        num = random.randint(2, 4)
        for j in range(num):
            spec = gaussian(L, a=random.random(), mu=random.randint(1, n), sigma=random.randint(1, int(n / 2))) + spec
        # 返回电致变色矩阵
        ECDMatrix[i, :] = spec
        # 返回观测矩阵，叠加CCD量子效率和电致变色吸收
        Phi[i, :] = np.multiply(spec, CCD)  # multiply逐元素相乘

    return Phi