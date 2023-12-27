import pandas as pd


# -----读入dataframe数据格式
# 测量光谱数据【改文件地址参数】
df = pd.read_excel(
    'D:/BIT课题研究/微型光谱成像仪/【数据】相机标定/透过法标定/20230623-UV/merged_透过.xlsx',
    header=0, index_col=0)  # 指定标题行和列名


# -----数据处理
# dataframe转为array数据格式
df_array = df.values  # (901, 30) 30条光谱纵向排列


# -----生成观测矩阵
# range(200, 501, 10)个通道实际波长对应的数据行为（400nm-701nm，10）
waves_num = range(200, 501, 5)
# 选择行，即每一行为该波长下所有光谱的数据
array_waves = df_array[waves_num]

# range(0,30,1)选出30条光谱
spec_num = []
for i in range(0, 30, 1):  # （0，30，1）选出30条光谱
    spec_num.append(i)
# 选择列，即每一列为一条光谱
array_filters = array_waves[:, spec_num]
# 转置成每一行为一条光谱数据
array_filters = array_filters.T
