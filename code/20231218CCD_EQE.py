import pandas as pd


# -----读入dataframe数据格式
# 测量光谱数据【改文件地址参数】
df = pd.read_excel(
    'D:/BIT课题研究/微型光谱成像仪/【数据】相机标定/单色仪标定/20230609-JAI/JAI标定.xlsx',
    header=0, index_col=0)  # 指定标题行和列名


# -----数据处理
# dataframe转为array数据格式
df_array = df.values


# -----生成ccd_EQE向量
# 选择行，即每一行为该波长下所有光谱的数据
waves_num = range(0, 31, 1)
array_waves = df_array[waves_num]

# 选择列，即选择哪个通道下的EQE
channel_num = 17

# 得到EQE向量列(filters_num, )
array_EQE = array_waves[:, channel_num]

