import pandas as pd


def ECD_spec():
    # -----读入dataframe数据格式
    # 背景光谱数据【改文件地址参数】
    df_back = pd.read_excel(
        'D:/BIT课题研究/微型光谱成像仪/【数据】导电聚合物数据/【数据】测试数据/ava/20230107-ava/device/Dark+Ref.xlsx',
        header=5, index_col=0)  # 指定标题行和列名
    Dark = df_back['Dark spectrum']
    Ref = df_back['Reference spectrum']

    # 测量光谱数据【改文件地址参数】
    df = pd.read_excel(
        'D:/BIT课题研究/微型光谱成像仪/【数据】导电聚合物数据/【数据】测试数据/ava/20230107-ava/device/12-25a 2000r Na-01M PMMA-2% CV/12-25a 2000r Na-01M PMMA-2% CV.xlsx',
        header=5, index_col=0)  # 指定标题行和列名
    # 列数目
    num_cols = df.columns.shape[0]
    # 重命名标题列为测量次数
    df.columns = list(range(0, num_cols))  # 以索引序号重命名
    df.columns.name = 'measurements'

    # -----数据处理
    # dataframe转为array数据格式
    Ref = Ref.values  # (982,)
    Dark = Dark.values  # (982,)
    df_array = df.values  # (982,1623)

    # 扣背景，自动保存的数据默认为抠完背景的
    BackGround = False
    if BackGround:
        for i in range(df_array.shape[1]):
            df_array[:, i] = df_array[:, i] - Dark  # (982,)-(982,)
        Ref = Ref - Dark

    Ref = Ref.reshape(-1, 1)  # (982,)->(982,1)
    df_array = df_array / Ref  # 计算为透过率


    # -----生成观测矩阵
    # range(400,700,10)31个通道实际波长对应的数据行
    waves_num = [231,241,252,263,273,283,294,304,315,325,336,346,357,368,378,389,399,409,420,431,441,452,463,473,484,495,505,516,526,537,548]
    for wave in waves_num:
        print(df.index[wave])
    # 选择行，即每一行为该波长下所有光谱的数据
    array_waves = df_array[waves_num]
    # 测量次数即光谱状态选择
    measure_num = []
    col_1 = 395
    col_2 = 670
    for i in range(col_1, col_2, 9):  # （395，670，11）选出31条光谱
        measure_num.append(i)
    # 选择列，即每一列为一条光谱
    array_measures = array_waves[:, measure_num]
    # 转置成每一行为一条光谱数据
    array_measures = array_measures.T

    return array_measures
