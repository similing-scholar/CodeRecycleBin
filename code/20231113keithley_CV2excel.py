"""将keithley的CV测试数据转换为Excel文件"""
import pandas as pd
import re


def kei_IV2excel(file_path, num_colunm, columns):
    """使用与原始txt数据只有两列"""
    # 读取文件内容
    with open(file_path, 'r') as file:
        content = file.readlines()

    # 匹配一个或多个非 \t 字符，后面跟着 '测试数据' 字符串
    pattern = r'([^\t]+测试数据)'
    matches = re.findall(pattern, content[0])
    print(matches)
    # 处理第一行内容，只保留'I-V测试数据'之前的部分
    content[0] = content[0].split(matches[0])[0].strip()

    # 提取电压和电流数据，并将其转换为浮点数
    data = [[float(value) for value in line.split()] for line in content]  # 从第二行开始处理，并将数据转换为浮点数
    df = pd.DataFrame(data, columns=columns)  # 【可修改第一列，第二列列名】

    # 将数据保存为Excel文件，包含处理后的第一行，指定工作表名称为 'CV'
    excel_output_path = file_path.replace('.txt', '.xlsx')
    with pd.ExcelWriter(excel_output_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, header=True, startrow=0, sheet_name='CV')  # 从第一行开始写入数据，包含标题行

    return print(f"Excel file saved to {excel_output_path}")


if __name__ == '__main__':
    file_path = 'C:/Users/JiaPeng/Desktop/test/CV汇总/20231030a-2000r/ca.txt'
    num_columns = 3  # 原始txt数据有几列，CV填2，方波填3

    if num_columns == 2:
        columns = ['Potential (V)', 'Current (A)']
        kei_IV2excel(file_path, num_columns, columns)
    elif num_columns == 3:
        columns = ['time (s)', 'Potential (V)', 'Current (A)']
        kei_IV2excel(file_path, num_columns, columns)