"""
将根目录文件夹的子文件夹下的所有图片复制到新的文件夹中
"""
import os
import shutil


def copy_png_images(source_folder, target_folder):
    # 遍历源文件夹及其子文件夹
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.png'):
                # 构建源文件的完整路径
                source_file = os.path.join(root, file)
                # 构建目标文件的完整路径
                target_file = os.path.join(target_folder, file)
                # 复制文件
                shutil.copy(source_file, target_file)
    return print('PNG图片已成功复制到目标文件夹.')


if __name__ == "__main__":
    source_folder = 'D:/BIT课题研究/微型光谱成像仪/【数据】导电聚合物数据/郁海凡登记/20230828数据/film/透过'
    target_folder = 'D:/BIT课题研究/微型光谱成像仪/【数据】导电聚合物数据/PJ_数据整理/化学氧化聚合数据/薄膜/[images+excels]薄膜光谱'
    copy_png_images(source_folder, target_folder)
