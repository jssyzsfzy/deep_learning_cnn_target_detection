import os

import re

filepath = r"C:\Users\Administrator\Desktop\yolo\out\outputs"  # 文件夹路径

if __name__ == "__main__":

    if not os.path.exists(filepath):
        print("目录不存在!!")

        os._exit(1)

    filenames = os.listdir(filepath)

    print("文件数目为%i" % len(filenames))
    # num = 0
    for name in filenames:
        filename = name.split('_')[1].split('.')[0]
        # count = name.split(')')[0].split('(')[1]
        # print(name.split(' ')[0])
        newname = 'bird_' + filename + '.xml'  # 若想要在名字前面加字符段，可用此语句

        os.rename(filepath + '\\' + name, filepath + '\\' + newname)
        # num += 1
        print("%s 已经改名完成%s" %(filepath + '\\' + name, filepath + '\\' + newname))
