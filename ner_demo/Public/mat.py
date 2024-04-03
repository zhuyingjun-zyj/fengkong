import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from matplotlib.font_manager import FontProperties
import pandas as pd


def matli(df, name):
    import math
    from tqdm import tqdm

    y1 = []
    numbers = []
    n = 10

    df_num = len(df)
    every_epoch_num = math.floor((df_num / n))

    for index in tqdm(range(n)):
        if index < n - 1:
            numbers.append(every_epoch_num)
            df_tem = df[every_epoch_num * index: every_epoch_num * (index + 1)]
        else:
            numbers.append(df_num - every_epoch_num * index)
            df_tem = df[every_epoch_num * index:]
        col_mean = df_tem.mean(axis=0)
        # print(type(col_mean))
        y1.append(float('%.3f' % col_mean["1"]))
        # print(col_mean["1"])

    max_num = max(numbers)

    l = [i for i in range(10)]

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

    fmt = '%.2f%%'
    yticks = mtick.FormatStrFormatter(fmt)  # 设置百分比形式的坐标轴

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(l, y1, 'or-', label=u'score');
    # ax1.yaxis.set_major_formatter(yticks)

    for i, (_x, _y) in enumerate(zip(l, y1)):
        plt.text(_x, _y, y1[i], color='black', fontsize=11)  # 将数值显示在图形上

    ax1.legend(loc=1)
    ax1.set_ylim([0, 1]);
    ax1.set_ylabel('score');

    ax2 = ax1.twinx()  # this is the important function
    plt.bar(l, numbers, alpha=0.3, color='green', label=u'simple num ', width=0.7)
    ax2.legend(loc=2)
    ax2.set_ylabel('simple num ')
    for x, y in enumerate(numbers):
        plt.text(x, y, "%d" % y,
                 horizontalalignment='center',  # 水平居中
                 verticalalignment='top',  # 垂直居中
                 )

    ax2.set_ylim([0, max_num + 50])  # 设置y轴取值范围
    plt.legend(prop={'family': 'SimHei', 'size': 10}, loc="upper left")
    plt.title(name + str(df_num))
    # plt.savefig("./")
    plt.xticks(l)
    plt.show()
