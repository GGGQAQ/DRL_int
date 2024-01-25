import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import datetime
def deserializer(path):
    with open(path,"rb") as f:
        return pickle.load(f)
def serializer(path,obj):
    directory = os.path.dirname(path)

    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path,"wb") as f:
        pickle.dump(obj,f)

def scatter_diag(x, y1, y2):
    # 创建点图
    plt.scatter(x, y1, color='blue', label='y1', marker='o')
    plt.scatter(x, y2, color='red', label='y2', marker='x')

    # 添加标签和标题
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot of y1 and y2')

    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()


def line_diag(x_axis, *y_axes):
    """
    绘制折线图

    参数:
    x_axis: x轴数据
    *y_axes: y轴数据，可变数量的参数

    返回:
    无返回值，直接显示图表
    """
    # 创建图表
    plt.figure()

    # 绘制折线图
    for y_axis in y_axes:
        plt.plot(x_axis, y_axis, marker='o')

    # 添加标题和标签
    plt.title("Line Chart")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # 显示图表
    plt.show()

def print_time(start_time, end_time):
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_date = datetime.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
    print(f"开始时间: {formatted_date}")
    formatted_date = datetime.datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
    print(f"结束时间: {formatted_date}")
    print(f"代码执行耗时：{int(hours)} 小时 {int(minutes)} 分钟 {seconds:.2f} 秒")


if __name__ == "__main__":
    pass