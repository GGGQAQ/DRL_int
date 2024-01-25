'''
初步数据分析
'''
from configurator.configuration import *



# 画一下Apoz分布图

import seaborn as sns
for layer in [0, 1, 2, 3, 4]:
    data = deserializer(f'{hmm_data_path}/activation{layer}.data')
    print(type(data[0]))
    for _ in range(data[0].shape[1]):
        result_array = np.array([arr[0][_] for arr in data])
        APoZ = np.count_nonzero(result_array) / result_array.shape[0]
        print(f"{1 - APoZ}")

        # 使用 seaborn 绘制 KDE 图
        sns.set(style="whitegrid")  # 设置 seaborn 样式
        sns.kdeplot(result_array, color="blue", fill=True)

        # 添加标签和标题
        plt.xlabel('X-axis')
        plt.ylabel('Density')
        plt.title(f'Kernel Density Estimation (KDE) of layer{layer}')

        # 显示图形
        plt.show()