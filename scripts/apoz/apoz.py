'''
统计神经元的apoz并提取模型
'''
from configurator.configuration import *

# 读激活值
layer = 1
activation = deserializer(f'{hmm_data_path}/activation{layer}.data')

activation = np.vstack(activation)

# 计算apoz
# 统计每一位变量为0的概率，即计算每列中0的个数除以样本数量
apoz = np.mean(activation == 0, axis=0)

print(apoz)

non_one_ratio = np.mean(apoz != 1)
print(non_one_ratio)
# 剔除神经元

# 根据保留神经元节点得到压缩模型

# 压缩模型效果评估evaluate_policy


