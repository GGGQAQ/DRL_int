
from configurator.configuration import *


from sklearn.manifold import TSNE
import random
# --------------------------------------------------------参数配置--------------------------------------------------------
# 算法ppo or dqn
Algorithm = 'dqn'
# 测试轮次
Round = 2000
# 测试数据路径
data_path = f'{Root_dir}/output/middle_result/dqn/{model_name}'
# ------------------------------------------------------------------------------------------------------------------------
colors = ['r', 'g', 'b', 'c', 'm']

# 生成示例数据，data是一个包含512维向量的Numpy数组
# data = np.random.rand(100, 512)
# print(data.shape)

buffer_obs = deserializer(data_path + '/activations_0.data')
# buffer_act = deserializer(data_path + '/act.data')
print(buffer_obs[1].shape)
print(type(buffer_obs))

samples_num = 10000
buffer_obs = buffer_obs[:samples_num]
buffer_act = [random.randint(0, 4) for _ in range(samples_num)]

for perplexity in [5, 15, 30, 50]:
    # 创建t-SNE模型
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=300, random_state=42)

    # 将高维数据降至二维
    X_embedded = tsne.fit_transform(buffer_obs)
    print(X_embedded.shape)

    # 绘制t-SNE图
    plt.figure(figsize=(8, 6))
    for i in range(len(X_embedded)):
        x, y = X_embedded[i]  # 获取每个点的坐标
        label = buffer_act[i]  # 获取对应点的颜色类别
        color = colors[label]  # 根据类别选择颜色
        plt.scatter(x, y, c=color)
    plt.title(f't-SNE Clustering {perplexity}perplexity {samples_num}samples_num')
    plt.show()



