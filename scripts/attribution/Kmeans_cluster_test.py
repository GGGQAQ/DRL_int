import numpy as np
from sklearn.cluster import KMeans
from utils.utils import *
from configurator.configuration import *
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import torch
# -----------------------------------------------------参数配置-----------------------------------------------------
# 实验环境
Env_name = 'highspeed-fast-v0'
# 模型文件
Model_file = 'highspeed-fast-v0env_100000steps_[256, 256]netarch_2023-09-05date_1-highspeedreward'

# 模型路径
Model_path = f'{Root_dir}\model\AV_model\highway_ppo\{Model_file}\model.zip'
# DQN or PPO ppo和dqn的policy结构不同，区别处理
Algorithm = 'ppo'

# attribution存放路径
Attr_path = f'{Root_dir}\output/attribution_result\ppo\{Model_file}'
# labels/action存放路径
Act_path = f"{Root_dir}/output/obs_act/{Algorithm}/{Model_file}/act.data"

# ------------------------------------------------------------------------------------------------------------------





# 取attribution
attr_01_path = os.path.join(Attr_path, 'attr_0123.data')
attr_01 = deserializer(attr_01_path)

samples_array = np.array([sample.cpu().detach().numpy() for sample in attr_01])
train_samples_array = samples_array[:40000, :]
test_samples_array = samples_array[40000:, :]


# attr_0123_path = os.path.join(Attr_path, 'attr_0123.data')
# attr_0123 = deserializer(attr_0123_path)

# 取labels
true_labels = deserializer(Act_path)
list_of_true_labels = [item for item in true_labels]
# torch.from_numpy(item).cpu().numpy()
np_of_true_labels = np.array(list_of_true_labels)
test_true_labels = np_of_true_labels[40000:]


# 定义要聚类的数量K
K = 5

# 创建KMeans聚类器，并进行训练
kmeans = KMeans(n_clusters=K)
kmeans.fit(samples_array)

# 获取聚类结果
cluster_centers = kmeans.cluster_centers_  # 获取各个簇的质心
labels = kmeans.labels_  # 获取每个样本所属的簇的标签
test_labels = kmeans.predict(test_samples_array)
# labels = kmeans.predict(data)

# 输出聚类结果
print("质心:", cluster_centers)
print("样本所属簇的标签:", labels)

ari = adjusted_rand_score(test_true_labels, test_labels)
ami = adjusted_mutual_info_score(test_true_labels, test_labels)


print("调整兰德指数（ARI）:", ari)
print("调整互信息（AMI）:", ami)






