代码：
## output中的数据
- hmm：存放日志数据<act, activation, obs, state>
- middle_result：存放神经网络中间激活值
- sampler_result: 存放top-k神经元索引
- parameters：存放模型参数供top-ck算法使用

## scripts中的所有脚本
- attribution：
  1. intergrated_gradients_test.py
  2. Kmeans_cluster_test.py
- model：
  - AV_model：存放DQN模型
  - parameters：存放DQN中神经网络的参数以供top-ck算法使用
- top-k：关于top-k算法的视线
  1. compressed_model_test.py
  2. find_test.py
  3. register_test.py
  4. sampler_test.py
- output:

  - middle_result: 存放神经网络中间激活值
  
  


## 之前的方法，top-ck算法的实现，先对模型压缩试试看
1. 运行scripts/AV_model/sb3_highway_dqn_train.py，得到待测模型
2. 运行scripts/top-k/register_test.py，得到网络的激活值数据（两个Relu层的输出值），存放在output/middle_result下，注意根目录下存放的是所有样本的激活值，而action0~4存放动作结果为0~4的样本
3. 运行scripts/top-k/find_test.py，得到关键神经元结点集合，存放在output/sampler_result下，
4. 运行scripts/top-k/compressed_model_test.py，对压缩后模型进行有效性验证

## 积分梯度归因算法
1. 运行scripts/AV_model/sb3_highway_dqn_train.py，得到待测模型
2. 运行scripts/top-k/register_test.py，得到网络的激活值数据（两个Relu层的输出值），存放在output/middle_result下，注意action5存放的是所有样本，而action0~4存放动作结果为0~4的样本
3. 积分梯度归因
4. 使用归因得到的结果进行聚类看一下是否有明显的效果

## 对DQN使用top-k算法进行模型压缩
1. 运行scripts/AV_model/train/official_script_sb3_highway_dqn.py，得到待测模型
2. 运行scripts/top-k/register_test.py，得到网络的激活值数据（两个Relu层的输出值），存放在output/middle_result下，注意根目录下存放的是所有样本的激活值，为了方便后续实验需要新建action5文件夹将所有样本的激活值放到action5文件夹下，而action0~4存放动作结果为0~4的样本
3. 运行scripts/top-k/find_test.py，得到关键神经元结点集合，存放在output/sampler_result下
4. 运行scripts/top-k/compressed_model_test.py，对压缩后模型进行有效性验证

## 对DQN使用top-ck算法进行模型压缩
1. 运行scripts/AV_model/train/official_script_sb3_highway_dqn.py，得到待测模型
2. 运行scripts/top-k/register_test.py，得到网络的激活值数据（三个Relu层的输出值），存放在output/middle_result下，注意根目录下存放的是所有样本的激活值，为了方便后续实验需要新建action5文件夹将所有样本的激活值放到action5文件夹下，而action0~4存放动作结果为0~4的样本
3. 运行scripts/top-k/find_test.py，得到关键神经元结点集合，存放在output/sampler_result下
4. 运行scripts/top-k/compressed_model_test.py，对压缩后模型进行有效性验证

## hmm模型
1. 得到待测模型（）
2. 数据收集：scripts/AV_model/data_collect/data_collect_of_hmm.py


## 新的DQN模型
1. scripts/AV_model/train/official_sb3_highway_dqn_grayscale.py是dqn的且以灰度图为观测空间的训练脚本
2. scripts/AV_model/data_collect/data_collect_of_hmm_grayscalesobs.py-提取hmm数据保存到output/hmm/dqn/{model_name}下
3. scripts/HMM/solve_parameters.py求解hmm模型参数
4. scripts/HMM/Viterbi.py求解观测序列，测试效果
4. 