import torch
from tqdm import tqdm
import numpy as np



# 这个函数和上面compute_integrated_gradient函数的功能一样
'''
输入：模型、input、baseline
输出：softmax最大值对于某一层的积分梯度
'''


'''
PPOnet网络结构：
ActorCriticnet(
  (features_extractor): FlattenExtractor(
    (flatten): Flatten(start_dim=1, end_dim=-1)
  )
  (pi_features_extractor): FlattenExtractor(
    (flatten): Flatten(start_dim=1, end_dim=-1)
  )
  (vf_features_extractor): FlattenExtractor(
    (flatten): Flatten(start_dim=1, end_dim=-1)
  )
  (mlp_extractor): MlpExtractor(
    (net_net): Sequential(
      (0): Linear(in_features=25, out_features=256, bias=True)
      (1): Tanh()
      (2): Linear(in_features=256, out_features=256, bias=True)
      (3): Tanh()
    )
    (value_net): Sequential(
      (0): Linear(in_features=25, out_features=256, bias=True)
      (1): Tanh()
      (2): Linear(in_features=256, out_features=256, bias=True)
      (3): Tanh()
    )
  )
  (action_net): Linear(in_features=256, out_features=5, bias=True)
  (value_net): Linear(in_features=256, out_features=1, bias=True)
)
'''

def integrated_gradients(net, input_data, baseline, steps=50):
    # 计算输入和基准输入之间的差异
    delta = input_data - baseline

    # 初始化积分梯度
    integrated_grads = torch.zeros_like(input_data)

    # 在积分路径上进行插值，并计算梯度
    for alpha in torch.linspace(0, 1, steps):
        interpolated_input = baseline + alpha * delta
        interpolated_input.requires_grad_(True)

        # 计算模型的输出
        output = torch.max(net(interpolated_input))

        # 计算相对于输入的梯度
        grads = torch.autograd.grad(outputs=output, inputs=interpolated_input)[0]

        # 将梯度累加到积分梯度上
        integrated_grads += grads

    # 对积分梯度进行归一化处理
    integrated_grads *= delta / steps
    return integrated_grads


