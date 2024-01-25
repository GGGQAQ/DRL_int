import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_loader import load_model
from configurator.configuration import *
import torchvision.models as models

# 网络模型
class Net01(torch.nn.Module):
    def __init__(self):
        super(Net01,self).__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(16,36,kernel_size=3,stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(36,2)
    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.gap(x)
        x = torch.flatten(x,start_dim=1)
        output = self.fc(x)
        return output

class Net02(torch.nn.Module):
    def __init__(self):
        super(Net02,self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64,2)
    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.gap(x)
        x = torch.flatten(x,start_dim=1)
        output = self.fc(x)
        return output

class Net03(torch.nn.Module):
    def __init__(self):
        super(Net03,self).__init__()
        # limit the input image size  (224，224)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 36, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(36*54*54,100)
        self.fc2 = nn.Linear(100,2)

    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = torch.flatten(x,start_dim=1)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output

class Net04(models.AlexNet):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)


class FlattenExtractor(nn.Module):
    def __init__(self):
        super(FlattenExtractor, self).__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        return self.flatten(x)

class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()

        self.features_extractor = FlattenExtractor()

        self.q_net = nn.Sequential(
            nn.Linear(25, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        x = self.features_extractor(x)
        q_values = self.q_net(x)
        return q_values

class Compressed_QNetwork(torch.nn.Module):
    def __init__(self, num1, num2):
        super(Compressed_QNetwork, self).__init__()

        # self.features_extractor = FlattenExtractor()

        self.q_net = nn.Sequential(
            nn.Linear(25, num1),
            nn.ReLU(),
            nn.Linear(num1, num2),
            nn.ReLU(),
            nn.Linear(num2, 5)
        )

    def forward(self, x):
        # x = self.features_extractor(x)
        x = x.reshape(-1)
        q_values = self.q_net(x)
        return q_values

    def compress(self, set1, set2, source_model):
        for i, idx in enumerate(set1):
            self.q_net[0].weight.data[i] = source_model.q_net[0].weight.data[idx]
            self.q_net[0].bias.data[i] = source_model.q_net[0].bias.data[idx]

        for i, row_idx in enumerate(set2):
            for j, col_idx in enumerate(set1):
                self.q_net[2].weight.data[i, j] = source_model.q_net[2].weight.data[row_idx, col_idx]
            self.q_net[2].bias.data[i] = source_model.q_net[2].bias.data[row_idx]

        for i, idx in enumerate(set2):
            self.q_net[4].weight.data[:, i] = source_model.q_net[4].weight.data[:, idx]

        self.q_net[4].bias.data = source_model.q_net[4].bias.data

        return


def compress_model(sets, env, model_original):
    model_compressed = DQN('MlpPolicy', env, policy_kwargs=dict(net_arch=[len(sets[0]), len(sets[1])]), verbose=1)

    # 根据原有模型的top-k神经元为新的压缩网络模型赋值
    compressed_net = Compressed_QNetwork(len(sets[1]), len(sets[2]))
    compressed_net.to('cuda:0')

    # 注意是sets[1] sets[2]
    compressed_net.compress(sets[1], sets[2], model_original.policy.q_net)

    # 将压缩网络模型参数赋值给DQN模型
    model_compressed.q_net.q_net[0].weight = nn.Parameter(compressed_net.q_net[0].weight)
    model_compressed.q_net.q_net[0].bias = nn.Parameter(compressed_net.q_net[0].bias)

    model_compressed.q_net.q_net[2].weight = nn.Parameter(compressed_net.q_net[2].weight)
    model_compressed.q_net.q_net[2].bias = nn.Parameter(compressed_net.q_net[2].bias)

    model_compressed.q_net.q_net[4].weight = nn.Parameter(compressed_net.q_net[4].weight)
    model_compressed.q_net.q_net[4].bias = nn.Parameter(compressed_net.q_net[4].bias)

    return model_compressed
