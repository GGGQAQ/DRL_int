'''
用于测试的简单model
'''
import torch
import torch.nn as nn


if __name__ == '__main__':
    print(torch.cuda.is_available())

    def forward_hook_fn(module, input, output):
        print("****************************************************************************************************************")
        print("module", module.name)
        # print('weight', module.weight.data)
        # print('bias', module.bias.data)
        print('input', input)
        print('output', output)

    # 创建网络实例
    net = MyNetwork()
    net.fc1.register_forward_hook(forward_hook_fn)
    net.relu1.register_forward_hook(forward_hook_fn)
    net.fc2.register_forward_hook(forward_hook_fn)
    net.relu2.register_forward_hook(forward_hook_fn)
    net.fc3.register_forward_hook(forward_hook_fn)
    x = torch.randn(128)
    print("x", x)
    y = net(x)
    print("y", y)



    # 打印网络结构
    # print(net)





