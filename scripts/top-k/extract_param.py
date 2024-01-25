'''
提取dqn的神经网络参数放到output/parameters下，方便top-ck算法使用
'''

from configurator.configuration import *


env = gym.make(env_name, render_mode=render_mode)
env.configure(env_config)
# load model
model = DQN.load(model_file, env=env)


# 提取神经网络模型的参数,方便top-ck算法使用
if __name__ == "__main__":

    net = model.q_net.q_net
    data = {}
    for name,parameter in net.named_parameters():

        print(name)

        name = name.split(' ')[0]
        data[name] = parameter.detach().cpu().numpy()

    serializer(param_path, data)
    print(f"神经网络参数存入{param_path}")
