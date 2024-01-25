'''
highway环境下基于DQN的激活值提取测试脚本
注册钩子函数并在测试过程对样本每一层的激活值进行提取
'''
from configurator.configuration import *
from find.register import Register
import torch

# 测试轮次
Round = 2000

if __name__ == '__main__':

    env = gym.make(env_name)
    env.configure(env_config)
    env.config['duration'] = 50
    env.reset()

    # 加载模型
    model = DQN.load(model_file, env=env)

    # 给q_net注册钩子
    register = Register(model.policy.q_net)

    # buffers[0] [1] [2] 分别存放三层的激活值: [输入层(25),激活层(256), (256)]
    buffers = dict()
    buffers[0] = []
    buffers[1] = []
    buffers[2] = []

    buffers_action = []

    # buffers_by_action 按照网络输出结果分类
    buffers_by_action = dict()
    for i in range(5):
        buffers_by_action[i] = dict()
        buffers_by_action[i][0] = []
        buffers_by_action[i][1] = []
        buffers_by_action[i][2] = []

    for _ in tqdm(range(Round)):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            buffers_action.append(action)
            # 提取激活值
            activations = register.extract()

            # 神经网络层的输入类型为tuple，输出类型为tensor，需要类型转换
            for i in range(len(activations)):
                activations[i] = torch.cat(activations[i])

            for index in range(len(activations)):
                buffers[index].append(activations[index].detach().cpu().numpy())
                buffers_by_action[int(action)][index].append(activations[index].detach().cpu().numpy())
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                print("round:", _)
                print("info:", info)

    for i in range(len(buffers)):
        buffers[i] = np.concatenate(buffers[i], axis=0)
        serializer(f"{middle_result_path}/activations_{i}.data", buffers[i])
        print(f"{middle_result_path}/activations_{i}.data 存储成功！")

    serializer(f"{middle_result_path}/actions.data", buffers_action)
    print(f"{middle_result_path}/actions.data 存储成功！")

    for _ in range(len(buffers_by_action)):
        for i in range(len(buffers_by_action[_])):
            buffers_by_action[_][i] = np.concatenate(buffers_by_action[_][i], axis=0)
            serializer(f"{middle_result_path}/action{_}/activations_{i}.data", buffers_by_action[_][i])
            print(f"{middle_result_path}/action{_}/activations_{i}.data 存储成功！")
