# collect data of <state, activation, action, state>

from configurator.configuration import *
from find.register import Register
from model.model import *


# -------------------------------------------参数配置-------------------------------------------
# 轮次
Round = 2000
# ----------------------------------------------------------------------------------------------

# determine the hidden state
def determine(y0, y1, y2):
    # 输入y0,y1,y2分别是自车、第一近的车、第二近的车的y坐标（注意absolute=True，即相对坐标。）
    # 输出是27种状态对应如下：
    # 自车：左车道，最近车：左车道，第二近车：左车道   state：0
    # 自车：左车道，最近车：左车道，第二近车：中车道   state：1
    # 自车：左车道，最近车：左车道，第二近车：右车道   state：2
    # 自车：左车道，最近车：中车道，第二近车：左车道   state：3
    # ......

    # 返回值为0~26，对应为27种状态
    def lane(y):
        if(y < -0.5):
            return -2
        elif(y < - 0.2):
            return -1
        elif(y < 0.2):
            return 0
        elif(y < 0.5):
            return 1
        else:
            return 2
    if(y0 < 0.2): # 自车左车道
        return 3 * lane(y1) + lane(y2)
    elif(y0 > 0.2 and y0 < 0.5): # 自车中间道
        return 9 + 3 * (lane(y1) + 1) + (lane(y2) + 1)
    else: # 自车右车道
        return 18 + 3 * (lane(y1) + 2) + (lane(y2) + 2)

def collect(model, env, hmm_data_path):
    # 加载模型

    register = Register(model.policy.q_net) # 注册钩子，用来提取激活值

    print(model.q_net)

    buffers = dict()
    # buffers['obs']存放obs
    # buffers['act']存放action
    buffers['obs'] = []
    buffers['state'] = []
    buffers['act'] = []
    buffers['activation'] = dict()
    buffers['activation'][0] = []
    buffers['activation'][1] = []
    buffers['activation'][2] = []
    buffers['activation'][3] = []
    buffers['activation'][4] = []

    for _ in tqdm(range(Round)):
        done = truncated = False
        obs, info = env.reset()
        print(obs.shape)

        while not (done or truncated):

            # record the obs, static, act, activation
            buffers['obs'].append(obs)
            buffers['state'].append(determine(obs[0][2], obs[1][2], obs[2][2]))

            # Predict
            action, _states = model.predict(obs, deterministic=True)

            buffers['act'].append(action)

            activations = register.extract()
            for index in range(len(activations)):
                buffers['activation'][index].append(activations[index].detach().cpu().numpy())
                # buffers_by_action[int(action)][index].append(activations[index].detach().cpu().numpy())

            # one step
            obs, reward, done, truncated, info = env.step(action)
            # env.render()
            # if done or truncated:
            #     print("round:", _)
            #     print("info:", info)


    print("obslen", len(buffers['obs']))
    print("actlen", len(buffers['act']))

    serializer(f"{hmm_data_path}/obs.data", buffers['obs'])
    # print(f"obs存储成功！：{hmm_data_path}/obs.data")

    serializer(f"{hmm_data_path}/state.data", buffers['state'])
    # print(f"state存储成功！：{hmm_data_path}/state.data")

    serializer(f"{hmm_data_path}/act.data", buffers['act'])
    # print(f"act存储成功！：{hmm_data_path}/act.data")

    for i in range(5):
        serializer(f'{hmm_data_path}/activation{i}.data', buffers['activation'][i])
    # print(f"activation存储成功！：{hmm_data_path}/activation0.data~activation4.data")

    print(f"存储成功！：{hmm_data_path}/")

if __name__ == '__main__':

    Train = True # 收集的是hmm的训练数据or测试数据

    # env = gym.make(env_name, render_mode='rgb_array')
    env = gym.make(env_name)
    env.configure(env_config)
    env.config['duration'] = 50
    env.reset()

    model = DQN.load(model_file, env=env)

    if(Train):
        collect(model)


    # if(Train):
    #     for K in [4, 5, 6, 7, 8, 9, 10]:
    #         test_model_name = f'top_{K}_model'
    #         hmm_data_path = f'{Root_dir}/output/hmm/dqn/{model_name}/{test_model_name}'
    #
    #         top_k_neuron_dir = f"{top_neuron_path}/action5/top-{K}-node-set.data"
    #         top_k_sets = deserializer(top_k_neuron_dir)
    #         top_k_model = compress_model(top_k_sets, env, model)
    #         collect(top_k_model, env, hmm_data_path)
    #
    #     for K in [4, 5, 6, 7, 8, 9, 10]:
    #         test_model_name = f'top_c{K}_model'
    #         hmm_data_path = f'{Root_dir}/output/hmm/dqn/{model_name}/{test_model_name}'
    #
    #         top_ck_neuron_dir = f"{top_neuron_path}/action5/top-c{K}-node-set.data"
    #         top_ck_sets = deserializer(top_ck_neuron_dir)
    #         top_ck_model = compress_model(top_ck_sets, env, model)
    #         collect(top_ck_model, env, hmm_data_path)
    # else:
    #     for _ in [0, 1, 2, 3, 4]:
    #         for K in [4, 5, 6, 7, 8, 9, 10]:
    #             test_model_name = f'top_{K}_model'
    #             hmm_data_path = f'{Root_dir}/output/hmm/dqn/{model_name}/{test_model_name}/test{_}'
    #
    #             top_k_neuron_dir = f"{top_neuron_path}/action5/top-{K}-node-set.data"
    #             top_k_sets = deserializer(top_k_neuron_dir)
    #             top_k_model = compress_model(top_k_sets, env, model)
    #             collect(top_k_model, env, hmm_data_path)
    #
    #         for K in [4, 5, 6, 7, 8, 9, 10]:
    #             test_model_name = f'top_c{K}_model'
    #             hmm_data_path = f'{Root_dir}/output/hmm/dqn/{model_name}/{test_model_name}/test{_}'
    #
    #             top_ck_neuron_dir = f"{top_neuron_path}/action5/top-c{K}-node-set.data"
    #             top_ck_sets = deserializer(top_ck_neuron_dir)
    #             top_ck_model = compress_model(top_ck_sets, env, model)
    #             collect(top_ck_model, env, hmm_data_path)




