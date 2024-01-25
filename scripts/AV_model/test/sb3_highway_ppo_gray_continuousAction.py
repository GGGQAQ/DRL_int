'''test'''

from configurator.configuration import *

# 测试轮次
Round = 5

# Create environment
env = gym.make(env_name, render_mode=render_mode)
env.configure(env_config)
env.config['duration'] = 50
env.reset()

pprint.pprint(env.config)
# load model
model = PPO.load(model_file, env=env)


Test = True


if(Test == True):
    buffer = list()
    forward_speed = list()
    scaled_speed = list()
    speed = list()
    heading = list()
    cos_heading = list()
    print(env.action_space)
    for _ in range(Round):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            # print(info['kinematics_obs'])
            buffer.append(info['kinematics_obs'])
            forward_speed.append(info['forward_speed'])
            scaled_speed.append(info['scaled_speed'])
            speed.append(info['speed'])
            heading.append(info['heading'])
            cos_heading.append(info['cos_heading'])
            print(action)
            env.render()
    x = [_[0][1] for _ in buffer]
    y = [_[0][2] for _ in buffer]
    vx = [_[0][3] for _ in buffer]
    vy = [_[0][4] for _ in buffer]

    import matplotlib.pyplot as plt

    X = list(range(len(x)))


    # # 绘制折线图
    # plt.plot(X, x, marker='o', linestyle='-')
    # # 添加标题和标签
    # plt.title('x')
    # # 显示图形
    # plt.show()
    #
    # # 绘制折线图
    # plt.plot(X, y, marker='o', linestyle='-')
    # # 添加标题和标签
    # plt.title('y')
    # # 显示图形
    # plt.show()
    #
    # # 绘制折线图
    # plt.plot(X, vx, marker='o', linestyle='-')
    # # 添加标题和标签
    # plt.title('vx')
    # # 显示图形
    # plt.show()
    #
    # # 绘制折线图
    # plt.plot(X, vy, marker='o', linestyle='-')
    # # 添加标题和标签
    # plt.title('vy')
    # # 显示图形
    # plt.show()
    #
    # # 绘制折线图
    # plt.plot(X, forward_speed, marker='o', linestyle='-')
    # # 添加标题和标签
    # plt.title('forward_speed')
    # # 显示图形
    # plt.show()
    #
    # # 绘制折线图
    # plt.plot(X, scaled_speed, marker='o', linestyle='-')
    # # 添加标题和标签
    # plt.title('scaled_speed')
    # # 显示图形
    # plt.show()
    #
    # # 绘制折线图
    # plt.plot(X, speed, marker='o', linestyle='-')
    # # 添加标题和标签
    # plt.title('speed')
    # # 显示图形
    # plt.show()
    #
    # # 绘制折线图
    # plt.plot(X, heading, marker='o', linestyle='-')
    # # 添加标题和标签
    # plt.title('heading')
    # # 显示图形
    # plt.show()
    #
    # # 绘制折线图
    # plt.plot(X, cos_heading, marker='o', linestyle='-')
    # # 添加标题和标签
    # plt.title('cos_heading')
    # # 显示图形
    # plt.show()







