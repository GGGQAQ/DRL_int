'''
数据处理
处理state，将采集到的state转为有限状态机中状态的一种。
'''
from configurator.configuration import *

def determine(state):

    y0 = state[0][2]
    y1 = state[1][2]
    y2 = state[2][2]

    # y0,y1,y2分别是自车、第一近的车、第二近的车的y坐标（注意absolute=True，即相对坐标。）
    # 输出是27种状态对应如下：
    # 自车：左车道，最近车：左车道，第二近车：左车道   state：0
    # 自车：左车道，最近车：左车道，第二近车：中车道   state：1
    # 自车：左车道，最近车：左车道，第二近车：右车道   state：2
    # 自车：左车道，最近车：中车道，第二近车：左车道   state：3
    # ......

    # 返回值为0~26，对应为27种状态
    # lane()判断周围车辆的位置
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


state = deserializer(f'{hmm_data_path}/state.data')
state_of_hmm = [determine(elem) for elem in state]
serializer(f'{hmm_data_path}/state_of_hmm.data', state_of_hmm)

# for _ in range(20):
#     state = deserializer(f'{hmm_data_path}/testdata/test{_}/state.data')
#     state_of_hmm = [determine(elem) for elem in state]
#     serializer(f'{hmm_data_path}/testdata/test{_}/state_of_hmm.data', state_of_hmm)












