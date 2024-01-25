import numpy as np

from configurator.configuration import *
from matplotlib.ticker import FuncFormatter


#

def calculate(samples):
    nums = len(samples)
    counter = np.zeros(samples[0].shape[1])
    for activation in samples:
        index = activation[0] == 0
        counter[index] += 1

    counter /= nums
    print(nums)
    return counter
    # for activation in samples:
    #     counter+=activation[0]
    #
    # counter/=nums
# print(counter)ewqdd4eeweeqw   A


if __name__ == '__main__':

    # hmm_data_path = f'{Root_dir}/output/hmm/dqn/highway-fast-modify-v0env_100000steps_[256]netarch_2023-12-26date'

    for layer in[0, 1, 2, 3, 4, 5, 6]:
        activation = deserializer(f'{hmm_data_path}/activation{layer}.data')
        data = calculate(activation)

        # plt.hist(data, bins=100, edgecolor='black', alpha=0.7)
        plt.hist(data, bins=100, edgecolor='black', alpha=0.7, density=True)


        plt.yscale('log')  # 使用对数刻度
        plt.title(f'layer={layer}, num={data.shape[0]}')
        plt.xlabel('')
        plt.ylabel('precent')
        plt.show()