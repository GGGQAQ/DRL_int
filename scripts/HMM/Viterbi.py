
from configurator.configuration import *


def compare_arrays(array1, array2):

    if array1.shape != array2.shape:
        print("wrong size")
        return

    # 计算相同位数的数量
    common_elements = np.sum(array1 == array2)

    # 计算总位数
    total_elements = np.prod(array1.shape)

    # 计算相同位数的百分比
    common_percentage = (common_elements / total_elements) # * 100
    return common_percentage

def hmm_model_test(model, test_path):
    # 输入hmm模型, 测试集路径, 输出在若干测试集上的结果
    decoded_states = np.empty(0)
    objective_state = np.empty(0)

    for _ in range(20):# 测试集个数
        obs_test = deserializer(f'{test_path}/test{_}/activation{layer}.data')
        objective_state_tmp = deserializer(f'{test_path}/test{_}/state_of_hmm.data')
        objective_state_tmp = np.array(objective_state_tmp)

        # 数据处理
        n = obs_test[0].shape[1]
        new_obs_test = list()
        for _ in obs_test:
            new_obs_test.append(_.reshape(n, ))

        # 使用维特比算法解码最佳路径
        decoded_states_tmp = model.decode(np.array(new_obs_test))
        decoded_states = np.concatenate((decoded_states, decoded_states_tmp[1]))
        objective_state = np.concatenate((objective_state, objective_state_tmp))

        # 画图
        scatter_diag([_ for  _ in range(len(decoded_states_tmp[1].tolist()))], decoded_states_tmp[1].tolist(), objective_state_tmp.tolist())

        # de = np.array(decoded_states_tmp[1])
        # de = np.floor_divide(de, 9)
        # objective_state_tmp = np.floor_divide(objective_state_tmp, 9)
        # print(compare_arrays(de, objective_state_tmp))
        # scatter_diag([_ for _ in range(len(decoded_states_tmp[1].tolist()))], de.tolist(), objective_state_tmp.tolist())


    print(compare_arrays(decoded_states, objective_state))


if __name__ == '__main__':

    layer = 4
    transmatrix = deserializer(f'{hmm_data_path}/parameters_transition_matrix.data')
    Gauss_distribution = deserializer(f'{hmm_data_path}/parameters_obs_distribution/layer{layer}.data')

    means = [obj.mean for obj in Gauss_distribution.values()]
    covs = [obj.cov for obj in Gauss_distribution.values()]

    # 创建GaussianHMM模型
    model = hmm.GaussianHMM(n_components=27, n_iter=100, covariance_type='full')

    # 定义模型参数（均值、协方差矩阵、初始概率、转移概率）
    model.startprob_ = np.full(27, 1/27)

    model.transmat_ = transmatrix

    model.means_ = np.array(means)  # 每个隐藏状态的均值
    model.covars_ = np.array(covs)  # 每个隐藏状态的协方差矩阵

    hmm_model_test(model, f'{hmm_data_path}/testdata')




    # for K in [4, 5, 6, 7, 8, 9, 10]:
    #     print(f'K={K}')
    #     for layer in [2]:
    #         print(f'layer={layer}')
    #         for test_model_name in [f'top_{K}_model', f'top_c{K}_model']:
    #             print(f'{test_model_name}')
    #             try:
    #                 test_model_name = f'top_{K}_model'
    #                 hmm_data_path = f'{Root_dir}/output/hmm/dqn/{model_name}/{test_model_name}'
    #                 transmatrix = deserializer(f'{hmm_data_path}/parameters_transition_matrix.data')
    #                 Gauss_distribution = deserializer(f'{hmm_data_path}/parameters_obs_distribution/layer{layer}.data')
    #
    #                 means = [obj.mean for obj in Gauss_distribution.values()]
    #                 covs = [obj.cov for obj in Gauss_distribution.values()]
    #
    #                 # 创建GaussianHMM模型
    #                 model = hmm.GaussianHMM(n_components=27, n_iter=100, covariance_type='full')
    #
    #                 # 定义模型参数（均值、协方差矩阵、初始概率、转移概率）
    #                 model.startprob_ = np.full(27, 1/27)
    #
    #                 model.transmat_ = transmatrix
    #
    #                 model.means_ = np.array(means)  # 每个隐藏状态的均值
    #                 model.covars_ = np.array(covs)  # 每个隐藏状态的协方差矩阵
    #
    #                 hmm_model_test(model, hmm_data_path)
    #
    #
    #             except Exception as e:
    #                 # 发生异常时执行的代码
    #                 continue


