from configurator.configuration import *
from sklearn.mixture import GaussianMixture

# 预处理
# buffers['state'] = [value // 9 for value in buffers['state']]
# print(buffers['activation'][0].shape)

# def determine(y0, y1, y2):



def transition_matrix():
    # 求状态转移矩阵
    # state_sequence是状态序列列表

    state_sequence = buffers['state']

    # 初始化27x27的转移概率矩阵为零
    transition_matrix = np.zeros((27, 27))
    # transition_matrix = np.zeros((3, 3))

    # 计算状态转移频率
    for i in range(len(state_sequence) - 1):
        current_state = state_sequence[i]
        next_state = state_sequence[i + 1]
        transition_matrix[current_state, next_state] += 1

    # 将转移频率转化为转移概率
    row_sums = transition_matrix.sum(axis=1)
    transition_matrix = transition_matrix / row_sums[:, np.newaxis]

    # 可视化转移概率矩阵
    plt.imshow(transition_matrix, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.title('State Transition Probability Matrix')
    plt.xlabel('Next State')
    plt.ylabel('Current State')
    plt.show()

    serializer(f'{hmm_data_path}/parameters_transition_matrix.data', transition_matrix)
    print(f"transition_matrix存储成功！：{hmm_data_path}/parameters_transition_matrix.data")

def state_obs_dis():
    # 求状态观测函数
    # state是隐藏状态序列，obs是观测序列每个元素是一个 (1, 256) 的 numpy.ndarray

    state = buffers['state']
    obs = buffers['activation']

    # 获取唯一的隐藏状态值
    unique_states = np.unique(state)

    # 初始化字典，用于存储每个隐藏状态下的观测模型参数
    obs_distribution = {}

    flag = True

    # 遍历每个隐藏状态，拟合高斯分布
    for s in unique_states:
        # 获取属于当前隐藏状态的观测数据
        obs_for_state = [obs[idx] for idx, label in enumerate(state) if label == s]

        obs_for_state = np.vstack(obs_for_state)
        # print(len(obs_for_state))

        # 计算观测数据的均值和协方差矩阵
        mean_vector = np.mean(obs_for_state, axis=0)
        cov_matrix = np.cov(obs_for_state, rowvar=False)

        def is_positive_definite(matrix):
            eigenvalues = np.linalg.eigvals(matrix)
            return all(eig > 0 for eig in eigenvalues)

        # 输出结果
        # print("Is Positive Definite:", is_positive_definite(cov_matrix))
        if(not is_positive_definite(cov_matrix)):
            flag = False

        # 创建多变量高斯分布模型
        obs_distribution[s] = multivariate_normal(mean=mean_vector, cov=cov_matrix, allow_singular=True)

    # return flag

    serializer(f'{hmm_data_path}/parameters_obs_distribution/layer{layer}.data', obs_distribution)
    print(f"obs_distribution存储成功！：{hmm_data_path}/parameters_obs_distribution/layer{layer}.data")

    # means = [obj.mean for obj in obs_distribution.values()]
    # covs = [obj.cov for obj in obs_distribution.values()]
    # serializer(f'{Data_path}/parameters_obs_distribution_means.data', means)
    # print(f"means存储成功！：{Data_path}/parameters_obs_distribution_means.data")
    # serializer(f'{Data_path}/parameters_obs_distribution_covs.data', covs)
    # print(f"covs存储成功！：{Data_path}/parameters_obs_distribution_covs.data")

if __name__ == '__main__':

    # 对某层进行分析
    for layer in range(7):
        buffers = dict()
        episode_len = deserializer(f"{hmm_data_path}/episode_len.data")
        # buffers['obs'] = deserializer(f"{hmm_data_path}/obs.data")
        buffers['state'] = deserializer(f"{hmm_data_path}/state_of_hmm.data")
        buffers['activation'] = deserializer(f"{hmm_data_path}/activation{layer}.data")
        # buffers['act'] = deserializer(f"{hmm_data_path}/act.data")

        transition_matrix()

        state_obs_dis()

    # 求转移矩阵

    # 求分布
    # state_obs_dis()


    # 以下是top-k方法的尝试
    # # 对哪一中间层建模layer可取0~4
    # for K in [4, 5, 6, 7, 8, 9, 10]:
    #     print("K=", K)
    #     test_model_name = f'top_{K}_model'
    #     hmm_data_path = f'{Root_dir}/output/hmm/dqn/{model_name}/{test_model_name}'
    #
    #     for layer in [0, 1, 2, 3, 4]:
    #         # 读数据
    #         buffers = dict()
    #         buffers['obs'] = deserializer(f"{hmm_data_path}/obs.data")
    #         buffers['state'] = deserializer(f"{hmm_data_path}/state.data")
    #         buffers['activation'] = deserializer(f"{hmm_data_path}/activation{layer}.data")
    #         buffers['act'] = deserializer(f"{hmm_data_path}/act.data")
    #
    #         # 预处理
    #         # n_components = buffers['activation'][0].shape[1]
    #         # print(buffers['activation'][0].shape)
    #         # data_array = np.vstack(buffers['activation'])
    #         # pca = PCA(n_components=n_components - 20)
    #         # pca.fit(data_array)
    #         # data_pca = pca.transform(data_array)
    #         # buffers['activation'] = data_pca.tolist()
    #         # new_buffer = list()
    #         # for _ in buffers['activation']:
    #         #     new_buffer.append(np.array(_).reshape((1, n_components - 20)))
    #         # buffers['activation'] = new_buffer
    #         state_obs_dis()
    #     # transition_matrix()
