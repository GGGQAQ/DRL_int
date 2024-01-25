'''
在执行find_test前将总体样本的activations_0.data与activations_1.data文件放到了action5文件夹中以统一处理
'''
from utils.utils import deserializer,serializer
from find.top_k_finder import Find
from configurator.configuration import *


# top-k or top-ck算法
algorithm = 'top-ck'


def top_k_find(output_path,find,K=7):

    top_k_node_sets = find.top_k_find()
    for i in range(len(top_k_node_sets)):
        print(f"第{i}层关键神经元个数：{len(top_k_node_sets[i])}，关键神经元：{top_k_node_sets[i]}")

    serializer(f"{output_path}/top-{K}-node-set.data", top_k_node_sets)

def top_ck_find(output_path, find,K=7):

    parameters = deserializer(param_path)
    preds = deserializer(f'{middle_result_path}/action5/actions.data')

    top_ck_node_sets = find.top_ck_find(parameters=parameters, preds=preds)

    for i in range(len(top_ck_node_sets)):
        newbuffer = set()
        for elem in top_ck_node_sets[i]:
            newbuffer.add(elem[1])
        top_ck_node_sets[i] = newbuffer

    for i in range(len(top_ck_node_sets)):
        print(f"第{i}层关键神经元个数：{len(top_ck_node_sets[i])}，关键神经元：{top_ck_node_sets[i]}")


    serializer(f"{output_path}/top-c{K}-node-set.data",top_ck_node_sets)

if __name__ == "__main__":

    for K in [3, 4, 5, 6, 7, 8, 9, 10]:

        # 遍历action0~action5文件夹
        for item in os.listdir(middle_result_path):
            cur_path = os.path.join(middle_result_path, item)

            if os.path.isdir(cur_path):

                # 加载激活值
                activations = dict()
                for i in range(3):
                    activations[i] = deserializer(f"{cur_path}/activations_{i}.data")

                if(algorithm == 'top-k'):
                    find = Find(K, activations)
                    print(item, f"的top-{K}神经元索引如下：")
                    top_k_find(os.path.join(top_neuron_path, item), find, K)

                elif(algorithm == 'top-ck'):
                    find = Find(K, activations)
                    print(item, f"的top-c{K}神经元索引如下：")
                    top_ck_find(os.path.join(top_neuron_path, item), find, K)

                else:
                    print("wrong algorithm")
