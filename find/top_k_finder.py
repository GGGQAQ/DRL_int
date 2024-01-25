import torch
import numpy as np

class Find:
    def __init__(self,K,activations):
        self.K = K
        # activations是所有样本的激活值
        self.activations = activations
    
    # 提取每一个样本的Top_k_node并返回top_k_node_sets
    def top_k_find(self):
        top_k_node_sets = {}
        # len(self.activations)是激活层的数量，self.activations=buffers，而buffers[0]和buffers[1]分别是激活层1和2的激活值，所以值是2，同时返回值set的长度也是2，即两层的top-k神经元
        for index in range(len(self.activations)):
            # _：样本数
            _ = len(self.activations[index])
            re_activation = self.activations[index].reshape(_,-1)
            top_k_node_sets[index] = self.__top_k_finder(re_activation)
        return top_k_node_sets

    def __top_k_finder(self,re_activation):
        # _：样本数
        # a_l:reshape后的长度
        _,a_l = re_activation.shape
        top_K_node_set = set()
        for index in range(_):
            single_activation = re_activation[index]
            # single_activation = single_activation.reshape(-1)
            # arg_sorted_activation为激活值从大到小排序的索引
            arg_sorted_activation = np.argsort(single_activation)[::-1]

            for i in range(self.K):
                idx = arg_sorted_activation[i]
                # channel = int(idx / a_l)
                position = idx % a_l
                if single_activation[idx] > 0:
                    # top_K_node_set.add((channel,position))
                    top_K_node_set.add(position)
        # print(top_K_node_set)
        return top_K_node_set

    def top_ck_find(self,parameters,preds):

        self.weight_list = ["0.weight", "2.weight", "4.weight"]
        self.weights = dict()

        # 用来记录每个样本的关键神经元结点
        self.target_node_sets = dict()
        for i in range(len(preds)):
            _ = set()
            _.add((0,int(preds[i])))
            self.target_node_sets[i] = _
        # 最终关键神经元集合
        top_ck_node_sets = {}

        for weight_name in self.weight_list:
            self.weights[weight_name] = parameters[weight_name]

        for index in range(len(self.weight_list)-1,-1,-1):

            # if len(self.activations[index].shape) == 4:
            #     _, a_c, a_h, a_w = self.activations[index].shape
            #     re_activation = self.activations[index].reshape(_, a_c, -1)
            #     top_ck_node_sets[index] = self.__top_k_finder(re_activation)
            # else:
            weight = self.weights[self.weight_list[index]]
            top_ck_node_sets[index], self.target_node_sets = self.__top_ck_finder(self.activations[index], self.target_node_sets,weight)
        return top_ck_node_sets


    def __top_ck_finder(self,activation,target_node_sets,weight):
        _, a_l = activation.shape
        top_ck_node_set = set()

        for index in range(_):
            node_set = set()
            target_node_set = target_node_sets[index]
            x = activation[index]
            # print(type(target_node_set))
            # if type(target_node_set) is set:
            for _,node in target_node_set:
                col = node
                weight_ = weight[col]
                mul_result = x * weight_
                argsort = np.argsort(mul_result)[::-1]
                for i in range(self.K):
                    node_set.add((0, argsort[i]))
                    top_ck_node_set.add((0, argsort[i]))

            target_node_sets[index] = node_set

        return top_ck_node_set,target_node_sets


    # def top_ck_find_baseOnResult(self):








