import pickle
from find.register import Register
# from configurator.configuration import activations_idx
from utils.utils import deserializer
import numpy as np
class Sampler:
    def __init__(self,sampler_node_path,activations):
        self.sampler_nodes = deserializer(sampler_node_path)
        self.activations = activations
    def sampler(self):
        sampler_result = dict()

        for index in range(len(self.activations)):
            activation = self.activations[index]
            sampler_node = self.sampler_nodes[index]
            # print(activation.shape)
            if len(activation.shape) == 4:
                _,a_c,a_w,a_h = activation.shape
                activation = activation.reshape(_,a_c,-1)
            else:
                _, a_l = activation.shape
                activation = activation.reshape(_,1,-1)
            li = []
            # print(activation.shape)
            for node in sampler_node:
                c,position = node
                tmp = activation[:,c,position]
                tmp = tmp[np.newaxis,:].T
                li.append(tmp)
            sampler_result[index] = np.concatenate(li,axis=1)
        return sampler_result



