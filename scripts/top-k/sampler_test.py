from find.sampler import Sampler
from utils.utils import deserializer,serializer

def top_k_sampler(activations,set_name,K,eps=0.03):
    sampler = Sampler("output/sampler_result/sampler_node_set/top-{}-node-set.data".format(K),activations)
    sampler_result = sampler.sampler()
    # 原代码如下：
    # if set_name != "test" or set_name != "train" or set_name !="val":
    # 改正：
    if set_name != "test" and set_name != "train" and set_name != "val":
        for i in range(len(sampler_result)):
            print(sampler_result[i].shape)
            serializer("output/sampler_result/{}/top-{}-sampler_result_{}_{}.data".format(set_name,K,i,eps),sampler_result[i])
    else:
        for i in range(len(sampler_result)):
            print(sampler_result[i].shape)
            serializer("output/sampler_result/{}/top-{}-sampler_result_{}.data".format(set_name,K,i),sampler_result[i])

def top_ck_sampler(activations,set_name,K,eps=0.03):
    sampler = Sampler("output/sampler_result/sampler_node_set/top-ck-{}-node-set.data".format(K),activations)
    sampler_result = sampler.sampler()
    # 原代码如下：
    # if set_name != "test" or set_name != "train" or set_name !="val":
    # 改正：
    if set_name != "test" and set_name != "train" and set_name != "val":
        for i in range(len(sampler_result)):
            print(sampler_result[i].shape)
            serializer("output/sampler_result/{}/top-ck-{}-sampler_result_{}_{}.data".format(set_name,K,i,eps),sampler_result[i])
    else:
        for i in range(len(sampler_result)):
            print(sampler_result[i].shape)
            serializer("output/sampler_result/{}/top-ck-{}-sampler_result_{}.data".format(set_name,K,i),sampler_result[i])

if __name__ == "__main__":
    K = 3
    # set_name = "test"
    # 改成train
    # set_name = "train"
    # 改成val
    set_name = "val"
    eps = 0.03

    buffers = dict()
    for i in range(3):
        buffers[i] = deserializer("output/middle_result/{}_{}.data".format(set_name,i,eps))
    # top_k_sampler(buffers,set_name,K,eps)
    top_ck_sampler(buffers,set_name,K,eps)

