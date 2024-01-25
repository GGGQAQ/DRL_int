import torch
class Register:
    '''
    registe layer in register_forward_hook,
    and extract the single image's activations in the net,
    extract the model's weight and bias etc.
    这部分是dqn的model.policy.q_net.q_net部分的钩子
    '''
    def __init__(self,net):
        self.activations = []
        self.net = net
        self.net.eval()

        # 下面两个代码段，在关键神经元提取的时候使用前者，收集hmm数据使用后者

        # # 前传的时候就会进行这个
        # def forward_hook(module,input,output):
        #     self.activations.append(input)
        #
        # self.net.q_net[0].register_forward_hook(forward_hook)
        # self.net.q_net[2].register_forward_hook(forward_hook)
        # self.net.q_net[4].register_forward_hook(forward_hook)


        def forward_hook(module, input, output):
            self.activations.append(output)

        self.net.features_extractor.linear[0].register_forward_hook(forward_hook)
        self.net.features_extractor.linear[1].register_forward_hook(forward_hook)
        self.net.q_net[0].register_forward_hook(forward_hook)
        self.net.q_net[1].register_forward_hook(forward_hook)
        self.net.q_net[2].register_forward_hook(forward_hook)
        # self.net.q_net[3].register_forward_hook(forward_hook)
        # self.net.q_net[4].register_forward_hook(forward_hook)


    def extract(self):
        # self.net(input)
        # 赋给activations然后清空掉self.activations
        activations = self.activations
        self.activations = []
        return activations


    def parameter_extract(self):
        parameters = {}
        for name,parameter in self.net.named_parameters():
            parameters[name] = parameter
        return parameters

class Register_2:
    '''
    这部分是DQN的model.policy.q_net.features_extractor.linear部分的钩子
    Sequential(
        (0): Linear(in_features=3072, out_features=512, bias=True)
        (1): ReLU()
    )
    '''
    def __init__(self,net):
        self.activations = []
        self.net = net
        self.net.eval()


        def forward_hook(module, input, output):
            self.activations.append(output)

        self.net.q_net[0].register_forward_hook(forward_hook)
        self.net.q_net[1].register_forward_hook(forward_hook)

    def extract(self):
        # self.net(input)
        # 赋给activations然后清空掉self.activations
        activations = self.activations
        self.activations = []
        return activations


    def parameter_extract(self):
        parameters = {}
        for name,parameter in self.net.named_parameters():
            parameters[name] = parameter
        return parameters

if __name__ == '__main__':
    Register








