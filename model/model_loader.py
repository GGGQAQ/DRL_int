import torch

# 模型的加载方法
def load_model(Net,model_path,**kwargs):
    net = Net(**kwargs)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(model_path,map_location=device))
    return net