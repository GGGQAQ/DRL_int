import cv2
import numpy as np
import numpy as np
from configurator.configuration import *
from utils.utils import deserializer,serializer
from PIL import Image
from tqdm import tqdm


# --------------------------------------------------------参数配置--------------------------------------------------------
# 环境
env_name = "highspeed-fast-v0"
# 实验名/模型文件
Model_file = 'highspeed-fast-v0env_20000steps_GrayscaleImage_Cnnnetarch_2023-10-23date'
# 模型路径
Model_path = f"{Root_dir}/model/AV_model/highway_dqn/{Model_file}/model"
# 算法ppo or dqn
Algorithm = 'dqn'
# 输出文件路径
Output_path = f"{Root_dir}/output/obs_act/{Algorithm}/{Model_file}"
# 测试轮次
Round = 2000
# 测试数据路径
data_path = 'D:/0Projects\DRL_based_AV_interpretability\output\obs_act\dqn\highspeed-fast-v0env_20000steps_GrayscaleImage_Cnnnetarch_2023-10-23date/'

# ------------------------------------------------------------------------------------------------------------------------

buffer = dict()
buffer['obs'] = deserializer(data_path + 'obs.data')
i = 0
for obs in tqdm(buffer['obs']):
    # 读取灰度图像
    gray_image = np.squeeze(buffer['obs'][i], axis=0)

    rgb_image = np.stack([gray_image, gray_image, gray_image], axis=-1)

    # 创建一个 RGB 图像对象
    image = Image.fromarray(rgb_image)

    # 保存RGB图像
    image.save(f"{Root_dir}/output/obs_act/dqn/highspeed-fast-v0env_20000steps_GrayscaleImage_Cnnnetarch_2023-10-23date/rgb_obs/{i}.png")
    i = i + 1