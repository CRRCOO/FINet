"""
Modified according to
https://github.com/GewelsJI/SINet-V2/blob/main/utils/fps.py
https://github.com/yuhuan-wu/MobileSal/blob/master/speed_test.py
""" 

import torch, time
import numpy as np

# define model
from Model.FINet import FINet
model = FINet(backbone='efficientb0', channels=(8, 24, 32, 64))

# 定义numpy输入矩阵
bs = 20
test_img = np.random.random((bs, 3, 384, 384)).astype('float32')
test_high = np.random.random((bs, 96, 48, 48)).astype('float32')
test_low = np.random.random((bs, 96, 48, 48)).astype('float32')

# 定义 pytorch & jittor 输入矩阵
pytorch_test_img = torch.Tensor(test_img).cuda()
test_high = torch.Tensor(test_high).cuda()
test_low = torch.Tensor(test_low).cuda()

# 跑turns次前向求平均值
turns = 100

# 定义 pytorch & jittor 的xxx模型，如vgg
pytorch_model = model.cuda()

# 把模型都设置为eval来防止dropout层对输出结果的随机影响
pytorch_model.eval()

# 测试Pytorch一次前向传播的平均用时
for i in range(10):
	# pytorch_result = pytorch_model(pytorch_test_img)  # Pytorch热身
	pytorch_result = pytorch_model(pytorch_test_img, test_high, test_low)  # Pytorch热身
torch.cuda.synchronize()
sta = time.time()
for i in range(turns):
	# pytorch_result = pytorch_model(pytorch_test_img)
	pytorch_result = pytorch_model(pytorch_test_img, test_high, test_low)
torch.cuda.synchronize()  # 只有运行了torch.cuda.synchronize()才会真正地运行，时间才是有效的，因此执行forward前后都要执行这句话
end = time.time()
tc_time = round((end - sta) / turns, 5)  # 执行turns次的平均时间，输出时保留5位小数
tc_fps = round(bs * turns / (end - sta), 0)  # 计算FPS
print(f"- Pytorch forward average time cost: {tc_time}, Batch Size: {bs}, FPS: {tc_fps}")
