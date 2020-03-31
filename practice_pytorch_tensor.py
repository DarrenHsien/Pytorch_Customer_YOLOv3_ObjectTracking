import torch
import numpy as np
import albumentations as A



#--創立Tensor
"""
data - 可以是list, tuple, numpy array, scalar或其他類型

dtype - 可以返回想要的tensor类型

device - 可以指定返回的設備

requires_grad - 可以指定是否進行紀錄圖的操作，默認為False
"""
print(torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]]))

# creates a torch.cuda.DoubleTensor
print(torch.tensor([[0.11111, 0.222222, 0.3333333]],
                 dtype=torch.float64,
                 device=torch.device('cuda:0')))  

#--Numpy to Tensor
a = np.array([1, 2, 3])
t = torch.from_numpy(a)
print(t)

#--返回大小为sizes的零矩阵
sizes = [10,10]
print(torch.zeros(*sizes, out=None))
#--返回与input相同size的零矩阵
print(torch.zeros_like(t))
#--f返回大小为sizes的单位矩阵
print(torch.ones(*sizes, out=None))
#--返回与input相同size的单位矩阵
print(torch.zeros_like(t))
#--返回大小为sizes,单位值为fill_value的矩阵
print(torch.full((5,5), 10))
#--返回与input相同size，单位值为fill_value的矩阵
print(torch.full_like(t,2))
#--返回从start到end, 单位步长为step的1-d tensor
print(torch.arange(start=0, end=10, step=1))
#--返回从start到end, 间隔中的插值数目为steps的1-d tensor
print(torch.linspace(start=0, end=100000, steps=100))
#--返回1-d tensor ，从10^start到10^end的steps个对数间隔
print(torch.logspace(start=0, end=3, steps=100))



#torch squeeze
a = torch.randn(1,3)
print(a)
print(a.shape)
c = a.unsqueeze(0)
print(c)
print(c.shape)

