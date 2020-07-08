# -*- coding: utf-8 -*-
"""
    Spyder Editor
    Author: Yihang Bao
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

#fake data
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())
x,y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_features,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)
    
    def forward(self,x): #前向传播方式定义
        x = torch.relu(self.hidden(x)) #套上激活函数
        x = self.predict(x)
        return x
     
net = Net(1,10,1) #输入层1个，隐藏层10个单元，输出层1个
print(net)

plt.ion() #实时图
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()
for t in range(100):
    prediction = net(x)
    loss = loss_func(prediction,y)
    optimizer.zero_grad()
    loss.backward() #梯度反向传播
    optimizer.step() #用优化器去更新前面的参数
    #以下可视化部分
    if t%5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
        plt.text(0.5,0,'Loss=%.4f' % loss.item())
        plt.pause(0.1)
        
plt.ioff() #实时图
plt.show()
        