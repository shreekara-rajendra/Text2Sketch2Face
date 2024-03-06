import torch
import torch.nn as nn
import torch.nn.functional as F 

## kernel_size mostly 4 for patch wise discriminator
## num->number of layers
class Critic(nn.Module):
    def __init__(self,in_channels,num = 4):
        super(Critic,self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size = 4,stride = 2,padding = 2),
            nn.LeakyReLU(0.2,inplace = True)
        ))
        curr = 64
        maxx =  64*8
        ##same convolution
        for i in range(num):
            self.layers.append(nn.Sequential(
                nn.Conv2d(curr,min(curr*2,maxx),kernel_size = 4,stride = 2,padding = 2),
                nn.BatchNorm2d(min(curr*2,maxx)),
                nn.LeakyReLU(0.2,inplace = True)
            ))
            curr =  min(curr*2,maxx)
        ##valid convolution
        self.layers.append(nn.Sequential(
            nn.Conv2d(curr,min(curr*2,maxx),kernel_size = 4,stride = 1,padding =2),
            nn.BatchNorm2d(min(curr*2,maxx)),
            nn.LeakyReLU(0.2,inplace = True)
        ))
        curr = min(curr*2,maxx)
        self.layers.append(nn.Sequential(
            nn.Conv2d(curr,1,kernel_size = 4,stride = 1,padding = 2),
            nn.Sigmoid()
        ))
            
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
