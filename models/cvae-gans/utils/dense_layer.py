import torch
import torch.nn as nn



class layer(nn.Module):
    def __init__(self,in_channels,k):
        super(layer,self).__init__()
        '''
        bottleneck has BN + RELU + 1CONV (OUTPUT CHANNELS = 4*K)
        '''
        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels,k*4,kernel_size = 1,bias = False,stride = 1,padding = 0)
        )
        '''
        block has BN + RELU + 3CONV (OUTPUT_CHANNELS = K)
        '''
        self.block = nn.Sequential(
            nn.BatchNorm2d(4*k, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.Conv2d(k*4,k,kernel_size = 3,bias = False,stride = 1,padding = 1)
        )
    
    def forward(self,x):
        x_temp = x
        x = self.bottleneck(x)
        x = self.block(x)
        x = torch.cat((x_temp,x),dim = 1)
        return x

'''
testing
l = layer(64)
t1 = torch.randn((1,64,224,224))
t2 = l(t1)
print(t2.shape)
'''    
