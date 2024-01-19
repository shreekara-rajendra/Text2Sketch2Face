import torch
import torch.nn as nn
from dense_block import dense_block
from transition_layer import *

class RESBLOCK(nn.Module):
  def __init__(self,channels):
    super(RESBLOCK,self).__init__()
    self.block = nn.Sequential(
        nn.Conv2d(in_channels = channels,out_channels = channels,kernel_size = 3,stride = 1,padding = 1,bias =  False),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace = True),
        nn.Conv2d(in_channels = channels,out_channels = channels,kernel_size = 3,stride = 1,padding = 1,bias =  False),
        nn.BatchNorm2d(channels),
    )
    self.relu = nn.ReLU(inplace = True)

  def forward(self,x):
    identity = x
    x = self.block(x)
    x += identity
    x = self.relu(x)
    return x

class G2(nn.Module):
    def __init__(self,att_channels):
        super(G2,self).__init__()
        '''
        input size is 64x64
        '''
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
        '''
        output 64 x 16 x 16
        '''
        self.dense_down1 = nn.Sequential(
            dense_block(64,6,32),
            transition_layer_down(256,0.5)
        )
        '''
        output 128 x 8 x 8
        '''
        self.dense_down2 = nn.Sequential(
            dense_block(128,12,32),
            transition_layer_down(512,0.5)
        )
        '''
        output 256 x 4 x 4
        '''
        self.dense_down3 = nn.Sequential(
            dense_block(256,24,32),
            transition_layer_down(1024,0.5)
        )
        '''
        output is 512 x 2 x 2
        '''
        self.dense_up1 = nn.Sequential(
            dense_block(512,1,256),
            transition_layer_up(768,128)
        )
        self.dense_up2 = nn.Sequential(
            dense_block(384,1,256),
            transition_layer_up(640,128)
        )
        self.dense_up3 = nn.Sequential(
            dense_block(256,1,128),
            transition_layer_up(384,64)
        )
        self.dense_up4 = nn.Sequential(
            dense_block(64,1,64),
            transition_layer_up(128,32)
        )
        self.dense_up5 = nn.Sequential(
            dense_block(32,1,32),
            transition_layer_up(64,16)
        )
        self.concat_attributes = nn.Sequential(
            nn.Conv2d(att_channels+512,512,kernel_size = 3,stride = 1,padding = 1,bias = False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True)
        )
        
    def forward(self,x):
        x = self.block1(x)
        x = self.dense_down1(x)
        x = self.dense_down2(x)
        x = self.dense_down3(x)
        return x
        
        
'''
testing
'''
x = torch.randn(1,1,64,64)        
g = G2(256)
y = g(x)
print(y.shape)