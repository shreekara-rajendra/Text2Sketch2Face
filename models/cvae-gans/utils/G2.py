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
        self.att_channels = att_channels
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
        self.res1 = RESBLOCK(512)
        self.block2 = nn.Sequential(
            nn.Conv2d(17,17,kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(17,1,kernel_size = 3,stride = 1,padding = 1),
            nn.Tanh()
        )      
    def forward(self,x_in,text_att):
        x1 = self.block1(x_in)
        x2 = self.dense_down1(x1)
        x3 = self.dense_down2(x2)
        x4 = self.dense_down3(x3)
        text_att = text_att.view(-1,self.att_channels,1,1)
        text_att = F.interpolate(text_att,size = [2,2],mode = 'nearest')
        x = self.concat_attributes(torch.cat([x4,text_att],dim = 1))
        x = self.res1(x)
        x = self.dense_up1(x)
        x = torch.cat([x,x3],dim = 1)
        x = self.dense_up2(x)
        x = torch.cat([x,x2],dim = 1)
        x = self.dense_up3(x)
        x = self.dense_up4(x)
        x = self.dense_up5(x)
        x = torch.concat([x,x_in],dim = 1)
        '''
        output is 17 x 64 x 64
        '''
        x = self.block2(x)
        x = self.block3(x)
        return x
               
'''
testing
'''
att = torch.randn(1,256)
x = torch.randn(1,1,64,64)        
g = G2(256)
y = g(x,att)
print(y.shape)