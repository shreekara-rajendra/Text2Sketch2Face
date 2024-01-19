import torch
import torch.nn as nn
import torch.nn.functional as F
'''
transition layer between denseblocks
'''
class transition_layer_down (nn.Module):
    def __init__(self,in_channels,compression_ratio,no_comp = False):
        super(transition_layer_down,self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels = in_channels , out_channels = int(in_channels*compression_ratio) ,kernel_size = 1 ,stride = 1 ,padding = 0, bias=False ),
            nn.AvgPool2d(kernel_size = 2, stride = 2,padding = 0)
        )
    
    def forward(self,x):
        x = self.block(x)
        return x
    
class transition_layer_up (nn.Module):
    def __init__(self,in_channels,out_channels):
        super(transition_layer_up,self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels = in_channels , out_channels=out_channels,kernel_size = 1 ,stride = 1 ,padding = 0, bias=False ),
        )
    
    def forward(self,x):
        x = self.block(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return x
    
