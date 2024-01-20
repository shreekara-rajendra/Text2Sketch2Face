import torch
import torch.nn as nn
from dense_layer import  layer
'''
every dense layer outputs k feature maps
k is the growth ratw
'''
class dense_block(nn.Module):
    def __init__(self,in_channels,num,k):
        '''
        num: number of layers in the dense block
        '''
        super(dense_block,self).__init__()
        self.block_layers = nn.ModuleList()
        for i in range(num):
            self.block_layers.add_module(f"dense_layer{i}",layer(in_channels + i*k,k))
        self.num = num
    
    def forward(self,x):
        for layer in self.block_layers :
            x = layer(x)
        return x