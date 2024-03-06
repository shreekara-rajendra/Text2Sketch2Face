import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def upsample(in_channels,out_channels):
    block = nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = 3,stride = 1,padding = 1,bias =  False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True),
    )
    return block

class Decoder1(nn.Module):
  def __init__(self,sketch_channels,attributes_channels,noise_channels,start_channels = 64,attribute_final = 256):
    super(Decoder1,self).__init__()
    self.sketch_channels = sketch_channels
    self.attributes_channels = attributes_channels
    self.noise_channels = noise_channels
    self.start_channels = start_channels
    self.attribute_final = attribute_final
    ## decode the sketch_code to sketch
    self.upsampling_block = nn.Sequential(
        upsample(start_channels ,start_channels // 2),
        upsample(start_channels // 2,start_channels // 4),
        upsample(start_channels // 4,start_channels // 8),
        upsample(start_channels // 8,start_channels // 16),
    )

    self.block = nn.Sequential(
        nn.Conv2d(in_channels = start_channels // 16,out_channels = sketch_channels,kernel_size = 3,stride = 1,padding = 1,bias =  False),
        nn.Tanh()
    )

  def forward(self,sketch_code,noise_code):

    sketch_code = sketch_code.view(-1,self.start_channels,4,4)
    sketch = self.upsampling_block(sketch_code)
    real_sketch = self.block(sketch)

    noise_code = noise_code.view(-1,self.start_channels,4,4)
    noise = self.upsampling_block(noise_code)
    fake_sketch = self.block(noise)
    return real_sketch,fake_sketch
