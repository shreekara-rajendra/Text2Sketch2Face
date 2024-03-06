import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Encoder1(nn.Module):
  def __init__(self,sketch_channels,attributes_channels,noise_dim,start_channels = 64,attribute_final = 256):
    super(Encoder1,self).__init__()
    self.sketch_channels = sketch_channels
    self.attributes_channels = attributes_channels
    self.noise_dim = noise_dim
    self.start_channels = start_channels
    self.attribute_final = attribute_final
    self.relu = nn.ReLU()

    ## 1024 dimensions for latent space
    self.latent_dim = self.start_channels * 16

    ## disentangle the text attributes
    self.disentangle = nn.Sequential(
          nn.Linear(attributes_channels,attribute_final, bias=False),
          nn.BatchNorm1d(attribute_final),
          nn.ReLU(inplace = True),
    )

    ## double the channels for mean,std
    self.double = nn.Sequential(
          nn.Linear((start_channels*16),(start_channels*32),bias=True),
          nn.ReLU(inplace = True),
    )

    ## bringing back to 1024 from 1280
    self.compress = nn.Sequential(
          nn.Linear((start_channels*16) + attribute_final,(start_channels*16),bias=False),
          nn.BatchNorm1d((start_channels*16)),
          nn.ReLU(inplace = True),
    )

    ## encode the sketch using the five conv layers
    self.EncodeSketch = nn.Sequential(
          nn.Conv2d(sketch_channels,start_channels, 5, 1, 2, bias=False),
          nn.ReLU(inplace = True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(start_channels, start_channels * 2, 5, 1, 2, bias=False),
          nn.BatchNorm2d(start_channels * 2),
          nn.ReLU(inplace = True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(start_channels * 2, start_channels * 4, 3, 1, 1, bias=False),
          nn.BatchNorm2d(start_channels * 4),
          nn.ReLU(inplace = True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(start_channels * 4, start_channels* 8, 3, 2, 1, bias=False),
          nn.BatchNorm2d(start_channels * 8),
          nn.ReLU(inplace = True),  # 4 x 4
          nn.Conv2d(start_channels * 8, start_channels * 16, 4, 1, 0, bias=False),
          nn.ReLU(inplace = True),  # 1 x 1
          nn.Dropout(0.5)
    )

    ## encode the noise vector
    self.EncodeNoise = nn.Sequential(
        nn.Linear(noise_dim, start_channels * 16, bias=False),
        nn.BatchNorm1d(start_channels * 16),
        nn.ReLU(inplace = True),
    )

  ## reparametrization trick
  def reparametrize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        temp = torch.randn(std.size(),device = std.device)
        return temp.mul(std).add_(mean)

  ## latent to distribution
  def latent_distribution(self, text_att):
        temp = self.double(text_att)
        mean = temp[:, :(self.start_channels*16)]
        logvar = temp[:, (self.start_channels*16):]
        return mean, logvar

  def forward(self,sketch,text_att,noise):

    disentangled_text = self.disentangle(text_att)
    sketch_encode = self.EncodeSketch(sketch)
    sketch_encode = sketch_encode.view(-1,self.start_channels*16)

    ## concatenate sketch_encode and disentangled_Text to get 1280
    sketch_text = torch.cat((sketch_encode,disentangled_text),dim = 1)
    ## bring back to 1024
    sketch_text = self.compress(sketch_text)
    ## get mean and log(variance) for network1 -> (sketch + Text)
    st_mean,st_logvar = self.latent_distribution(sketch_text)
    ## input for decoder
    sketch_code = self.reparametrize(st_mean,st_logvar)

    ## encode noise
    noise_encode = self.EncodeNoise(noise)
    ## concatenate noise_encode and disentangled_Text to get 1280
    noise_text = torch.cat((noise_encode,disentangled_text),dim = 1)
    ## bring back to 1024
    noise_text = self.compress(noise_text)
    ## get mean and log(variance) for network1 -> (noise + Text)
    nt_mean,nt_logvar = self.latent_distribution(noise_text)
    ##input for decoder
    noise_code = self.reparametrize(nt_mean,nt_logvar)

    return st_mean,st_logvar,nt_mean,nt_logvar,sketch_code,noise_code,disentangled_text