import torch
import os
import argparse
import numpy as np
import time
from utils.Encoder import Encoder
from utils.Decoder import Decoder
from utils.Critic import Critic
from utils.Generator import Generator
from utils.losses import *
from tqdm import tqdm

 
def mkdir(path):
    if os.path.exists(path) == False :
        os.makedirs(path)

def mkdirs(paths):
    if type(paths) == str:  
        paths = [paths]      
    for path in paths:
        mkdir(path)

def train_cvae (args,dataloader):
    lr = args.lr
    beta1 = args.beta1
    beta2 = args.beta2
    attribute_channels = args.attribute_channels
    sketch_channels = args.sketch_channels
    noise_channels = args.noise_channels
    epochs = args.epoch
    E = Encoder(sketch_channels,attribute_channels,noise_channels)
    D = Decoder(sketch_channels,attribute_channels,noise_channels)
    E_optimizer = torch.optim.Adam(E.parameters(),lr = lr,betas = (beta1,beta2))
    D_optimizer = torch.optim.Adam(D.parameters(),lr = lr,betas = (beta1,beta2))
    loop = tqdm(dataloader)
    lambda1 = 0.0001
    for epoch in range(epochs):
        for batch_idx, (data) in enumerate(loop):
            # data -> sketch + text_att + noise  :: type : tensor :: device : cuda
            real_sketch = data[0]
            text_att = data[1]
            noise = data[2]
            st_mean,st_logvar,nt_mean,nt_logvar,sketch_code,noise_code,disentangled_text = E(real_sketch,text_att,noise)
            real_sketch_recon,fake_sketch = D(sketch_code,noise_code)
            E_optimizer.zero_grad()
            D_optimizer.zero_grad()
            KLloss_st = KLDivergence(st_mean,st_logvar)
            KLloss_nt = KLDivergence(nt_mean,nt_logvar)
            Gloss_st = GausianSimilarity(real_sketch_recon,real_sketch)
            Gloss_nt = GausianSimilarity(fake_sketch,real_sketch)
            loss = (KLloss_nt+KLloss_st) + (lambda1*(Gloss_nt+Gloss_st))
            loss.backward()
            E_optimizer.step()
            D_optimizer.step()

def train_gan (args,dataloader):
    lr = args.lr
    beta1 = args.beta1
    beta2 = args.beta2
    attribute_channels = args.attribute_channels
    sketch_channels = args.sketch_channels
    noise_channels = args.noise_channels
    epochs = args.epoch
    E = Encoder(sketch_channels,attribute_channels,noise_channels)
    D = Decoder(sketch_channels,attribute_channels,noise_channels)
    C = Critic(2,4)
    G = Generator(256)
    C_optimizer = torch.optim.Adam(C.parameters(),lr = lr,betas = (beta1,beta2))
    G_optimizer = torch.optim.Adam(G.parameters(),lr = lr,betas = (beta1,beta2))
    loss_fn_fake = patch_fake()
    loss_fn_real = patch_real()
    loop = tqdm(dataloader)
    lambda_c  = 0.5
    lambda_g_1 = 10
    lambda_g_2 = 50
    perp_loss = perceptual_loss()
    rec_loss = reconstruction_loss()
    for epoch in range(epochs):
        for batch_idx, (data) in enumerate(loop):
             # data -> sketch + text_att + noise  :: type : tensor :: device : cuda
            real_sketch = data[0]
            text_att = data[1]
            noise = data[2]
            st_mean,st_logvar,nt_mean,nt_logvar,sketch_code,noise_code,disentangled_text = E(real_sketch,text_att,noise)
            real_sketch_recon,fake_sketch = D(sketch_code,noise_code)
            real_recon_sketch = G(real_sketch_recon,disentangled_text)
            fake_recon_sketch = G(fake_sketch,disentangled_text)
            C_optimizer.zero_grad()
            ## experimental
            '''
            Critic Loss
            '''
            ## loss for fake
            fake_input = torch.cat([real_sketch_recon[0],real_recon_sketch],1)
            fake_pred = C(fake_input.detach())
            fake_loss = loss_fn_fake(fake_pred)
            ## loss for real
            real_input = torch.cat([real_sketch_recon[0],real_sketch],1)
            real_pred = C(real_input.detach())
            real_loss = loss_fn_real(real_pred)
            loss_c = (real_loss+fake_loss)*lambda_c
            loss_c.backward()
            C_optimizer.step()
            G_optimizer.zero_grad()
            '''
            Generator Loss
            '''
            fake_input = torch.cat([real_sketch_recon[0],real_recon_sketch],1)
            fake_pred = C(fake_input.detach())
            fake_loss = loss_fn_real(fake_pred)
            ploss = perp_loss(real_sketch,real_recon_sketch)
            rloss = rec_loss(real_sketch,real_recon_sketch)
            loss_g = fake_loss + (ploss * lambda_g_1) + (rloss * lambda_g_2)
            G_optimizer.step()
            
            
                        
            
            
            
             
    





def main(arg):
    print("received arguments")
    print(arg)
    









if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text to Sketch Model")
    parser.add_argument('--lr',type = float,default  = 0.0002,help = "Learning rate")
    parser.add_argument('--beta1',type = float,default = 0.5,help = "beta1 of adam optimizer")
    parser.add_argument('--beta2',type = float,default = 0.999,help = "beta2 of adam optimizer")
    parser.add_argument('--attribute_channels',type = int,default = 20,help = "dimension of text attributes")
    parser.add_argument('--epoch',type = int,default = 100,help = "epochs")
    parser.add_argument('--noise_channels',type = int,default = 1024,help = "dim of noise")
    parser.add_argument('--sketch_channels',type = int , default = 1,help = "number of sketch channels")
    args = parser.parse_args()
    main(args)