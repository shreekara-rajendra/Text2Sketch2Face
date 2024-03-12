from t2s_model import Discriminator, Generator, Discriminator2
from t2s_loader import fetch_loader
from t2s_utils import generate_imgs
from torch import optim
import torch
import os

EPOCHS = 100
BATCH_SIZE = 256
LOAD_MODEL = True

IMAGE_SIZE = 64

Channels = 1

model_path = './model'
if not os.path.exists(model_path):
    os.makedirs(model_path)
samples_path = './samples'
if not os.path.exists(samples_path):
    os.makedirs(samples_path)
db_path = './data'
if not os.path.exists(samples_path):
    os.makedirs(samples_path)

# Networks
ab_gen = Generator(out_channels=Channels)
a_disc = Discriminator(channels=Channels)
b_disc = Discriminator2(channels=Channels)

# Load previous model   
if LOAD_MODEL:
    ab_gen.load_state_dict(torch.load(os.path.join(model_path, 'ab_gen.pkl')))
    a_disc.load_state_dict(torch.load(os.path.join(model_path, 'a_disc.pkl')))
    b_disc.load_state_dict(torch.load(os.path.join(model_path, 'b_disc.pkl')))

# Define Optimizers
g_opt = optim.Adam(list(ab_gen.parameters()) , lr=0.0002, betas=(0.5, 0.999),
                   weight_decay=2e-5)
d_opt = optim.Adam(list(a_disc.parameters()) + list(b_disc.parameters()) , lr=0.0002, betas=(0.5, 0.999),
                   weight_decay=2e-5)

# Data loaders
a_loader = fetch_loader('/content/img_align_celeba/img_align_celeba', '/content/list_attr_celeba.csv',BATCH_SIZE,IMAGE_SIZE,'/content/image_celeb_sketch/sketch')
iters_per_epoch = len(a_loader)

# Fix images for viz
a_fixed = next(iter(a_loader))['att']

# GPU Compatibility
is_cuda = torch.cuda.is_available()
if is_cuda:
    ab_gen= ab_gen.cuda()
    a_disc, b_disc = a_disc.cuda(), b_disc.cuda()

    a_fixed = a_fixed.cuda()

# Cycle-GAN Training
for epoch in range(EPOCHS):
    ab_gen.train()
    a_disc.train()
    b_disc.train()

    i = 0
    for a_real in a_loader:
        att, img = a_real['att'], a_real['img']
        if is_cuda:
            att, img = att.cuda(), img.cuda()

        # Fake Images
        fake = ab_gen(att*(torch.rand(BATCH_SIZE, 40).cuda())+(torch.rand(BATCH_SIZE, 40).cuda())*2)
        
        # Training discriminator
        # att_out = a_disc(img)
        # a_d_loss = torch.mean((att_out - att) ** 2) 

        b_real_out = b_disc(img)
        b_fake_out = b_disc(fake.detach())
        b_d_loss = (torch.mean((b_real_out - 1) ** 2) + torch.mean(b_fake_out ** 2))

        d_opt.zero_grad()
        d_loss = (b_d_loss)
        d_loss.backward()
        d_opt.step()

        # Training Generator
        # a_fake_out = a_disc(fake)
        b_fake_out = b_disc(fake)

        # a_g_loss = torch.mean((a_fake_out - att) ** 2)
        b_g_loss = torch.mean((b_fake_out - 1) ** 2)
        # g_gan_loss = (a_g_loss + b_g_loss)

        img_loss = (img - fake).abs().mean()
        g_opt.zero_grad()
        g_loss = b_g_loss + img_loss
        g_loss.backward()
        g_opt.step()
        i+=1
        if i % 15 == 0:
            print("Epoch: " + str(epoch + 1) + "/" + str(EPOCHS)
                  + " it: " + str(i) + "/" + str(iters_per_epoch)
                  + "\timg_loss:" + str(round(img_loss.item(), 4))
                  + "\td_loss:" + str(round(b_d_loss.item(), 4))
                  + "\tg_loss:" + str(round(b_g_loss.item(), 4)))
            generate_imgs(a_fixed, ab_gen, samples_path, epoch=i, str_epoch=str(epoch+1))
    torch.save(ab_gen.state_dict(), os.path.join(model_path, 'ab_gen.pkl'))
    torch.save(a_disc.state_dict(), os.path.join(model_path, 'a_disc.pkl'))
    torch.save(b_disc.state_dict(), os.path.join(model_path, 'b_disc.pkl'))

    

generate_imgs(a_fixed, ab_gen, samples_path)
