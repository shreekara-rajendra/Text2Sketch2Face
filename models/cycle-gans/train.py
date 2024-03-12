from model import Discriminator, Generator
from loader import fetch_loader
from utils import generate_imgs
from torch import optim
import torch
import os

EPOCHS = 1000
BATCH_SIZE = 256
LOAD_MODEL = False

IMAGE_SIZE = 64
A_Channels = 1
B_Channels = 3

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
ab_gen = Generator(in_channels=A_Channels, out_channels=B_Channels)
# ba_gen = Generator(in_channels=B_Channels, out_channels=A_Channels)
# a_disc = Discriminator(channels=A_Channels)
# b_disc = Discriminator(channels=B_Channels)

# Load previous model   
if LOAD_MODEL:
    ab_gen.load_state_dict(torch.load(os.path.join(model_path, 'ab_gen.pkl')))
    # ba_gen.load_state_dict(torch.load(os.path.join(model_path, 'ba_gen.pkl')))
    # a_disc.load_state_dict(torch.load(os.path.join(model_path, 'a_disc.pkl')))
    # b_disc.load_state_dict(torch.load(os.path.join(model_path, 'b_disc.pkl')))

# Define Optimizers
g_opt = optim.Adam(list(ab_gen.parameters()) , lr=0.0002, betas=(0.5, 0.999),
                   weight_decay=2e-5)
# d_opt = optim.Adam(list(a_disc.parameters()) + list(b_disc.parameters()), lr=0.00002, betas=(0.5, 0.999),
#                    weight_decay=2e-5)

# Data loaders
a_loader = fetch_loader('/content/img_align_celeba/real',BATCH_SIZE,IMAGE_SIZE,'/content/image_celeb_sketch/sketch')
iters_per_epoch = len(a_loader)

# Fix images for viz
a_fixed = next(iter(a_loader))

# GPU Compatibility
is_cuda = torch.cuda.is_available()
if is_cuda:
    ab_gen = ab_gen.cuda()#, ba_gen.cuda()
    # a_disc, b_disc = a_disc.cuda(), b_disc.cuda()

    a_fixed = a_fixed.cuda()

# Cycle-GAN Training
for epoch in range(EPOCHS):
    ab_gen.train()
    # ba_gen.train()
    # a_disc.train()
    # b_disc.train()

    for i, (a_real, b_real) in a_loader:

        if is_cuda:
            a_real, b_real = a_real.cuda(), b_real.cuda()

        # Fake Images
        b_fake = ab_gen(a_real)
        # a_fake = ba_gen(b_real)

        # Training discriminator
        # a_real_out = a_disc(a_real)
        # a_fake_out = a_disc(a_fake.detach())
        # a_d_loss = (torch.mean((a_real_out - 1) ** 2) + torch.mean(a_fake_out ** 2)) / 2

        # b_real_out = b_disc(b_real)
        # b_fake_out = b_disc(b_fake.detach())
        # b_d_loss = (torch.mean((b_real_out - 1) ** 2) + torch.mean(b_fake_out ** 2)) / 2

        # d_opt.zero_grad()
        # d_loss = a_d_loss + b_d_loss
        # d_loss.backward()
        # d_opt.step()

        # # Training Generator
        # a_fake_out = a_disc(a_fake)
        # b_fake_out = b_disc(b_fake)

        # a_g_loss = torch.mean((a_fake_out - 1) ** 2)
        # b_g_loss = torch.mean((b_fake_out - 1) ** 2)
        # g_gan_loss = a_g_loss + b_g_loss

        g_loss = (b_real - b_fake).abs().mean()
        # b_g_ctnt_loss = (b_real - ab_gen(a_fake)).abs().mean()
        # g_ctnt_loss = a_g_ctnt_loss + b_g_ctnt_loss

        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        if i % 25 == 0:
            print("Epoch: " + str(epoch + 1) + "/" + str(EPOCHS)
                  + " it: " + str(i) + "/" + str(iters_per_epoch)
                  # + "\ta_d_loss:" + str(round(a_d_loss.item(), 4))
                  # + "\ta_g_loss:" + str(round(a_g_loss.item(), 4))
                  # + "\ta_g_ctnt_loss:" + str(round(a_g_ctnt_loss.item(), 4))
                  # + "\tb_d_loss:" + str(round(b_d_loss.item(), 4))
                  # + "\tb_g_loss:" + str(round(b_g_loss.item(), 4))
                  + "\tb_g_ctnt_loss:" + str(round(g_loss.item(), 4)))
            generate_imgs(a_fixed, a_fixed, ab_gen, ab_gen, samples_path, epoch=i, str_epoch=str(epoch+1))
    torch.save(ab_gen.state_dict(), os.path.join(model_path, 'ab_gen.pkl'))
    # torch.save(ba_gen.state_dict(), os.path.join(model_path, 'ba_gen.pkl'))
    # torch.save(a_disc.state_dict(), os.path.join(model_path, 'a_disc.pkl'))
    # torch.save(b_disc.state_dict(), os.path.join(model_path, 'b_disc.pkl'))

    
generate_imgs(a_fixed, a_fixed, ab_gen, ab_gen, samples_path)
