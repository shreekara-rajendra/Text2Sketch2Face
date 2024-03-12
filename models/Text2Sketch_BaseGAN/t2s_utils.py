import torch
import torchvision.utils as vutils
import math
import os


def generate_imgs(a, ab_gen, samples_path, epoch=0, str_epoch='ok'):
    ab_gen.eval()
    fake = ab_gen(a)
    a_imgs = torch.zeros((fake.shape[0], 3, fake.shape[2], fake.shape[3]))
    a_imgs = fake.cpu()
    rows = math.ceil((fake.shape[0]) ** 0.5)
    a_imgs_ = vutils.make_grid(a_imgs, normalize=True, nrow=rows)
    vutils.save_image(a_imgs_, os.path.join(samples_path, str_epoch + '_a2b_' + str(epoch) + '.png'))
