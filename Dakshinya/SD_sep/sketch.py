import cv2 as cv
import numpy
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import os
from pathlib import Path
p = Path('pics')
os.mkdir(p)
os.chdir(p)
os.mkdir('train')
os.mkdir('test')
os.chdir('/content/pytorch-CycleGAN-and-pix2pix')

for i in range(30000):
  images_path = "/content/imgs/celeba_hq_256"
  if i<10:
    each=Path('0000'+str(i)+'.jpg')
  elif i<100:
    each=Path('000'+str(i)+'.jpg')
  elif i<1000:
    each=Path('00'+str(i)+'.jpg')
  elif i<10000:
    each=Path('0'+str(i)+'.jpg')
  new = os.path.join(images_path,each)

  if i<1000:
   new2 = os.path.join("/content/pytorch-CycleGAN-and-pix2pix/pics/train",each)
  else:
   new2 = os.path.join("/content/pytorch-CycleGAN-and-pix2pix/pics/test",each)
  image = cv.imread(new,1)
  im1 = cv.cvtColor(cv.cvtColor(image,cv.COLOR_BGR2RGB),cv.COLOR_RGB2GRAY)

  im2 = cv.GaussianBlur(cv.bitwise_not(im1),(21, 21),sigmaX=0, sigmaY=0)
  im3 = cv.divide(im1,255-im2,scale=255)
  im_3_2 = cv.cvtColor(im3,cv.COLOR_GRAY2BGR)
  im4 = cv.hconcat([image[...,::-1],im_3_2])
  plt.axis('off')
  plt.imshow(im4,cmap="gray")
  plt.savefig(new2)
