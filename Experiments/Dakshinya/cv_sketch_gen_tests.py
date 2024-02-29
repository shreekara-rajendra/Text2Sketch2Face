import cv2 as cv
import matplotlib.pyplot as plt
#Open CV test 1 on a sample image
image = cv.imread('cycle.jpg',1)
im1 = cv.cvtColor(cv.cvtColor(image,cv.COLOR_BGR2RGB),cv.COLOR_BGR2GRAY)
im2 = cv.GaussianBlur(cv.bitwise_not(im1),(21, 21),sigmaX=0, sigmaY=0)
im3 = cv.divide(im1,255-im2,scale=255)
im4 = cv.hconcat(image,im3)
plt.axis('off')
#plt.imshow(im4)

#Open CV test 2 on a sample image
image = cv.imread('cycle.jpg',1)
im0 = cv.cvtColor(image,cv.COLOR_BGR2RGB)
im1 = cv.cvtColor(im0,cv.COLOR_RGB2GRAY)
im2 = cv.GaussianBlur(cv.bitwise_not(im1),(21, 21),sigmaX=0, sigmaY=0)
im3 = cv.divide(im1,255-im2,scale=255)
im_3_2 = cv.cvtColor(im3,cv.COLOR_GRAY2BGR)
im4 = cv.hconcat([image[...,::-1],im_3_2])
plt.axis('off')
plt.imshow(im4,cmap="gray")
