import cv2

cv2.namedWindow('original',cv2.WINDOW_NORMAL)  # Create a named window for the image to be shown
cv2.namedWindow('transformed',cv2.WINDOW_NORMAL)

img=cv2.imread('monalisa.jpg')

gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
inverted_gray=255-gray_img
blur_inverted_gray=cv2.GaussianBlur(inverted_gray,(121,121),0)

inverted_blur=255-blur_inverted_gray

sketch=cv2.divide(gray_img,inverted_blur,scale=256.0)  


cv2.imshow('original', img)  # Display the original image in the window named 'original'
cv2.imshow( 'transformed', sketch )  # Display the transformed image in the window named 'transformed'

cv2.waitkey(0) 

