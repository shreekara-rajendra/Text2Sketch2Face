import os
import cv2

def generate_sketch(ds_src_path, ds_dst_path):
    if os.path.exists(ds_dst_path):
        os.makedirs(ds_dst_path)
    for filename in os.listdir(ds_src_path):
        img = cv2.imread(os.path.join(ds_src_path,filename))
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inverted_image = cv2.bitwise_not(gray_image)
        blur_image = cv2.GaussianBlur(inverted_image, (21, 21), 0)
        inverted_blur = cv2.bitwise_not(blur_image)
        sketch = cv2.divide(gray_image, inverted_blur, scale=256.0)
        cv2.imwrite(os.path.join(ds_dst_path,filename), sketch)
