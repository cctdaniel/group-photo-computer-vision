from PIL import Image
import os, sys
import glob
import numpy as np
import cv2 
import time
import matplotlib.pyplot as plt

image_list_array = []
# load sample images from folder s01
for filename in glob.glob('../sample_pictures/s01/*.jpg'): 
    im=Image.open(filename)
    image_list_array.append(np.array(im))

# optical flows will be stored here
list_optical_flows = []
# add zero optical flow for the reference image which is at first position
shape_image_reference = image_list_array[0].shape
shape_optical_flow = [shape_image_reference[0], shape_image_reference[1]]
t1 = time.time()

prevImg = image_list_array[1]
prevImg = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
img = image_list_array[0]
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
optical_flow = cv2.calcOpticalFlowFarneback(prevImg, img, None, 0.5, 3, 15, 2, 5, 1.2, 0)
mag, _ = cv2.cartToPolar(optical_flow[...,0], optical_flow[...,1])
list_optical_flows.append(mag)

# calculate optical flow for the rest of the images
prevImg = image_list_array[0]
prevImg = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
t2 = time.time() - t1
for index in range(1, len(image_list_array)):
    img = image_list_array[index]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    optical_flow = cv2.calcOpticalFlowFarneback(prevImg, img, None, 0.5, 3, 15, 2, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(optical_flow[...,0], optical_flow[...,1])
    prevImg = img
    list_optical_flows.append(mag)
t3 = time.time() - (t1+t2)
print(t2)
print(t3)

max_velocity_vectors_list = []
for i in range(len(list_optical_flows)):
    max_velocity_vectors_list.append(np.amax(list_optical_flows[i]))
print(max_velocity_vectors_list)
print(np.argsort(max_velocity_vectors_list))