from PIL import Image
import os, sys
import glob
import numpy as np
import cv2 as cv

image_list = []
for filename in glob.glob('../sample_pictures/s01_p7/*.jpg'): # load sample images from folder s01_p7
    im=Image.open(filename)
    image_list.append(im)

# # resize image
# for infile in image_list:
#     outfile = os.path.splitext(infile.filename)[0] + ".thumbnail"
#     if infile != outfile:
#         try:
#             im = infile
#             im = im.resize((400, 400))
#             im.save(outfile, "JPEG")
#         except IOError:
#             print "cannot create thumbnail for '%s'" % infile

# optical flows will be stored here
list_optical_flows = []

image_list_array = []
for infile in image_list:
    pix = np.array(infile)
    image_list_array.append(pix)

# add zero optical flow for the reference image which is at first position
shape_image_reference = image_list_array[0].shape
shape_optical_flow = [shape_image_reference[0],
                      shape_image_reference[1],
                      2]
zero_optical_flow = np.zeros(shape_optical_flow, np.float32)
list_optical_flows.append(zero_optical_flow)

# calculate optical flow for the rest of the images
prevImg = image_list_array[0]
prevImg = cv.cvtColor(prevImg, cv.COLOR_BGR2GRAY)
for index in range(1, len(image_list_array)):
    img = image_list_array[index]
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    optical_flow = cv.calcOpticalFlowFarneback(prevImg, img, None, 0.5, 3, 15, 2, 5, 1.2, 0)
    prevImg = img
    list_optical_flows.append(optical_flow)

# print(list_optical_flows[0])

number = 0
max = 0
max_velocity_vectors_list = []
for i in list_optical_flows:
    for j in i:
        for k in j:
            for l in k:
                number += abs(l)
            if number > max:
                max = number
            number = 0
    max_velocity_vectors_list.append(max)
    max = 0

rank_list = max_velocity_vectors_list
rank_list.sort()
optimal_rank = rank_list[1]
optimal_rank_index = max_velocity_vectors_list.index(optimal_rank)
print(optimal_rank)
print(optimal_rank_index)
