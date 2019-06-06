from PIL import Image
import os, sys
import glob
import numpy as np
import cv2 as cv

import sys
sys.path.append('../detect_blink')
# user-defined libraries
import detect_blinks
import face_detector

image_list = []
for filename in glob.glob('../sample_pictures/s01/*.JPG'): # load sample images from folder s01
    im=Image.open(filename)
    image_list.append(im)

images = [cv.imread(file) for file in glob.glob('../sample_pictures/s01/*.JPG')]

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
list_blinks = []

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

score_blinks = []
# evaluate eye blinks
for img in images:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_detector.hogg_face_detector(gray)
    val = detect_blinks.detect_blink(img, gray, faces)
    score_blinks.append(val) # need to decide how to evaluate it

# print(score_blinks)

rank_list = max_velocity_vectors_list
rank_list.sort()
optimal_rank = rank_list[1]
optimal_rank_index = max_velocity_vectors_list.index(optimal_rank)

copy_of_max_velocity_vectors_list = max_velocity_vectors_list
copy_of_max_velocity_vectors_list.remove(optimal_rank)

number_of_same_min_velocity_vectors = 0;
for i in max_velocity_vectors_list:
    if i == optimal_rank:
        number_of_same_min_velocity_vectors += 1

for i in range(0, number_of_same_min_velocity_vectors):
    copy_of_max_velocity_vectors_list.remove(optimal_rank)
    copy_of_optimal_rank_index = copy_of_max_velocity_vectors_list.index(optimal_rank)
    optimal_rank_blink_score = score_blinks[optimal_rank_index]
    copy_of_optimal_rank_blink_score = score_blinks[copy_of_optimal_rank_index]
    if copy_of_optimal_rank_blink_score < optimal_rank_blink_score:
        optimal_rank = rank_list[i]
        optimal_rank_index = copy_of_optimal_rank_index

optimal_image_number = optimal_rank_index + 1
# print("Optimal rank", optimal_rank)
print("We recommend image", optimal_image_number)
# print("Optimal rank blink score", score_blinks[optimal_rank_index])
# print(score_blinks)
