import glob
import cv2 as cv
# user-defined libraries
import detect_blinks
import face_detector

images = [cv.imread(file) for file in glob.glob("../data.*")]
score_blinks = []
score_optflow = []

#num_people

# evaluate eye blinks
for img in range(0, len(images)):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_detector.hogg_face_detector(gray)
    score_blinks.append(detect_blinks.detect_blink(gray, faces)) # need to decide how to evaluate it 