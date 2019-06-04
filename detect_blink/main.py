import glob
import cv2 as cv

# user-defined libraries
import detect_blinks
import face_detector


image = cv.imread('../data/random.jpg')
images = []
images.append(image)
#images = [cv.imread(file) for file in glob.glob("../data/*.jpg")]
score_blinks = []
score_optflow = []

#num_people

# evaluate eye blinks
for img in images:
    cv.imshow("Image", img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_detector.hogg_face_detector(gray)
    print(len(faces))


    
    score_blinks.append(detect_blinks.detect_blink(img, gray, faces)) # need to decide how to evaluate it 