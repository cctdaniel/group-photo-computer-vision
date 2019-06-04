import cv2
import dlib

# in future, may implement to dnn face detector from cv2
def hogg_face_detector(image):
    hogFaceDetector = dlib.get_frontal_face_detector()
    faceRects = hogFaceDetector(frameDlibHogSmall, 0)
    return faceRects