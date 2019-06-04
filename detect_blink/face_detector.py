import dlib

def hogg_face_detector(image):
    detector = dlib.get_frontal_face_detector()
    rects = detector(image)
    #print(rects)
    return rects