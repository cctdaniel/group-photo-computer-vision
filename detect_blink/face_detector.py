import dlib

def hogg_face_detector(image):

    detector = dlib.get_frontal_face_detector()
    #rects = detector(frameDlibHogSmall, 0)
    rects = detector(image, 1)

    return rects