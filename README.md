# How to Get the Best Group Photo

## 1. Overview
When we take group photos, we often see someone who blinked his/her eyes, someone who is blurred because they move at that moment or someone who didn't do the promised motion (e.g. making a V sign). These types of people spoil a nice group photo. However, it is bothersome to find the best group photo by checking such people manually among the series of photos.
The purpose of our project is to find the best group photo among these series of photos. We will first detect the faces in the group photo, and then exclude the photos which has blinking or blurred image. Finally, we will check that all people in the photo has presented the same pose which was previously promised.

## 2. Criteria for Best Group Photo and Input Assumption

- Everyone in the group photo are stationary.
- Everyone in the group photo are non-blinking.
- Everyone in the group photo take the same motion.

**Assume that enough input images are given and the images are sequential group photos.**

## 3. Steps for our project

### 3.1 Face Detection
Using real time face detectors will make eye blink detection process easier as there are multiple people in the input picture. There are many face detection algorithms.  will be referenced. This source uses Haar Feature-based Cascade Classifiers to detect faces in an image and outputs bounding boxes around individual faces.

### 3.2 Optical Flow
People who are ready for photo taking stay still and do not move for a set period until the photo is taken. On the other hand, people who are not ready for photo taking tend to move a lot. (e.g. fixing their hair) By using optical flow method, we will measure each pixel’s velocity. If the maximum velocity is greater than the threshold, it will be considered as an image containing people who are not ready. [OpenCV source](https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af) will be referenced. 


### 3.3. Eye Blink Detection
Eye blink detection is a series of detections. One implementation is eye blink detection by Soukupova [3]. It first detects facial landmarks and detects if the eye is closed or open by SVM linear classifier that is trained on blinking and non-blinking patterns.


### 3.4. Pose Estimation
Pose Estimation is a general problem in computer vision where we detect the position and orientation of an object. It detects keypoint locations that describe the object. The typical expected output of pose estimation is simple skeleton of human body with connected joints. One of the most common models is the Multi-Person Pose Estimation model proposed by Perceptual Computing Lab at Carnegie Mellon University [4]. This model produces 2D locations of keypoints for each person in the image. It exploits two branch multi-stage CNN - one is for predicting confidence maps of body part locations and the other is for measuring degree of association between parts.

Here we will generate skeleton picture for each person in the group picture using Multi-Person Pose Estimation model which is mentioned above. The measurement of similarity between poses are still left as a challenge for us. At first trial, we will assume that the image of model pose is given as an input, so the similarity will be decided based on the model image. When the number of people whose pose is similar enough with model image equals to the whole people detected at face detection 4.2, it is plausible to determine that everyone is taking same pose. If we succeed in doing this, we will also pick one person from the group picture as a standard, rather than receiving additional input image.