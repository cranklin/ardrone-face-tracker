#!/usr/bin/python
import sys
import time
import math
import datetime
import serial
import cv

# Parameters for haar detection
# From the API:
# The default parameters (scale_factor=2, min_neighbors=3, flags=0) are tuned
# for accurate yet slow object detection. For a faster operation on real video
# images the settings are:
# scale_factor=1.2, min_neighbors=2, flags=CV_HAAR_DO_CANNY_PRUNING,
# min_size=<minimum possible face size

min_size = (25,25)
image_scale = 2
haar_scale = 1.2
min_neighbors = 4
haar_flags = 0

# For OpenCV image display
WINDOW_NAME = 'FaceTracker'

def track(img, threshold=100):
    '''Accepts BGR image and optional object threshold between 0 and 255 (default = 100).
       Returns: (x,y) coordinates of centroid if found
                (-1,-1) if no centroid was found
                None if user hit ESC
    '''
    cascade = cv.Load("haarcascade_frontalface_alt_tree.xml")
    #cascade = cv.Load("haarcascade_frontalface_default.xml")
    gray = cv.CreateImage((img.width,img.height), 8, 1)
    small_img = cv.CreateImage((cv.Round(img.width / image_scale),cv.Round (img.height / image_scale)), 8, 1)

    # convert color input image to grayscale
    cv.CvtColor(img, gray, cv.CV_BGR2GRAY)

    # scale input image for faster processing
    cv.Resize(gray, small_img, cv.CV_INTER_LINEAR)

    cv.EqualizeHist(small_img, small_img)

    center = (-1,-1)
    if(cascade):
        t = cv.GetTickCount()
        # HaarDetectObjects takes 0.02s
        faces = cv.HaarDetectObjects(small_img, cascade, cv.CreateMemStorage(0), haar_scale, min_neighbors, haar_flags, min_size)
        t = cv.GetTickCount() - t
        if faces:
            faces.sort()
            ((x, y, w, h), n) = faces[-1]
            # the input to cv.HaarDetectObjects was resized, so scale the
            # bounding box of each face and convert it to two CvPoints
            pt1 = (int(x * image_scale), int(y * image_scale))
            pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
            cv.Rectangle(img, pt1, pt2, cv.RGB(255, 0, 0), 3, 8, 0)
            #cv.Rectangle(img, (x,y), (x+w,y+h), 255)
            # get the xy corner co-ords, calc the center location
            x1 = pt1[0]
            x2 = pt2[0]
            y1 = pt1[1]
            y2 = pt2[1]
            centerx = x1+((x2-x1)/2)
            centery = y1+((y2-y1)/2)
            center = (centerx, centery)
        else:
            center = None
    else:
        center = None

    cv.NamedWindow(WINDOW_NAME, 1)
    cv.ShowImage(WINDOW_NAME, img)

    if cv.WaitKey(5) == 27:
        center = None
    return center

if __name__ == '__main__':

    capture = cv.CaptureFromCAM(0)

    while True:

        if not track(cv.QueryFrame(capture)):
            break

