import cv2
import numpy as np
import os
import time

# Playing video from file:
cap = cv2.VideoCapture('VID_50ac1f0b07c7ba4b6ae851ce8bb8a51a.mp4')

try:
    if not os.path.exists('images'):
        os.makedirs('images')
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
    #time.sleep(1)
    # Saves image of the current frame in jpg file
        name = './images/frame' + str(currentFrame) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
        currentFrame += 1
    else:
        break
    # To stop duplicate images

   