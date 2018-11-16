import cv2

import numpy as np

import copy

import math

import pyautogui

from collections import deque


from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *
import tkinter as tk

#

# parameters

cap_region_x_begin=0.5  # start point/total width

cap_region_y_end=0.8  # start point/total width

threshold = 35  #  BINARY threshold

blurValue = 41  # GaussianBlur parameter

bgSubThreshold = 50

far = (0, 0)
end = (0, 0)
start = (0, 0)
learningRate = 0
# variables

isBgCaptured = 0   # bool, whether the background captured

triggerSwitch = False  # if true, keyborad simulator works

def printThreshold(thr):

    print("! Changed threshold to "+str(thr))

def removeBG(frame):

    fgmask = bgModel.apply(frame,learningRate=learningRate)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)



    kernel = np.ones((3, 3), np.uint8)



    fgmask = cv2.erode(fgmask, kernel, iterations=1)



    res = cv2.bitwise_and(frame, frame, mask=fgmask)



    return res



def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count


    #  convexity defect

    hull = cv2.convexHull(res, returnPoints=False)



    if len(hull) > 3:

        defects = cv2.convexityDefects(res, hull)

        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0

            pointList = []

            for i in range(defects.shape[0]):  # calculate the angle

                global far
                global end
                global start
                s, e, f, d = defects[i][0]

                start = tuple(res[s][0])

                end = tuple(res[e][0])

                far = tuple(res[f][0])

                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)

                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)

                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem

                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers

                    cnt += 1

                    # cv2.circle(drawing, far, 8, [211, 84, 100], -1) # This is for the circle on the valley between the fingers

                    cv2.circle(drawing, end, 8, [211, 84, 100], -1)

# Click operation
                    # temp = list(far)

                    # pyautogui.click(temp[0], temp[1])
            return True, cnt

    return False, 0

#Camera

camera = cv2.VideoCapture(0)

#camera.set(10, 200)

camera.set(cv2.CAP_PROP_FRAME_WIDTH,600)

camera.set(cv2.CAP_PROP_FRAME_HEIGHT,500)

cv2.namedWindow('trackbar')

cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)

# Define the upper and lower boundaries for a color to be considered "Blue"
blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])

# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)

# Setup deques to store separate colors in separate arrays
bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]

bindex = 0
gindex = 0
rindex = 0
yindex = 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Setup the Paint interface
paintWindow = np.zeros((471,636,3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), colors[0], -1)
paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), colors[1], -1)
paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), colors[2], -1)
paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), colors[3], -1)
cv2.putText(paintWindow, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)



while camera.isOpened():


    ret, frame = camera.read()

    threshold = cv2.getTrackbarPos('trh1', 'trackbar')

    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter

    frame = cv2.flip(frame, 1)  # flip the frame horizontally

    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),

                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

    cv2.imshow('original', frame)

    #  Main operation

    if isBgCaptured == 1:  # this part wont run until background captured

        img = removeBG(frame)

        img = img[0:int(cap_region_y_end * frame.shape[0]),

                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI

        # cv2.imshow('mask', img)

        # convert the image into binary image

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)

        # cv2.imshow('blur', blur)

        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

        # cv2.imshow('ori', thresh)

        # get the coutours

        thresh1 = copy.deepcopy(thresh)

        _, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        length = len(contours)

        maxArea = -1

        if length > 0:

            for i in range(length):  # find the biggest contour (according to area)

                temp = contours[i]

                area = cv2.contourArea(temp)

                if area > maxArea:

                    maxArea = area

                    ci = i

            res = contours[ci]

            hull = cv2.convexHull(res)

            drawing = np.zeros(img.shape, np.uint8)

            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2) # Outline of hands

            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)# Dots at the depression point

            isFinishCal, cnt = calculateFingers(res,drawing)

            if triggerSwitch is True:

                if isFinishCal is True and cnt <= 5:

                    print (cnt+1)

        cv2.imshow('contour output', drawing)
        _, count = calculateFingers(res,drawing)

        if(count+1 >1 and count+1 < 3 ):
            if len(contours) > 0:
            	# Sort the contours and find the largest one -- we
            	# will assume this contour correspondes to the area of the bottle cap
                cnt = sorted(contours, key = cv2.contourArea, reverse = True)[0]
                # Get the radius of the enclosing circle around the found contour
                ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                # Draw the circle around the contour
                cv2.circle(frame, (int(end[0]), int(end[1])), int(radius), (0, 255, 255), 2)
                # Get the moments to calculate the center of the contour (in this case Circle)
                M = cv2.moments(cnt)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

                if center[1] <= 65:
                    if 40 <= center[0] <= 140: # Clear All
                        bpoints = [deque(maxlen=512)]
                        gpoints = [deque(maxlen=512)]
                        rpoints = [deque(maxlen=512)]
                        ypoints = [deque(maxlen=512)]

                        bindex = 0
                        gindex = 0
                        rindex = 0
                        yindex = 0

                        paintWindow[67:,:,:] = 255
                    elif 160 <= center[0] <= 255:
                            colorIndex = 0 # Blue
                    elif 275 <= center[0] <= 370:
                            colorIndex = 1 # Green
                    elif 390 <= center[0] <= 485:
                            colorIndex = 2 # Red
                    elif 505 <= center[0] <= 600:
                            colorIndex = 3 # Yellow
                else :
                    if colorIndex == 0:
                        bpoints[bindex].appendleft(center)
                    elif colorIndex == 1:
                        gpoints[gindex].appendleft(center)
                    elif colorIndex == 2:
                        rpoints[rindex].appendleft(center)
                    elif colorIndex == 3:
                        ypoints[yindex].appendleft(center)
            # Append the next deque when no contours are detected (i.e., bottle cap reversed)
            else:
                bpoints.append(deque(maxlen=512))
                bindex += 1
                gpoints.append(deque(maxlen=512))
                gindex += 1
                rpoints.append(deque(maxlen=512))
                rindex += 1
                ypoints.append(deque(maxlen=512))
                yindex += 1

            # Draw lines of all the colors (Blue, Green, Red and Yellow)
            points = [bpoints, gpoints, rpoints, ypoints]
            for i in range(len(points)):
                for j in range(len(points[i])):
                    for k in range(1, len(points[i][j])):
                        if points[i][j][k - 1] is None or points[i][j][k] is None:
                            continue
                        cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                        cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

        # Show the frame and the paintWindow image
        cv2.imshow("Tracking", frame)
        cv2.imshow("Paint", paintWindow)

    # Keyboard OP
    k = cv2.waitKey(10)

    if k == 27:  # press ESC to exit

        break

    elif k == ord('b'):  # press 'b' to capture the background

        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

        isBgCaptured = 1

        print( '!!!Background Captured!!!')

    elif k == ord('r'):  # press 'r' to reset the background

        bgModel = None

        triggerSwitch = False

        isBgCaptured = 0

        print ('!!!Reset BackGround!!!')

    elif k == ord('n'):

        triggerSwitch = True

print ('!!!Trigger On!!!')
