import cv2
import numpy as np
import copy
import math
import sqlite3
from collections import deque
# import predict_2
# import predict_1
import predict_interface_usage as piu
from tkinter import *

cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
threshold = 24  # BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50

far = (0, 0)
end = (0, 0)
start = (0, 0)
learningRate = 0

isBgCaptured = 0  # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works

num = 0

conn = sqlite3.connect('test.db')
con = sqlite3.connect('database.db')
curr = conn.cursor()
cur = con.cursor()
curr.execute("""CREATE TABLE IF NOT EXISTS PREV(VAL INTEGER NOT NULL)""")
cur.execute("""CREATE TABLE IF NOT EXISTS NUMBER(VAL INTEGER NOT NULL)""")

if curr and cur:
    print("DBs Created Successfully")

curr.execute("SELECT * FROM PREV")
cursor = curr.fetchall()
for row in cursor:
    if row[0] >= 0:
        num = row[0]
    else:
        num = 0


def remove_bg(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def calculate_fingers(res, drawing):  # -> finished bool, cnt: finger count
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)
            cnt = 0
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
                    cv2.circle(drawing, end, 8, [211, 84, 100], -1)
            return True, cnt
    return False, 0


camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])
kernel = np.ones((5, 5), np.uint8)

bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]

bindex = 0
gindex = 0
rindex = 0
yindex = 0

colors = [(0, 0, 0), ]
colorIndex = 0
paintWindow = np.zeros((400, 400, 3)) + 255
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)
count = 0
lis = []

while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    if isBgCaptured == 1:  # this part wont run until background captured

        img = remove_bg(frame)

        img = img[0:int(cap_region_y_end * frame.shape[0]),
              int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        thresh1 = copy.deepcopy(thresh)
        _, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        dr = np.zeros(img.shape, np.uint8)
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
            dr = drawing
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)  # Outline of hands
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)  # Dots at the depression point
            isFinishCal, cnt = calculate_fingers(res, drawing)
            if triggerSwitch is True:
                if isFinishCal is True and cnt <= 5:
                    print(cnt + 1)

        cntr_output_window_name = "contour output"
        cv2.namedWindow(cntr_output_window_name)  # Create a named window
        cv2.moveWindow(cntr_output_window_name, 20, 40)
        cv2.imshow(cntr_output_window_name, dr)

        _, count = calculate_fingers(res, dr)
        if 1 < count + 1 < 3:  # Checking if the no. of fingers is 2 if not then dont draw
            if len(contours) > 0:

                # Sort the contours and find the largest one -- we
                # will assume this contour corresponds to the area of the bottle cap
                cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
                # Get the radius of the enclosing circle around the found contour
                ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                # Draw the circle around the contour
                cv2.circle(frame, (int(end[0]), int(end[1])), int(radius), (0, 255, 255), 2)
                # Get the moments to calculate the center of the contour (in this case Circle)
                M = cv2.moments(cnt)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

                if center[1] <= 65:
                    if 40 <= center[0] <= 140:  # Clear All
                        bpoints = [deque(maxlen=512)]
                        gpoints = [deque(maxlen=512)]
                        rpoints = [deque(maxlen=512)]
                        ypoints = [deque(maxlen=512)]

                        bindex = 0
                        gindex = 0
                        rindex = 0
                        yindex = 0

                        paintWindow[67:, :, :] = 255
                    elif 160 <= center[0] <= 255:
                        colorIndex = 0  # Blue
                    elif 275 <= center[0] <= 370:
                        colorIndex = 1  # Green
                    elif 390 <= center[0] <= 485:
                        colorIndex = 2  # Red
                    elif 505 <= center[0] <= 600:
                        colorIndex = 3  # Yellow
                else:
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
                        cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 10)
                        cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 10)

        # Show the frame and the paintWindow image
        tracking_window_name = "Tracking"
        cv2.namedWindow(tracking_window_name)  # Create a named window
        cv2.moveWindow(tracking_window_name, 660, 40)  # Move it to (40,30)
        cv2.imshow(tracking_window_name, frame)

        paint_window_name = "Paint"
        cv2.namedWindow(paint_window_name)  # Create a named window
        cv2.moveWindow(paint_window_name, 420, 40)  # Move it to (40,30)
        cv2.imshow(paint_window_name, paintWindow)

    # Keyboard OP
    k = cv2.waitKey(10)

    if k == 27:  # press ESC to exit
        conn.commit()
        conn.close()
        cursor = cur.execute("SELECT * FROM NUMBER")
        for row in cursor:
            lis.append(row[0])
        cur.execute("DROP TABLE NUMBER")
        su = 0

        for i in lis:
            su = su + i

        if su == 0:
            con.commit()
            con.close()
            lis.clear()
            exit(0)
        a = Tk()
        a.geometry("500x200")
        a.title("Predicted Value")
        f = Frame(a)
        f.grid()
        lbl = Label(f, font=("Helvetika", 50))
        lbl.grid(row=0, column=0)
        lbl['text'] = "+".join([str(x) for x in lis])
        lbl['text'] += "\nSum: " + str(su)
        a.after(4000, lambda: a.destroy())  # Destroy the widget after 30 seconds
        a.mainloop()

        con.commit()
        con.close()
        lis.clear()
        break

    if k == ord('c'):  # press 'c' to Clear Screen

        bpoints = [deque(maxlen=512)]
        gpoints = [deque(maxlen=512)]
        rpoints = [deque(maxlen=512)]
        ypoints = [deque(maxlen=512)]

        bindex = 0
        gindex = 0
        rindex = 0
        yindex = 0
        paintWindow[67:, :, :] = 255
        print("SCREEN CLEARED")

    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print('Background Captured')

    elif k == ord('r'):  # press 'r' to reset the background
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0

    elif k == ord('n'):
        triggerSwitch = True

    elif k == ord('s'):  # press 's' to Save and clear screen
        st = "img/img{}.png".format(num)
        num = num + 1
        curr.execute("UPDATE PREV SET VAL = {}".format(num))
        conn.commit()
        crop_img = paintWindow[180:380, 20:220]
        cv2.imwrite(st, crop_img)
        val = piu.train_and_obtain_number(st[4:-4])  # To get the name without the .png extension

        if val != -1:
            cur.execute("INSERT INTO NUMBER VALUES({})".format(val))
            con.commit()

        a = Tk()
        a.geometry("130x170")
        a.title("Predicted Value")
        f = Frame(a)
        f.grid()
        lbl = Label(f, font=("Helvetika", 100))
        lbl.grid(row=0, column=0)
        lbl['text'] = val
        a.after(2000, lambda: a.destroy())  # Destroy the widget after 30 seconds
        a.mainloop()

        bpoints = [deque(maxlen=512)]
        gpoints = [deque(maxlen=512)]
        rpoints = [deque(maxlen=512)]
        ypoints = [deque(maxlen=512)]

        bindex = 0
        gindex = 0
        rindex = 0
        yindex = 0

        paintWindow[67:, :, :] = 255
        print("SCREEN SAVED")
