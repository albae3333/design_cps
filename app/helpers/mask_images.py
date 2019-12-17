import pandas as pd
import cv2.cv2 as cv2
import imutils
import numpy as np
from os.path import isfile, join
from os import listdir
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

polygon = []
clicked = False
imgpredsize = 128


def mouse_drawing(event, x, y, flags, params):
    global polygon
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left click:({},{})".format(x, y))
        polygon.append((x, y))
        clicked = True


def normalize(data):
    data = np.array(data, dtype=np.float32)
    # data -= data.mean()
    data -= 9.5
    # data /= data.std()
    data /= 20.0
    return data


dirlist = os.listdir("./input/with_pipe")
dirlist.sort()
fromto = (0,len(dirlist))
for i in range(fromto[0], fromto[1]):
    print(dirlist[i])
    img = join("./input/with_pipe", dirlist[i])

    file = cv2.imread(img, 1)
    # resize picture -> make same width and height
    oldshape = file.shape
    imgsize = min(file.shape[0], file.shape[1])
    diff = (oldshape[0]-imgsize, oldshape[1]-imgsize)
    # crop new cubic shaped image
    file = file[int(diff[0]/2):diff[0]+imgsize, int(diff[1]/2):diff[1]+imgsize, :]
    img = file.copy()
    original = img.copy()
    polygon.clear()
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", mouse_drawing)

    while True:
        cv2.imshow("Frame", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break
        if key & 0xFF == ord("c") and len(polygon) > 0:
            cnt = np.array(polygon)
            mask = np.zeros(original.shape, dtype=np.uint8)
            cv2.fillPoly(mask, pts=[cnt], color=(255,255,255))

            # apply the mask
            masked_image = cv2.bitwise_and(original, mask)

            # resizing
            original = cv2.resize(original, (128, 128),
                                 interpolation=cv2.INTER_AREA)
            imgnorm = normalize(masked_image)
            imgnorm = cv2.resize(imgnorm, (128, 128),
                                 interpolation=cv2.INTER_AREA)

            cv2.namedWindow("Cut")
            cv2.imshow("Cut", original)
            cv2.namedWindow("CutMask")
            cv2.imshow("CutMask", imgnorm)
            cv2.waitKey(0)

            imgnorm = imgnorm*255
            cv2.imwrite("./app/dataset/cut/"+str(i)+".png", original)
            cv2.imwrite("./app/dataset/maske/"+str(i)+".png", imgnorm)

            polygon.clear()
            cv2.destroyWindow("Cut")
            cv2.destroyWindow("CutMask")

            # break
        if clicked == True:
            cnt = np.array(polygon)
            img = file.copy()
            if len(polygon) > 2:
                cv2.drawContours(img, [cnt], 0, (0, 0, 255), 1)
            for pnt in polygon:
                cv2.circle(img, pnt, 3, (0, 0, 255), -1)
            clicked = False
