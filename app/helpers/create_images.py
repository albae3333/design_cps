import cv2 as cv
import numpy as np
import imutils as im
import os
import math

# functions
def resize(image, scale):
    """
    resizes image based on scale input
    @param scale: type = tuple -> resizes absolute by input pixels
    @param scale: type = constant -> resizes relative by input percent
    @return image: type = np.array -> resized image
    """
    if not isinstance(scale, tuple):
        # check if width != height and cut overflow
        # difference between width and height
        fillup = abs(int((image.shape[0] - image.shape[1]) / 2))
        if image.shape[0] > image.shape[1]:
            # apply border (designed by border pixels) to left/right
            image = cv.copyMakeBorder(image,0,0,fillup,fillup,cv.BORDER_REPLICATE)
        elif image.shape[0] < image.shape[1]:
            # apply border (designed by border pixels) to top/bottom
            image = cv.copyMakeBorder(image,fillup,fillup,0,0,cv.BORDER_REPLICATE)
        width = int(image.shape[1] * scale / 100)
        height = int(image.shape[0] * scale / 100)
        dim = (width, height)
    else:
        dim = scale
    # resize operation
    if(dim == (128,128)):
        return image
    try:
        ret = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    except:
        print(dim)
        exit(0)
    return ret

def hypo(a):
    """
    Calculates hypotenusis of square.
    @param a: height or width of square
    @return hypo: returns hypotenusis
    """
    ret = 0
    if a < 0:
        ret = int(math.sqrt(a**2 + a**2)) * -1
    else:
        ret = int(math.sqrt(a**2 + a**2))
    return ret

def pixelMap(a, aMax, b):
    """
    Maps value into other dimensions.
    @param a: current value
    @param aMax: max value of a
    @param b: max value of b
    @return mappedB: mapped b
    """
    return int(a / aMax * b)

def clickCb(event, x, y, flags, param):
    # callback function for click event
    if event == cv.EVENT_LBUTTONUP:
        # increase rect size
        rectOffA[1] += OFFSET
        rectOffA[0] -= OFFSET
        rectOffB[1] -= OFFSET
        rectOffB[0] += OFFSET
    if event == cv.EVENT_RBUTTONUP:
        if rectOffA[0] < -1 and rectOffA[1] > 1 and rectOffB[0] > 1 and rectOffB[1] < -1:
            rectOffA[1] -= OFFSET
            rectOffA[0] += OFFSET
            rectOffB[1] += OFFSET
            rectOffB[0] -= OFFSET
    if event == cv.EVENT_MBUTTONUP:
        global crosshair
        crosshair[0] = x
        crosshair[1] = y
    if event == cv.EVENT_MOUSEMOVE:
        global rectangle
        rectangle[0] = x
        rectangle[1] = y

def rotate_pixels(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    source: https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python/34374437
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos((math.pi*angle)/180) * (px - ox) - math.sin((math.pi*angle)/180) * (py - oy)
    qy = oy + math.sin((math.pi*angle)/180) * (px - ox) + math.cos((math.pi*angle)/180) * (py - oy)
    return (int(qx), int(qy))

def offsetImage(originalImage, offsetX, offsetY):
    """
    https://www.docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html
    """
    rc = originalImage.shape
    M = np.float32([[1,0,offsetX],[0,1,offsetY]])
    dst = cv.warpAffine(originalImage,M,(rc[1],rc[0]))
    return dst

def getFileName(url):
    """
    Slice filename without extension out of an URL.
    """
    return url.split('/')[-1].split('.')[0]

def takeFirst(element):
    return element[0]

# constants
INPUT = "app/dataset"
OUTPUT = "output"
CUTS = []
MASKS = []
for (path, dirnames, filenames) in os.walk(INPUT + "/cut"):
    CUTS.extend(os.path.join(path, name) for name in filenames)
CUTS.sort()
print("Cuts: ")
print(CUTS)
for (path, dirnames, filenames) in os.walk(INPUT + "/maske"):
    MASKS.extend(os.path.join(path, name) for name in filenames)
MASKS.sort()
print("Masken: ")
print(MASKS)
FILES = []
for i in range(len(CUTS)):
    FILES.append((CUTS[i],MASKS[i]))

# settings
SCALEPERCENT = 100 # percent of original size or absolute values as touple (w, h)
ROTATIONSPEED = 10 # in degrees
REPLICATIONPERCENT = 50 # border added to image
OVERWRITE = True
FILETYPE = ".csv"
IMAGETYPE = ".png"
LOG_FILENAMES = False
# visual settings
OFFSET = 3
THICKNESS = 2
COLORRECT = (255, 255, 0)
COLORMARK = (0, 255, 255)
COLORMAX = (0, 0, 255)
WINDOWNAME = "Training Image Creator For Neural Networks"

# assign window type
cv.namedWindow(WINDOWNAME, cv.WINDOW_GUI_NORMAL | cv.WINDOW_AUTOSIZE)
cv.namedWindow(WINDOWNAME + "mask", cv.WINDOW_GUI_NORMAL | cv.WINDOW_AUTOSIZE)

# set callback mouse clicked
if(LOG_FILENAMES):
    cv.setMouseCallback(WINDOWNAME, clickCb)

# globals
rectangle = [0,0]
crosshair = [0,0]
MAPPEDPIXELS = [0,0]

# offset rectangle
rectOffA = [-40,40]
rectOffB = [40,-40]

if LOG_FILENAMES and (OVERWRITE or not os.path.isfile(path + FILETYPE)):
    TXTDEFAULTCONTENT = "image_name;x;y\n"
    txt = open(OUTPUT + "/cuts/log" + FILETYPE, "w")
    txt.write(TXTDEFAULTCONTENT)
    txt.close()
    txt_mask = open(OUTPUT + "/masks/log" + FILETYPE, "w")
    txt_mask.write(TXTDEFAULTCONTENT)
    txt_mask.close()

FILES.sort(key=takeFirst)

for fileIndex in range(0,len(FILES)):
    fle,mask = FILES[fileIndex]
    print("Current file: " + fle + " (" + str(fileIndex+1) + ") from total of " + str(len(FILES)) + " files.")
    # start processing
    imgClean = cv.imread(fle, cv.IMREAD_COLOR) # basic image
    maskClean = cv.imread(mask, cv.IMREAD_GRAYSCALE) # basic mask
    # resize operation
    resized = resize(imgClean, SCALEPERCENT)
    resizedMask = resize(maskClean, SCALEPERCENT)

    # widen picture by replicating border pixels
    pxls = int(resized.shape[0] * REPLICATIONPERCENT / 100)
    resized = cv.copyMakeBorder(resized,pxls,pxls,pxls,pxls,cv.BORDER_REPLICATE)
    resizedMask = cv.copyMakeBorder(resizedMask,pxls,pxls,pxls,pxls,cv.BORDER_CONSTANT, value=(0,0,0))

    # initialize variables
    # picture size
    rectangle[0] = resized.shape[1] / 2
    rectangle[1] = resized.shape[0] / 2
    crosshair[0] = resized.shape[1] / 2
    crosshair[1] = resized.shape[0] / 2

    # save flag (go=1 starts image processing)
    go = 0

    # distance to middle
    distance = [0,0]

    while go == 0 and LOG_FILENAMES == True:
        # copy image from "clean" duplicate
        for i in range(2):
            distance[i] = crosshair[i] - resized.shape[1-i] / 2
            # print(str(i) +" - POS_CROSS" + ": " + str(crosshair[i]) + ";SHAPE: " + str(resized.shape[1-i]))
        if distance != [0,0]:
            resized = offsetImage(resized, -1 * distance[0], -1 * distance[1])
            resizedMask = offsetImage(resizedMask, -1 * distance[0], -1 * distance[1])
            for i in range(2):
                crosshair[i] = resized.shape[1-i] / 2
        img = resized.copy()
        msk = resizedMask.copy()
        # img2 = resized.copy()
        # calc point A rect
        pRectX = (int(rectangle[0] + rectOffA[0]), int(rectangle[1] + rectOffA[1]))
        # calc point B rect
        pRectY = (int(rectangle[0] + rectOffB[0]), int(rectangle[1] + rectOffB[1]))
        # calc point A border
        pBorderX = (int(rectangle[0] + hypo(rectOffA[0])), int(rectangle[1] + hypo(rectOffA[1])))
        # calc point B border
        pBorderY = (int(rectangle[0] + hypo(rectOffB[0])), int(rectangle[1] + hypo(rectOffB[1])))
        # draw rectangle
        cv.rectangle(img, pRectX, pRectY, COLORRECT, THICKNESS)
        #cv.drawMarker(img, pRectX, COLORMARK, cv.MARKER_DIAMOND, 15, THICKNESS)
        #cv.drawMarker(img, pRectY, COLORMARK, cv.MARKER_DIAMOND, 15, THICKNESS)
        # draw border
        cv.rectangle(img, pBorderX, pBorderY, COLORMAX, THICKNESS)
        # draw marker
        cv.drawMarker(img, (int(crosshair[0]), int(crosshair[1])), COLORMARK, cv.MARKER_CROSS, 25, THICKNESS)
        # show image on canvas
        cv.imshow(WINDOWNAME, img)
        cv.imshow(WINDOWNAME + "mask", msk)
        # wait for key press event
        CROPBORDER = abs(rectOffA[0]) + abs(rectOffA[1])
        MAPPEDPIXELS[0] = pixelMap(crosshair[0] - pRectX[0], CROPBORDER, 128)
        MAPPEDPIXELS[1] = pixelMap(crosshair[1] - (resized.shape[0] - pRectX[1]), CROPBORDER, 128)
        k = cv.waitKey(1)
        # print(k)
        # react on key event
        if k == 113: # q = quit
            cv.destroyAllWindows()
            exit(0)
        elif k == 115: # s = save
            go = 1

    # rotation
    angle = 0
    fileIndex = 0

    # main
    while angle < 360:
        # increase angle every 3 iterations
        angle += ROTATIONSPEED
        for x in range(-1,1):
            # flip image in every drection including both directions
            # and save a copy of the current image rotation-flip-combination
            # as 128x128 px formatted png resulting in ~ 4 images/angle
            img = resized.copy()
            msk = resizedMask.copy()
            img = im.rotate(img, angle) #, center=(crosshair[1], crosshair[0]))
            msk = im.rotate(msk, angle)
            img = cv.flip(img, x)
            msk = cv.flip(msk, x)
            if(LOG_FILENAMES):
                cv.imshow(WINDOWNAME, img)
                cv.imshow(WINDOWNAME + "mask", msk)
                cv.waitKey(20)
            path = OUTPUT + "/cuts/" + getFileName(fle) + "_" + str(fileIndex)
            path_mask = OUTPUT + "/masks/" + getFileName(mask) + "_" + str(fileIndex)
            while not OVERWRITE and os.path.isfile(path + IMAGETYPE):
                print(path + "already exists. Skipping it.")
                fileIndex += 1
                path = OUTPUT + "/cuts/" + getFileName(fle) + "_" + str(fileIndex)
                path_mask = OUTPUT + "/masks/" + getFileName(mask) + "_" + str(fileIndex)
            if(LOG_FILENAMES):
                print(path + IMAGETYPE)
            # calc image crop
            y = int(rectangle[1] - rectOffA[1])
            x = int(rectangle[0] - rectOffB[0])
            h = abs(int(rectOffB[1] - rectOffA[1]))
            w = abs(int(rectOffB[0] - rectOffA[0]))
            # show and save crop
            crop_img = img[y:y+h, x:x+w].copy()
            crop_mask = msk[y:y+h, x:x+w].copy()
            # if x < 0:
            #     MAPPEDPIXELS[0] = 128 - MAPPEDPIXELS[0]
            #     MAPPEDPIXELS[1] = 128 - MAPPEDPIXELS[1]
            # elif x > 0:
            #     MAPPEDPIXELS[0] = 128 - MAPPEDPIXELS[0]
            # elif x == 0:
            #     MAPPEDPIXELS[1] = 128 - MAPPEDPIXELS[1]
            if LOG_FILENAMES:
                TXTCONTENT = path + FILETYPE + ";" + str(MAPPEDPIXELS[0]) + ";" + str(MAPPEDPIXELS[1]) + "\n"
                txt = open(OUTPUT + "/cuts/log" + FILETYPE, "a")
                txt.write(TXTCONTENT)
                txt.close()
                TXTCONTENT = path_mask + FILETYPE + ";" + str(MAPPEDPIXELS[0]) + ";" + str(MAPPEDPIXELS[1]) + "\n"
                txt_mask = open(OUTPUT + "/masks/log" + FILETYPE, "a")
                txt_mask.write(TXTCONTENT)
                txt_mask.close()
            cv.imwrite(path + IMAGETYPE, resize(crop_img, (128, 128)))
            cv.imwrite(path_mask + IMAGETYPE, resize(crop_mask, (128, 128)))
            fileIndex += 1
