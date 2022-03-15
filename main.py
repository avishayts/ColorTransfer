import sys

import cv2
import numpy as np
import string
from PIL import Image

refPt = []
selected = False
clicked = False

def main():
    # pathImage = input("Enter path for image: ")
    # pathTarget = input("Enter path for target: ")
    # cv2.waitKey()
    # image = Image.open(pathImage)
    # source = Image.open(pathTarget)
    # source.show()
    pathImage = "workers.jpg"
    pathTarget = "forest.jpg"
    image = cv2.imread(pathImage)
    source = cv2.imread(pathTarget)
    image = resize(image, 800)
    source = resize(source, 400)
    cv2.imshow("Source", source)
    # cv2.waitKey()
    clone = image.copy()
    target = crop_picture(clone, image)
    transfer = color_transfer(source, target)
    (h, w, _) = target.shape
    transfer = cv2.resize(transfer, (w, h), interpolation=cv2.INTER_AREA)
    image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]] = transfer
    cv2.imshow("Image", image)
    cv2.waitKey(0)


def keyboard_input():
    text = ""
    letters = string.ascii_lowercase + string.digits
    while True:
        key = cv2.waitKey(1)
        for letter in letters:
            if key == ord(letter):
                text = text + letter
        if key == ord("\n") or key == ord("\r"):  # Enter Key
            break
    return text


def resize(image, width):
    height = int(image.shape[0] * (width / float(image.shape[1])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def crop_picture(clone, source):
    global selected, clicked
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", click_and_crop, clone)
    while True:
        # display the image and wait for a keypress
        cv2.imshow("Image", clone)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            clicked = False
            selected = False
            clone = source.copy()
            cv2.setMouseCallback("Image", click_and_crop, clone)
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            if selected:
                break
            else:
                print("Need to select area")
    roi = source[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    return roi


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, selected, clicked
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if not clicked:
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            # cropping = True
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            refPt.append((x, y))
            # adjust refPt coordinates if needed
            if refPt[0][0] > refPt[1][0]:
                if refPt[0][1] > refPt[1][1]:  # from right to left and down to up
                    temp = refPt[0]
                    refPt[0] = refPt[1]
                    refPt[1] = temp
                else:  # from right to left and up to down
                    x_0 = refPt[0][0]
                    x_1 = refPt[1][0]
                    refPt[0] = [x_1, refPt[0][1]]
                    refPt[1] = [x_0, refPt[1][1]]
            else:
                if refPt[0][1] > refPt[1][1]:  # from left to right and down to up
                    y_0 = refPt[0][1]
                    y_1 = refPt[1][1]
                    refPt[0] = [refPt[0][0], y_1]
                    refPt[1] = [refPt[1][0], y_0]
            # draw a rectangle around the region of interest
            selected = True
            cv2.rectangle(param, refPt[0], refPt[1], (0, 255, 0), 2)
            clicked = True


def image_stats(image):
    # split stats and get mean and standard of each stat
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return image stats
    return lMean, lStd, aMean, aStd, bMean, bStd


def color_transfer(source, target):
    # convert the images from the RGB to LAB color space
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # get image stats from the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    # subtract the mean from the data points
    (l, a, b) = cv2.split(source)
    l -= lMeanSrc
    a -= aMeanSrc
    b -= bMeanSrc

    # scale the data points comprising the synthetic image by factors determined by the respective standard deviations
    l = (lStdTar / lStdSrc) * l
    a = (aStdTar / aStdSrc) * a
    b = (bStdTar / bStdSrc) * b

    # add the averages computed for the photograph
    l += lMeanTar
    a += aMeanTar
    b += bMeanTar

    # fix the pixels range to [0, 255] if they fall outside this range
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # merge the stats and convert back to the RGB color space
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    # return the color transferred image
    return transfer


if __name__ == '__main__':
    main()
