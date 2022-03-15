import cv2
import numpy as np

refPt = []
cropping = False


def main():
    pathImage = "photos/forest.jpg"
    pathTarget = "photos/t.jpg"
    image = cv2.imread(pathImage)
    source = cv2.imread(pathTarget)
    image = resize(image, 800)
    source = resize(source, 400)
    cv2.imshow("Source", source)
    clone = image.copy()
    target = crop_picture(clone, image)
    transfer = color_transfer(source, target)
    (h, w, _) = target.shape
    transfer = cv2.resize(transfer, (w, h), interpolation=cv2.INTER_AREA)
    image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]] = transfer
    cv2.imshow("Image", image)
    cv2.waitKey(0)


def resize(image, width):
    height = int(image.shape[0] * (width / float(image.shape[1])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def crop_picture(clone, source):
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", click_and_crop, clone)
    while True:
        # display the image and wait for a keypress
        cv2.imshow("Image", clone)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            clone = source.copy()
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
    roi = source[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    return roi


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))

        # FIX REFPT!!XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # FIX REFPT!!XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # FIX REFPT!!XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

        cropping = False
        # draw a rectangle around the region of interest
        cv2.rectangle(param, refPt[0], refPt[1], (0, 255, 0), 2)
        # cv2.imshow("image", param)


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
