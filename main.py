import cv2
import numpy as np

refPt = []
cropping = False


def image_stats(image):
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())
    # return the color statistics
    return lMean, lStd, aMean, aStd, bMean, bStd


def _min_max_scale(arr, new_range=(0, 255)):
    # get array's current min and max
    mn = arr.min()
    mx = arr.max()
    # check if scaling needs to be done to be in new_range
    if mn < new_range[0] or mx > new_range[1]:
        # perform min-max scaling
        scaled = (new_range[1] - new_range[0]) * (arr - mn) / (mx - mn) + new_range[0]
    else:
        # return array if already in range
        scaled = arr

    return scaled


def _scale_array(arr, clip=True):
    if clip:
        scaled = np.clip(arr, 0, 255)
    else:
        scale_range = (max([arr.min(), 0]), min([arr.max(), 255]))
        scaled = _min_max_scale(arr, new_range=scale_range)

    return scaled


def show_image(title, image, width=300):
    # resize the image to have a constant width, just to
    # make displaying the images take up less screen real
    # estate
    r = width / float(image.shape[1])
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # show the resized image
    cv2.imshow(title, resized)


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
        cropping = False
        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)


pathSource = "ocean_sunset.jpg"
pathTarget = "ocean_day.jpg"
image = cv2.imread("ocean_sunset.jpg")
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()
    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break
roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
cv2.imshow("r", roi)
cv2.waitKey(0)
    #
    # # if there are two reference points, then crop the region of interest
    # # from the image and display it
    #
    # # convert the images from the RGB to L*ab* color space, being
    # # sure to utilizing the floating point data type (note: OpenCV
    # # expects floats to be 32-bit, so use that instead of 64-bit)
    # source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    # target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
    #
    # # compute color statistics for the source and target images
    # (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    # (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)
    #
    # # subtract the means from the target image
    # (l, a, b) = cv2.split(target)
    # l -= lMeanTar
    # a -= aMeanTar
    # b -= bMeanTar
    #
    # l = (lStdTar / lStdSrc) * l
    # a = (aStdTar / aStdSrc) * a
    # b = (bStdTar / bStdSrc) * b
    #
    # # add in the source mean
    # l += lMeanSrc
    # a += aMeanSrc
    # b += bMeanSrc
    #
    # # clip/scale the pixel intensities to [0, 255] if they fall
    # # outside this range
    # l = _scale_array(l)
    # a = _scale_array(a)
    # b = _scale_array(b)
    #
    # # merge the channels together and convert back to the RGB color
    # # space, being sure to utilize the 8-bit unsigned integer data
    # # type
    # transfer = cv2.merge([l, a, b])
    # transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
    # show_image("Transfer", transfer)
    #
    # cv2.waitKey(0)
    # # return the color transferred image
    # return transfer


# if __name__ == '__main__':
#     main()
