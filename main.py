import cv2
import numpy as np

refPt = []
selected = False
clicked = False


def main():
    # get path of image and source as input
    # pathImage = input("Enter path for image: ")
    # pathSource = input("Enter path for source: ")
    pathImage = "source.jpg"
    pathSource = "t.jpg"

    # show and resize image and source
    image = cv2.imread(pathImage)
    source = cv2.imread(pathSource)
    image = resize(image, 800)
    source = resize(source, 400)
    cv2.imshow("Source", source)
    cv2.setWindowProperty("Source", cv2.WND_PROP_TOPMOST, 1)

    # crop target from image and transfer color from target to source
    print("Menu:\n\tr = reset selected area\n\tc = crop selected area")
    target = crop_picture(image)
    transfer = color_transfer(source, target)

    # resize color transferred source to fit the cropped target
    (h, w, _) = target.shape
    transfer = cv2.resize(transfer, (w, h), interpolation=cv2.INTER_AREA)
    image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]] = transfer
    cv2.imshow("Image", image)
    cv2.waitKey(0)


def resize(image, width):
    height = int(image.shape[0] * (width / float(image.shape[1])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def crop_picture(image):
    global selected, clicked
    clone = image.copy()
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", click_and_crop, clone)
    while True:
        # display the image and wait for a keypress
        cv2.imshow("Image", clone)
        cv2.setWindowProperty("Image", cv2.WND_PROP_TOPMOST, 1)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the selected area
        if key == ord("r") or key == ord("R"):
            # indicates that need to select area
            clicked = False
            selected = False
            clone = image.copy()
            cv2.setMouseCallback("Image", click_and_crop, clone)
        # if the 'c' key is pressed, break from the loop and crop selected area
        elif key == ord("c") or key == ord("C"):
            if selected:
                break
            else:
                print("No area selected")
    # return cropped target
    roi = image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    return roi


def click_and_crop(event, x, y, flags, param):
    global refPt, selected, clicked
    if not clicked:
        # get initial (x, y) coordinates by clicking mouse button
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
        # get final (x, y) coordinates by releasing mouse button
        elif event == cv2.EVENT_LBUTTONUP:
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
            # draw a rectangle around the selected area
            cv2.rectangle(param, refPt[0], refPt[1], (0, 255, 0), 2)
            # indicates that area is selected, and can't select another area until reset
            clicked = True
            # indicates that crop can be done after selecting area
            selected = True


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
