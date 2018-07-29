import os

import cv2
import pytesseract
from PIL import Image


def process_image(imagename):
    image = cv2.imread(imagename)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    smooth = cv2.fastNlMeansDenoising(gray, None, 12, 7, 21)

    # th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    threshed = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 14)

    points = cv2.findNonZero(threshed)
    rect = cv2.minAreaRect(points)

    (cx, cy), (w, h), ang = rect
    if w > h:
        w, h = h, w
        ang += 90

    matrix = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    rotated = cv2.warpAffine(threshed, matrix, (image.shape[1], image.shape[0]))

    filename = "{}-ocr.png".format(imagename)

    cv2.imwrite(filename, rotated)

    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    print('\n' + imagename + '\n----\n\n' + text)

process_image('report1.jpg')
process_image('report2.jpg')
process_image('report3.jpg')
# process_image('receipt.jpg')
