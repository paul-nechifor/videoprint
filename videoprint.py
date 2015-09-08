#!/usr/bin/env python2

import cv2
import numpy as np
import sys


def process_video(in_file, out_file, width=1920):
    height = 256
    capture = cv2.VideoCapture(in_file)
    n_frames = int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    frames_per_column = int(n_frames / width)
    image = np.zeros((height, width, 3), np.uint8)

    for i_column in xrange(width):
        column1 = np.zeros((height, 1))
        column2 = np.zeros((height, 1))
        column3 = np.zeros((height, 1))
        for _ in xrange(frames_per_column):
            _, frame = capture.read()
            hist1 = cv2.calcHist([frame], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([frame], [1], None, [256], [0, 256])
            hist3 = cv2.calcHist([frame], [2], None, [256], [0, 256])
            column1 += hist1
            column2 += hist2
            column3 += hist3
        column1 = (column1 / (column1.max() / 255)).astype(np.uint8)
        column2 = (column2 / (column2.max() / 255)).astype(np.uint8)
        column3 = (column3 / (column3.max() / 255)).astype(np.uint8)
        for i, c in enumerate(column1):
            image[i][i_column][0] = c
        for i, c in enumerate(column2):
            image[i][i_column][1] = c
        for i, c in enumerate(column3):
            image[i][i_column][2] = c

    capture.release()

    cv2.imwrite(out_file, image)


if __name__ == '__main__':
    process_video(*sys.argv[1:3])
