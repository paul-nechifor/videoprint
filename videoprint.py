#!/usr/bin/env python2

from colorsys import hls_to_rgb
import cv2
import numpy as np
import sys


def process_video(in_file, out_file, width=1000):
    height = 256
    capture = cv2.VideoCapture(in_file)
    n_frames = int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    frames_per_column = int(n_frames / width)
    image = np.zeros((height, width, 3), np.uint8)

    for i_column in xrange(width):
        column = np.zeros((height, 1))
        for _ in xrange(frames_per_column):
            _, frame = capture.read()
            hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
            column += hist
        column = (column / (column.max() / 255)).astype(np.uint8)
        for i, c in enumerate(column):
            image[i][i_column] = c

    capture.release()

    cv2.imwrite(out_file, image)


if __name__ == '__main__':
    process_video(*sys.argv[1:3])
