#!/usr/bin/env python
# Copyright 2013 (C) Matthieu Tourne
# @author Matthieu Tourne <matthieu.tourne@gmail.com>

import sys
import cv2
import cv2.cv as cv
import numpy as np

import utils

def get_mask_from_contours(img_gray, contour):
    # "color" is a white mask
    color = 255
    mask = np.zeros((img_gray.shape), np.uint8)
    cv2.drawContours(mask, [contour], 0, color,
                     -1, cv2.CV_AA)
    return mask

def get_water_mask(img_gray, img_rgb):
    #CANNY_THRSH_LOW = 300
    #CANNY_THRSH_HIGH = 600
    CANNY_THRSH_LOW = 100
    CANNY_THRSH_HIGH = 2000
    edge = cv2.Canny(img_gray, CANNY_THRSH_LOW, CANNY_THRSH_HIGH, apertureSize=5)
    kern = np.ones((5, 5))
    # dilatation connects most of the disparate edges
    edge = cv2.dilate(edge, kern)
    #utils.visualise_edges(img_rgb, edge, name="coast_edges")
    # invert edges to create blobs
    edge_inv = np.zeros((img_gray.shape), np.uint8)
    edge_inv.fill(255)
    edge_inv = edge_inv - edge
    #utils.visualise_edges(img_rgb, edge_inv, name="coast_edges_inv")
    contours0, hierarchy0 = cv2.findContours(edge_inv.copy(), cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
    contours = contours0
    # contours of inv edges
    # if contour is >= "relevant area" consider it
    x, y = img_gray.shape
    max_area = x * y
    relevant_area = max_area * 0.05
    potential_masks = []
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if (a > max_area * 0.4):
            # area is over 40% of the image, it's probably water
            return get_mask_from_contours(img_gray, cnt)
        elif (a >= relevant_area):
            # area is over 5% of the image, store them and test them
            potential_masks.append(get_mask_from_contours(img_gray, cnt))
    # create an rg_chroma image for color detection
    rgb_norm = utils.rgb_normalized(img_rgb)
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for mask in potential_masks:
        # TODO (mtourne): implement
        # test potential masks for color (hue, rg_chroma ??)
        utils.get_means_std(img_rgb, rgb_norm=rgb_norm, mask=mask)
        pass
    # TODO (mtourne): remove this (mask of ones)
    return np.ones((img_gray.shape), np.uint8)

def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)


def main():
    filename = sys.argv[1]
    img = cv2.imread(filename)

    img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = get_water_mask(img_gray, img)

    cv2.imwrite('coast_mask.jpg', mask)
    res = apply_mask(img, mask)
    cv2.imwrite('coast_mask_applied.jpg', res)


if __name__ == "__main__":
    main()
