#!/usr/bin/env python
# Copyright 2013 (C) Matthieu Tourne
# @author Matthieu Tourne <matthieu.tourne@gmail.com>

import sys
import cv2
import cv2.cv as cv
import numpy as np
import pickle

import utils
import lbp

fd = open("texture/water_texture_hist.bin", 'rb')
WATER_LBP_HIST = pickle.load(fd)
fd.close()

def get_contour_mask(img_gray, contour):
    # "color" is non zero
    color = 255
    mask = np.zeros((img_gray.shape), np.uint8)
    cv2.drawContours(mask, [contour], 0, color,
                     -1, cv2.CV_AA)
    return mask

def get_contour_poly(contour):
    # XXX (mtourne) high precision might make the approxPoly slower ?
    poly = cv2.approxPolyDP(contour, 2, True)
    return poly

def get_mask_patch(img_gray, mask):
    dist = cv2.distanceTransform(mask, cv.CV_DIST_L1, 3)
    max_val = np.amax(dist)
    # force the patch to be max 50 x 50
    if max_val >= 100:
        delta = 25
    else:
        delta = max_val / 4
    x, y = np.unravel_index(np.argmax(dist), mask.shape)
    return img_gray[x-delta:x+delta, y-delta:y+delta]

def get_water_mask(img_gray, img_rgb):
    CANNY_THRSH_LOW = 300
    CANNY_THRSH_HIGH = 600
    #CANNY_THRSH_LOW = 100
    #CANNY_THRSH_HIGH = 2000
    edge = cv2.Canny(img_gray, CANNY_THRSH_LOW, CANNY_THRSH_HIGH, apertureSize=5)
    kern = np.ones((5, 5))
    # dilatation connects most of the disparate edges
    edge = cv2.dilate(edge, kern)
    #utils.visualize_edges(img_rgb, edge, name="coast_edges")
    # invert edges to create contiguous blobs
    edge_inv = np.zeros((img_gray.shape), np.uint8)
    edge_inv.fill(255)
    edge_inv = edge_inv - edge
    #utils.visualize_edges(img_rgb, edge_inv, name="coast_edges_inv")
    contours0, hierarchy0 = cv2.findContours(edge_inv.copy(), cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
    contours = contours0
    # contours of inv edges
    # if contour is >= "relevant area" consider it
    x, y = img_gray.shape
    max_area = x * y
    relevant_area = max_area * 0.05
    potential_masks = []
    res = []
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a >= relevant_area:
            # area is over 5% of the image, test it for water
            mask = get_contour_mask(img_gray, cnt)
            mask_patch = get_mask_patch(img_gray, mask)
            score = lbp.test_patch(WATER_LBP_HIST, mask_patch)
            if score > 0.90:
                # very likely water texture
                poly = get_contour_poly(cnt)
                res.append((mask, poly))
    return res

def apply_mask(img, mask):
    ''' apply mask sets the mask to the mean intensity of the image
    this is so CMO doesn't try to respond with the "mask" '''
    masked = cv2.bitwise_and(img, img, mask=mask)
    # Note other pixels at zero (outside of the mask) are going to be set to mean ..
    img_mean = np.mean(img)
    masked[masked == 0] = img_mean
    return masked
    # this creates a white line between the 2 zones
    #ones = np.ones((img.shape), np.uint8)
    #mask_inv = cv2.bitwise_and(ones, ones, mask=(255 - mask))
    #cv2.imwrite('coast_mask_inv.jpg', mask_inv)
    #return cv2.addWeighted(masked, 1, mask_inv, img_mean, 0)


def main():
    filename = sys.argv[1]
    img = cv2.imread(filename)

    img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    masks = get_water_mask(img_gray, img)
    if len(masks) < 1:
        print "No water"
        return

    vis = img.copy()
    cv2.drawContours( vis, in_poly, -1, (0, 255, 0), 3 )
    cv2.imwrite('coast_mask_poly.jpg', vis)

    cv2.imwrite('coast_mask.jpg', mask)
    res = apply_mask(img, mask)
    cv2.imwrite('coast_mask_applied.jpg', res)


if __name__ == "__main__":
    main()
