#!/usr/bin/env python
# Copyright 2013 (C) Matthieu Tourne
# @author Matthieu Tourne <matthieu.tourne@gmail.com>

import sys
import cv2
import cv2.cv as cv
import numpy as np

import find_coastline

'''cmo stands for close - open morphology operation
the goal is to keep areas that are much whiter than the background
and also areas that are much darker than the background

See :
D. Casasent and A. Ye, "Detection filters and algorithm fusion
for ATR," Image Processing, IEEE Transactions on, vol. 6, pp. 114-125,
1997.
'''

# entropy threshold also does a good job at finding
# interesting things in something that has very little entropy (like water)
# it runs in O(k * n) (n pixels, k entropies to test for in the ranges)
def find_entropy_threshold(img_gray, ranges):
    hist_item = cv2.calcHist([img_gray],[0],None,[256],[0,256])
    # compute a mask of non zero items to avoid
    # computing a log(0) later on
    mask = hist_item != 0
    entropies = []
    for k in ranges:
        # [:k] is [0, k[, but [k:] is [k, end]
        range1 = hist_item[:k]
        range2 = hist_item[k:]

        sum_r1 = np.sum(range1)
        sum_r2 = np.sum(range2)
        # protect against div by zero / log(0) => nan
        if sum_r1 == 0 or sum_r2 == 0:
            # insert a 0 to keep the same len as "ranges"
            entropies.append(0)
            continue

        mask1 = mask[:k]
        mask2 = mask[k:]

        sum_ln_r1 = np.sum(np.multiply(range1[mask1], np.log(range1[mask1])))
        sum_ln_r2 = np.sum(np.multiply(range2[mask2], np.log(range2[mask2])))

        entropy = (np.log(sum_r1) + np.log(sum_r2)
                   - np.true_divide(sum_ln_r1, sum_r1)
                   - np.true_divide(sum_ln_r2, sum_r2))
        entropies.append(entropy)
    entropies = np.array(entropies)
    # get the maximum "k"
    k = np.argmax(entropies)
    # return a value in the original range
    return ranges[k]

def entropy_features(img_gray):
    ERODE_KERN = 3
    # Test 24 thresholds using entropy thresholding
    ranges = np.arange(100, 240, 10)
    thresh = find_entropy_threshold(img_gray, ranges)
    print thresh
    _, res = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)

    # final opening to remove eventual micro features
    kern = np.ones((ERODE_KERN, ERODE_KERN))
    #res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kern)
    #res = cv2.erode(res, kern)
    #res = cv2.dilate(res, kern)
    return res


def cmo(src, opened, closed):
    # * sign preserving cmo
    #   white features > 0, dark features < 0 (maxima)
    #   return (2 * src) - closed - opened
    # * normal cmo
    #   both white and dark features are maxima > 0
    return closed - opened

def full_cmo(img_gray, object_size, min=True):
    # 1-D horizontal and vertical SE (structural element)

    horiz = np.ones((1, object_size), dtype=np.uint8)
    vert = np.ones((object_size, 1), dtype=np.uint8)

    opened_horiz = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, horiz)
    closed_horiz = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, horiz)
    cmo_horiz = cmo(img_gray, opened_horiz, closed_horiz)

    opened_vert = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, vert)
    closed_vert = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, vert)
    cmo_vert = cmo(img_gray, opened_vert, closed_vert)

    # literature says minimum, but sometimes max works better!
    # creates more false positives obviously
    if min:
        res = np.minimum(cmo_horiz, cmo_vert)
    else:
        res = np.maximum(cmo_horiz, cmo_vert)
    return res

def cmo_features(img_gray, thresh, object_size=7, cmo_min=True):
    ''' object_size is the kernel size for cmo
    Note : doesn't work too well if the object_size is too big
    '''
    # final erode
    ERODE_KERN = 3
    # theta for gaussian smoothing (not that important)
    THETA_SMOOTH = 2

    # smooth the result of CMO over half of the object size
    smooth_size = object_size // 2
    if smooth_size %2 == 0:
        smooth_size += 1

    # convert img to 0 - 1 floating img
    img_gray = np.float32(img_gray) / 255;

    # compute full cmo on horiz and vert 1D slits
    cmo = full_cmo(img_gray, object_size, min=cmo_min)
    #utils.visualize_cmo(cmo, name="boats_cmo_step1.jpg")

    # gaussian filter over the output, to smooth results
    cmo = cv2.GaussianBlur(cmo, (smooth_size, smooth_size),
                           THETA_SMOOTH)
    #utils.visualize_cmo(cmo, name="boats_cmo_step2.jpg")

    # threshold
    ret, res = cv2.threshold(cmo, thresh, 255, cv2.THRESH_BINARY)

    # convert back to integer
    res = np.uint8(res)

    # final opening to remove eventual micro features
    kern = np.ones((ERODE_KERN, ERODE_KERN))
    res = cv2.erode(res, kern)
    #utils.visualize_cmo(res, name="boats_cmo_step3.jpg")
    res = cv2.dilate(res, kern, iterations=2)
    return res

# this looks a lot like the function above
def process_threshed(img):
    ERODE_KERN = 2

    # final opening to remove eventual micro features
    kern = np.ones((ERODE_KERN, ERODE_KERN))
    res = cv2.erode(img, kern)
    res = cv2.dilate(res, kern, iterations=2)
    return res

def get_polys(img_thresh):
    copy = np.array(img_thresh)
    # mark contours of the blobs
    contours0, hierarchy = cv2.findContours(copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # TODO (mtourne): filter out areas that are outside of a range
    # return a list of contours
    return [ cv2.approxPolyDP(cnt, 2, True) for cnt in contours0 ]


# in_poly is the region of interest
def draw_box(vis, contours, in_poly):
    ''' draw boxes over vis (destructive) '''
    for contour in contours:
        draw = True
        for point in contour:
            point = tuple(point[0])
            if cv2.pointPolygonTest(in_poly, point, True) <= 0:
                draw = False
                break

        # draw it over the img
        if draw:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
    return vis

def main():
    filename = sys.argv[1]
    img = cv2.imread(filename)

    # resize 5000 x 4000 images by something more manageable
    img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find and apply mask, to keep only the water
    masks = find_coastline.get_water_mask(img_gray, img)
    if len(masks) < 1:
        print "No water"
        return

    vis = img.copy()
    i = 0
    for (mask, in_poly) in masks:
        i += 1
        img_tmp = find_coastline.apply_mask(img_gray, mask)
        cv2.imwrite('coast_masked_{}.jpg'.format(i), img_tmp)

        # CMO extraction
        # 0.135 works well
        res = cmo_features(img_tmp, 0.135, object_size=11, cmo_min=True)
        polys = get_polys(res)
        vis = draw_box(img, polys, in_poly)

    cv2.imwrite('boats_boxes.jpg', vis)


if __name__ == "__main__":
    main()
