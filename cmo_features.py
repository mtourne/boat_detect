#!/usr/bin/env python
# Copyright 2013 (C) Matthieu Tourne
# @author Matthieu Tourne <matthieu.tourne@gmail.com>

import sys
import cv2
import cv2.cv as cv
import numpy as np
import tempfile

import find_coastline

'''cmo stands for close - open morphology operation
the goal is to keep areas that are much whiter than the background
and also areas that are much darker than the background

See :
D. Casasent and A. Ye, "Detection filters and algorithm fusion
for ATR," Image Processing, IEEE Transactions on, vol. 6, pp. 114-125,
1997.
'''

DEBUG = False

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

def get_polys(img_thresh, in_mask_poly, all_polys):
    ''' find all the boxes of boats in the thresholded image, that are
    not outside of the mask ploygon shape '''
    copy = np.array(img_thresh)
    # mark contours of the blobs
    contours0, hierarchy = cv2.findContours(copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        poly = cv2.approxPolyDP(cnt, 2, True)
        # test all the (4) points of the poly
        for point in poly:
            point = tuple(point[0])
            if cv2.pointPolygonTest(in_mask_poly, point, True) <= 0:
                break
        # none of the points where outside of the mask shape, add to the res list
        all_polys.append(poly)
    return all_polys

def draw_box(vis, polys):
    ''' draw boxes over vis (destructive) '''
    for poly in polys:
        x, y, w, h = cv2.boundingRect(poly)
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
    return vis

def get_all_boats(img_gray, img_gray_orig):
    ''' return a list of image patches with coordinates x,y,w,h in the original
    image
    The actual long, lat of the boat can then be interpolated from the coord of the uav'''
    # find and apply mask, to keep only the water
    masks = find_coastline.get_water_mask(img_gray, img_gray_orig)
    if len(masks) < 1:
        return []

    if DEBUG:
        i = 0
        vis = img_gray.copy()
        vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    all_polys = []
    for (mask, in_mask_poly) in masks:
        img_tmp = find_coastline.apply_mask(img_gray, mask)
        # CMO extraction
        # 0.135 works well
        res = cmo_features(img_tmp, 0.135, object_size=11, cmo_min=True)
        get_polys(res, in_mask_poly, all_polys)

        if DEBUG:
            i += 1
            # write one different coast_mask output per mask
            cv2.imwrite('coast_masked_{}.jpg'.format(i), img_tmp)

    if DEBUG:
        # write all the boat boxes on the same output
        vis = draw_box(vis, all_polys)
        # write img containing boat boxes
        cv2.imwrite('boats_boxes.jpg', vis)

    return all_polys

def get_boat_vignettes(img_orig, img_downsampled, polys):
    ''' return the areas of the original image with boats in them
    add some padding around it '''
    PADDING = 5
    x_resize, y_resize =  img_downsampled.shape
    x_orig, y_orig, _ = img_orig.shape
    x_coeff = np.divide(x_orig, x_resize)
    y_coeff = np.divide(y_orig, y_resize)
    res = []
    for poly in polys:
        x, y, w, h = cv2.boundingRect(poly)
        if x > PADDING:
            x1 = (x - PADDING) * x_coeff
        else:
            x1 = 0
        if y > PADDING:
            y1 = (y - PADDING) * y_coeff
        else:
            y1 = 0
        if x + w + PADDING < x_resize:
            x2 = (x + w + PADDING) * x_coeff
        else:
            x2 = x_orig - 1
        if y + h + PADDING < y_resize:
            y2 = (y + h + PADDING) * y_coeff
        else:
            y2 = y_orig - 1
        # create a temporary file and write the jpeg vignette
        img_patch = img_orig[y1:y2, x1:x2]
        #img_patch = img_downsampled[y:y+h, x:x+w]
        filename = tempfile.mkstemp(prefix='/tmp/boat_', suffix='.jpg')[1]
        cv2.imwrite(filename, img_patch)
        res.append({ 'coord' : (x1, x2, y1, y2), 'img_file': filename})
    return res


def process(filename):
    img = cv2.imread(filename)
    img_gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize 5000 x 4000 images by something more manageable
    img_gray = cv2.resize(img_gray_orig, (0,0), fx=0.2, fy=0.2)

    polys = get_all_boats(img_gray, img_gray_orig)
    # extract boat vignettes from the original image
    # from the image boxes found
    return get_boat_vignettes(img, img_gray, polys)

def main():
    filename = sys.argv[1]
    output = process(filename)
    print output


if __name__ == "__main__":
    main()
