#!/usr/bin/env python

import numpy as np
import cv2
import cv2.cv as cv
import sys

from skimage.feature import local_binary_pattern

import utils
import pickle
import glob

# settings for LBP
METHOD = 'uniform'

DEBUG = False

def hist_intersection(p, q):
    return np.sum(np.minimum(p, q))

def load_lbp(filename):
    img_gray = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    lbp = get_lbp_hist(img_gray)
    return lbp

def get_lbp_hist(img_patch):
    # lbp 1, 8
    radius = 1
    n_points = 8
    lbp1 = local_binary_pattern(img_patch, n_points, radius, METHOD)
    hist1 = get_hist(lbp1, n_points + 2)
    # lbp 2, 16
    radius = 2
    n_points = 16
    lbp2 = local_binary_pattern(img_patch, n_points, radius, METHOD)
    hist2 = get_hist(lbp2, n_points + 2)
    # lbp 3, 24
    radius = 3
    n_points = 24
    lbp3 = local_binary_pattern(img_patch, n_points, radius, METHOD)
    hist3 = get_hist(lbp3, n_points + 2)
    # stacking histograms is not a real way to do joint probabilities
    # but in practice it works fine
    hists = np.hstack((hist1, hist2, hist3))
    # divide the 3 hists by 3
    return np.multiply(hists, 1. / 3.)

def get_hist(lbp, bins):
    hist, _ = np.histogram(lbp, bins=bins, normed=True, range=(0, bins))
    return hist

def init_water_textures():
    histograms = []
    for filename in glob.glob('texture/water_texture*.png'):
        print filename
        img_gray = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        hists = get_lbp_hist(img_gray)
        histograms.append((filename, hists))
        # Note (mtourne): eventually build an image pyramid
        # to accomodate multiscale textures
    return histograms

def get_max_similarity(ref_hists, hist_test):
    max_score = 0
    for (filename, hist) in ref_hists:
        score = hist_intersection(hist, hist_test)
        if score > max_score:
            best_filename = filename
            max_score = score
    if DEBUG:
        print("Best similarity: {}, original texture: {}".format(
            max_score, best_filename))
    return max_score

def test_texture(ref_hists, filename):
    hist_test = load_lbp(filename)
    score = get_max_similarity(ref_hists, hist_test)
    return score

def test_patch(ref_hists, img_patch):
    lbp_hist = get_lbp_hist(img_patch)
    score = get_max_similarity(ref_hists, lbp_hist)
    if DEBUG:
        cv2.imwrite('coast_water_patch.jpg', img_patch)
    return score

def main():
    ref_hists = init_water_textures()
    print "Water"
    test_texture(ref_hists, "texture/test_water_texture.jpg")
    print "Brick"
    test_texture(ref_hists, "texture/test_brick_texture.jpg")
    if len(sys.argv) > 1:
        print sys.argv[1]
        test_texture(ref_hists, sys.argv[1])
    fd = open("texture/water_texture_hist.bin", "wb")
    pickle.dump(ref_hists, fd)
    fd.close()

if __name__ == '__main__':
    main()
