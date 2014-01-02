#!/usr/bin/env python
# Copyright 2013 (C) Matthieu Tourne
# @author Matthieu Tourne <matthieu.tourne@gmail.com>

import cv2
import numpy as np

from numpy.lib.stride_tricks import as_strided

def correct_gamma(img_rgb, gamma):
    inverse_gamma = 1.0 / gamma

    lut_matrix = []
    for i in range(0, 256):
        lut = (np.true_divide(i, 255) ** inverse_gamma) * 255.0
        lut_matrix.append(lut)
    lut_matrix = np.array(lut_matrix)
    res = cv2.LUT(img_rgb, lut_matrix)
    return np.uint8(res)


def get_gradient(img_gray):
    # grad x (with sobel, could be with scharr)
    grad_x = cv2.Sobel(img_gray, -1, 1, 0)
    abs_grad_x = cv2.convertScaleAbs(grad_x)

    # grad y
    grad_y = cv2.Sobel(img_gray, -1, 0, 1)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # Total Gradient (approximate)
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# incorrect or not good
def equalize_hist_rgb(img_rgb):
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2YCR_CB)
    y = ycrcb[:,:,0]
    y = cv2.equalizeHist(y)
    ycrcb[:,:,0] = y
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)

def rgb_normalized(rgb_input):
    rgb = np.float32(rgb_input)
    norm = np.zeros(rgb.shape, np.float32)
    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]
    # sum of the 3 chan on each point (matrix of m x n)
    sum=b+g+r
    # avoid 0 divs
    sum[sum == 0] = 1
    b = b / sum
    g = g / sum
    r = r / sum
    # normalize from 0 - 255
    b = ((b - b.min()) / (b.max() - b.min())) * 255
    g = ((g - g.min()) / (g.max() - g.min())) * 255
    r = ((r - r.min()) / (r.max() - r.min())) * 255
    norm[:,:,0] = b
    norm[:,:,1] = g
    norm[:,:,2] = r
    return np.uint8(norm)

def hist_curve(im, mask):
    ''' compute histogram of the image,
    and then plot its curve '''
    bins = np.arange(256).reshape(256,1)
    h = np.zeros((300,256,3))
    if len(im.shape) == 2:
        color = [(255,255,255)]
    elif im.shape[2] == 3:
        color = [ (255,0,0),(0,255,0),(0,0,255) ]
    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([im],[ch],mask,[256],[0,256])
        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        pts = np.int32(np.column_stack((bins,hist)))
        cv2.polylines(h,[pts],False,col)
    y=np.flipud(h)
    return y

def get_hist_curves(img_rgb, img_hsv=None, rgb_norm=None, mask=None, name=""):
    ''' print out the histogram curve for rgb, hsv,
    rg chroma (rgb normalized)'''
    curve = hist_curve(img_rgb, mask)
    cv2.imwrite('{}_rgb.jpg'.format(name), curve)

    if img_hsv is not None:
        curve = hist_curve(img_hsv, mask)
        cv2.imwrite('{}_hsv.jpg'.format(name), curve)

    if rgb_norm is not None:
        curve = hist_curve(rgb_norm, mask)
        cv2.imwrite('{}_rgbnorm.jpg'.format(name), curve)

def visualize_edges(img, edges, name=""):
    vis = img.copy()
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    vis[edges != 0] = (0, 255, 0)
    cv2.imwrite("{}.jpg".format(name), vis)

def visualize_cmo(cmo_img, name=""):
    cmo_img[cmo_img < 0] = 0
    # normalization
    cmo_img = (cmo_img - cmo_img.min()) / (cmo_img.max() - cmo_img.min()) * 255
    cv2.imwrite(name, np.uint8(cmo_img))

def get_means_std(img_rgb, img_hsv=None, rgb_norm=None, mask=None):
    b=img_rgb[:,:,0]
    mean, std = cv2.meanStdDev(b, mask=mask)
    print "Blue chan, mean: {}, std: {}".format(
        mean, std)
    g=img_rgb[:,:,1]
    mean, std = cv2.meanStdDev(g, mask=mask)
    print "Green chan, mean: {}, std: {}".format(
        mean, std)
    r=img_rgb[:,:,2]
    mean, std = cv2.meanStdDev(r, mask=mask)
    print "Red chan, mean: {}, std: {}".format(
        mean, std)

    if img_hsv is not None:
        h=img_hsv[:,:,0]
        mean, std = cv2.meanStdDev(h, mask=mask)
        print "H chan, mean: {}, std: {}".format(
            mean, std)
        s=img_hsv[:,:,1]
        mean, std = cv2.meanStdDev(s, mask=mask)
        print "S chan, mean: {}, std: {}".format(
            mean, std)
        v=img_hsv[:,:,2]
        mean, std = cv2.meanStdDev(v, mask=mask)
        print "V chan, mean: {}, std: {}".format(
            mean, std)

    if rgb_norm is not None:
        b_norm=rgb_norm[:,:,0]
        mean, std = cv2.meanStdDev(b_norm, mask=mask)
        print "Blue chan Norm, mean: {}, std: {}".format(
            mean, std)
        g_norm=rgb_norm[:,:,1]
        mean, std = cv2.meanStdDev(g_norm, mask=mask)
        print "Green chan Norm, mean: {}, std: {}".format(
            mean, std)
        r_norm=rgb_norm[:,:,2]
        mean, std = cv2.meanStdDev(r_norm, mask=mask)
        print "Red chan Norm, mean: {}, std: {}".format(
            mean, std)

def image_windows(A, xsize, ysize, xstep=1, ystep=1):
    all_windows = as_strided(
        A, ((A.shape[0] - xsize + 1) / xstep, (A.shape[1] - ysize + 1) / ystep, xsize, ysize),
        (A.strides[0] * xstep, A.strides[1] * ystep, A.strides[0], A.strides[1]))
    return all_windows

# palette data from matplotlib/_cm.py
_jet_data =   {'red':   ((0., 0, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89,1, 1),
                         (1, 0.5, 0.5)),
               'green': ((0., 0, 0), (0.125,0, 0), (0.375,1, 1), (0.64,1, 1),
                         (0.91,0,0), (1, 0, 0)),
               'blue':  ((0., 0.5, 0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65,0, 0),
                         (1, 0, 0))}

cmap_data = { 'jet' : _jet_data }

def make_cmap(name, n=256):
    data = cmap_data[name]
    xs = np.linspace(0.0, 1.0, n)
    channels = []
    eps = 1e-6
    for ch_name in ['blue', 'green', 'red']:
        ch_data = data[ch_name]
        xp, yp = [], []
        for x, y1, y2 in ch_data:
            xp += [x, x+eps]
            yp += [y1, y2]
        ch = np.interp(xs, xp, yp)
        channels.append(ch)
    return np.uint8(np.array(channels).T*255)
