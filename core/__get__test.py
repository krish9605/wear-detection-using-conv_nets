import uav2functions as uav2

import os
import pdb
import time
from copy import deepcopy

import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import urllib2
from skimage import io

t = time.time()
ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))


def get_testdata_from_server(camera=1, test_server_link=False):
    image_list = []
    pos_list = []
    img_pos_list = []
    print('---------------fetching_images--------------')
    img_url = uav2.read_camera_2d(camera, ts)
    pos_url = uav2.read_position(camera, ts)

    if (test_server_link):
        print('---------------testing_server_link--------------')
        image = io.imread(img_url[0]['c_url'])
        cv2.imshow('image', image)
        cv2.waitKey()

    for i in range(len(img_url)):
        image_list.append(io.imread(img_url[i]['c_url']))
        cv2.imwrite("../test/0/" + str(i) + "_imcor.png", io.imread(img_url[i]['c_url']))

    for i in range(len(pos_url)):
        f = urllib2.urlopen(pos_url[i]['t_url'])
        pos = f.read()
        pos_list.append(pos)

    print('---------------packaging_images_w_positions_for_test-----------------')
    for (img, pos) in zip(image_list, pos_list):
        img_pos_list.append([img, pos])

    return img_pos_list


if __name__ == "__main__":
    get_testdata_from_server()
