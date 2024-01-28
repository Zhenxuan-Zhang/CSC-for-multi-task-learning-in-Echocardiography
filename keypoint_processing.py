#traning set
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
import numpy.matlib
def landmark(center_x,center_y,IMAGE_HEIGHT, IMAGE_WIDTH):
    R = np.sqrt(2**2 + 2**2)
    Gauss_map = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    mask_x = np.matlib.repmat(center_x, IMAGE_HEIGHT, IMAGE_WIDTH)
    mask_y = np.matlib.repmat(center_y, IMAGE_HEIGHT, IMAGE_WIDTH)
    x1 = np.arange(IMAGE_WIDTH)
    x_map = np.matlib.repmat(x1, IMAGE_HEIGHT, 1)
    y1 = np.arange(IMAGE_HEIGHT)
    y_map = np.matlib.repmat(y1, IMAGE_WIDTH, 1)
    y_map = np.transpose(y_map)
    Gauss_map = np.sqrt((x_map-mask_x)**2+(y_map-mask_y)**2)
    Gauss_map = np.exp(-0.5*Gauss_map/R)
    return Gauss_map

def locmap(pot):
    gauss_batch = []
    for i in range(0,pot.shape[0]):
        gauss_tp = []
        for j in range(0,pot.shape[1]):
            g_map1 = landmark(pot[i, j, 0, 0],pot[i, j, 0, 1],128,128)
            g_map2 = landmark(pot[i, j, 1, 0],pot[i, j, 1, 1],128,128)
            g_map3 = landmark(pot[i, j, 2, 0],pot[i, j, 2, 1],128,128)
            g_map4 = landmark(pot[i, j, 3, 0],pot[i, j, 3, 1],128,128)
            Gauss_map = (g_map1+g_map2+g_map3+g_map4)/4
            gauss_tp.append(Gauss_map)
        gauss_batch.append(gauss_tp)
    gauss_batch = np.array(gauss_batch)[:, :, :, :, np.newaxis]
    return gauss_batch