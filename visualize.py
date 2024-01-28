#traning set
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
import numpy.matlib
from torchvision.transforms import transforms
import cv2
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import timeit
import os
from tqdm import tqdm
from tqdm import trange
import sys
import random
import itertools
import colorsys
from skimage.measure import find_contours
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display

unloader = transforms.ToPILImage()

def tensor_to_img(tensor_pred , data_num, frame_num,title=None):
    #image = unloader(image)
    #tensor_pred = tensor_pred.type(torch.float32)
    #pred = tensor_pred.cpu().clone()  # we clone the tensor to not do changes on it
    pred = tensor_pred
    pred = unloader(pred)
    #fig = plt.figure()
    pred = np.array(pred)
    pred = cv2.resize(pred, (256,256) , interpolation=cv2.INTER_AREA) 
    #plt.imshow(pred, cmap = 'gray')
    return pred

def tensor_to_lb(tensor_pred , data_num, frame_num,title=None):
    #image = unloader(image)
    tensor_pred = torch.sigmoid(tensor_pred)
    tensor_pred = torch.gt(tensor_pred, 0.5)
    tensor_pred = tensor_pred.type(torch.float32)
    pred = tensor_pred.cpu().clone()  # we clone the tensor to not do changes on it
    pred = unloader(pred)
    #fig = plt.figure()
    pred = np.array(pred)
    pred = cv2.resize(pred, (256,256) , interpolation=cv2.INTER_AREA) 
    #plt.imshow(pred, cmap = 'gray')
    return pred


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.4):
    """Apply the given mask to the image.
    """
    image = image.copy()
    mask_out = cv2.Canny(mask.astype(np.uint8),0,1)
    kernel = np.ones((2, 2), dtype=np.uint8)
    mask_out = cv2.dilate(mask_out, kernel, 1)
    for c in range(3):
        image[:, :, c] = np.where(mask >= 0.5,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    for c in range(3):
        image[:, :, c] = np.where(mask_out >= 0.5,
                                  color[c] * 255,
                                  image[:, :, c])
    return image

unloader = transforms.ToPILImage()
train_mode = 'mtl'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True
l_r = 0.0002  #0.0002
device = torch.device("cuda")
def fill_contour(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)  # 轮廓的个数
    max_area = 0
    for i  in range(n):
        if cv2.contourArea(contours[i]) > max_area:
            max_area = cv2.contourArea(contours[i])
    cv_contours = []
    if n != 1:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < max_area:
                cv_contours.append(contour)
                x, y, w, h = cv2.boundingRect(contour)
                img[y:y + h, x:x + w] = 1
            else:
                continue
    else:
        pass
    
    return img

def tensor_im(tensor_pred , data_num, frame_num,title=None):
    #tensor_pred = torch.sigmoid(tensor_pred)
    #tensor_pred = torch.gt(tensor_pred, 0.5)
    tensor_pred = tensor_pred.type(torch.float32)
    pred = tensor_pred.cpu().clone()  # we clone the tensor to not do changes on it
    pred = unloader(pred)
    pred = np.array(pred)
    #plt.imshow(pred)
    return pred
    
def tensor_save(tensor_pred , data_num, frame_num,title=None):
    tensor_pred = torch.sigmoid(tensor_pred)
    tensor_pred = torch.gt(tensor_pred, 0.5)
    tensor_pred = tensor_pred.type(torch.float32)
    pred = tensor_pred.cpu().clone()  # we clone the tensor to not do changes on it
    pred = unloader(pred)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10, 10))
    pred = np.array(pred)
    #plt.imshow(pred)
    return pred

def landmark(center_x,center_y,IMAGE_HEIGHT, IMAGE_WIDTH):
    R = np.sqrt(1**1 + 1**1)
    Gauss_map = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    # 直接利用矩阵运算实现
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

def norm(img):
    img = np.array(img, dtype=np.float32)
    img -= np.mean(img)
    img /= (np.std(img) + 1e-12)
    return img
    
def locmap(pot):
    gauss_batch = []
    for i in range(0,pot.shape[0]):
        gauss_tp = []
        for j in range(0,pot.shape[1]):
            g_map1 = landmark(pot[i, j, 0, 0],pot[i, j, 0, 1],128,128)
            g_map2 = landmark(pot[i, j, 1, 0],pot[i, j, 1, 1],128,128)
            g_map3 = landmark(pot[i, j, 2, 0],pot[i, j, 2, 1],128,128)
            g_map4 = landmark(pot[i, j, 3, 0],pot[i, j, 3, 1],128,128)
            Gauss_map = [g_map1,g_map2,g_map3,g_map4]
            Gauss_map = norm(Gauss_map)
            gauss_tp.append(Gauss_map)
        gauss_batch.append(gauss_tp)
    gauss_batch = np.array(gauss_batch)[:, :, :, :, :]
    return gauss_batch

def point_color(potmap):
    image = np.zeros((128,128,3),dtype=int)
    kernel = np.ones((2, 2), dtype=np.uint8)
    color1 = [1,1,0]
    color2 = [0,1,1]
    color3 = [1,0,1]
    color4 = [1,0,0]
    alpha = 1
    k = 10
    for c in range(3):
        image[:, :, c] = np.where(potmap[0, :, :] >= k,alpha * color1[c] * 255,
                                  image[:, :, c])
    for c in range(3):
        image[:, :, c] = np.where(potmap[1, :, :] >= k,alpha * color2[c] * 255,
                                  image[:, :, c])
    for c in range(3):
        image[:, :, c] = np.where(potmap[2, :, :] >= k,alpha * color3[c] * 255,
                                  image[:, :, c])
    for c in range(3):
        image[:, :, c] = np.where(potmap[3, :, :] >= k,alpha * color4[c] * 255,
                                  image[:, :, c])
    return image
    
model = Mtl_net(img_dim=128,in_channels=1,out_channels=128,head_num=4,mlp_dim=512,block_num=8,
                     patch_dim=16,class_num=1,drop_rate = 0.2,seq_frame=30,mode =train_mode,height=128,weight=128).to(device)
model.load_state_dict(torch.load(''),True)
model.train()
size = 128
ki = image_test.to(device)
kg = label_test.to(device)
kp = point_test.to(device)
kf = frame_test.to(device)
for i in range(99,109):
    a = ki[i:i+1]
    d = kg[i:i+1]
    c = kp[i:i+1]
    e = kf[i:i+1]
    plt.subplots(figsize=(15,40))
    color_lvla = [(1,0,0),(0,1,0)]
    with torch.no_grad():
        b,b1,b2,b3,b4 = model(a)
        #torch.tanh(b)
        print(a.shape,c.shape,b.shape,b1.shape,b2.shape,b3.shape,b4.shape)
        print(b1[0,0,:,:],c[0,0,:,:])
        print(e,b3)
        for k in range(2):
            j = k
            plt.subplot(161)
            img = tensor_im(a[0,j,0,:,:].to(torch.uint8),0,1).astype(np.uint8)
            alp,bet = 53,11
            plt.imshow(img*alp+bet,cmap = 'gray')
            plt.axis('off')
            image_ori = np.zeros((128,128,3),dtype=int)
            image_ori[:,:,0] = (img*alp+bet)[:,:]
            image_ori[:,:,1] = (img*alp+bet)[:,:]
            image_ori[:,:,2] = (img*alp+bet)[:,:]
            plt.subplot(162)
            gt = tensor_save(d[0,j,0,:,:].to(torch.float32),0,1)
            gt = apply_mask(image_ori,gt,color_lvla[1])
            plt.imshow(gt)
            plt.axis('off')
            plt.subplot(163)
            pred = tensor_save(b4[0,j,0,:,:].to(torch.float32),0,1)
            pred = apply_mask(image_ori,pred,color_lvla[1])
            plt.imshow(pred)
            plt.axis('off')
            plt.subplot(164)
            x = locmap(np.array(c.cpu()))[0,j,:,:]
            max_index = np.unravel_index(np.argmax(x, axis=None), x.shape)
            max_value = x[max_index]
            #print(max_index,max_value)
            comap = point_color(locmap(np.array(c.cpu()))[0,j,:,:])
            plt.imshow(comap)
            plt.axis('off')
            plt.subplot(165)
            b1map = point_color(locmap(np.array(b1.cpu()))[0,j,:,:])
            plt.imshow(b1map)
            plt.axis('off')
            plt.subplot(166)
            b1map = point_color(locmap(np.array(b1.cpu()))[0,j,:,:])
            # plt.imshow(b1map)
            # plt.axis('off')
            peak_list = np.array(e.cpu()[0])
            # print(peak_list.shape)
            if peak_list[j] == -1:
                plt.imshow(img,cmap = 'PuRd_r')
            elif peak_list[j] == 0:
                plt.imshow(img)
            else:
                plt.imshow(img,cmap = 'BrBG_r')
            plt.axis('off')