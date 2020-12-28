import numpy as np
import os
import random
from scipy import io as sio
import sys
import torch
from torch.utils import data
from PIL import Image, ImageOps

import pandas as pd
import glob

from config import cfg
from .setting import cfg_data


def letterbox(img,den, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    # shape = img.shape[:2]  # current shape [height, width]
    shape=[img.height,img.width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        # dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img=img.resize(new_unpad, Image.BILINEAR)
        den = den.resize(new_unpad, Image.BILINEAR)
        # img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = ImageOps.expand(img, (left, top, right, bottom), fill=color)
    den = ImageOps.expand(den, (left, top, right, bottom), fill=(0))
    # img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img,den, ratio, (dw, dh)

class muldataset(data.Dataset):
    def __init__(self, data_path, mode, main_transform=None, img_transform=None, gt_transform=None):

        # self.img_path = data_path + '/img'
        # self.gt_path = data_path + '/den'
        self.data_files=glob.glob(os.path.join(data_path, '*', mode,'img', '*.jpg'))
        # self.data_files = [filename for filename in os.listdir(self.img_path) \
        #                    if os.path.isfile(os.path.join(self.img_path,filename))]
        self.num_samples = len(self.data_files) 
        self.main_transform=main_transform  
        self.img_transform = img_transform
        self.gt_transform = gt_transform     
    
    def __getitem__(self, index):
        fname = self.data_files[index]
        img, den = self.read_image_and_gt(fname)      
        if self.main_transform is not None:
            img, den = self.main_transform(img,den) 
        if self.img_transform is not None:
            img = self.img_transform(img)         
        if self.gt_transform is not None:
            den = self.gt_transform(den)               
        return img, den

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self,fname):
        # img = Image.open(os.path.join(self.img_path,fname))
        img = Image.open(fname)
        if img.mode == 'L':
            img = img.convert('RGB')


        # den = sio.loadmat(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.mat'))
        # den = den['map']
        den = pd.read_csv(os.path.join(fname.replace('.jpg', '.csv').replace('img/', 'den/')), sep=',',header=None).values
        
        den = den.astype(np.float32, copy=False)    
        den = Image.fromarray(den)
        img, den, ratio, pad = letterbox(img,den, cfg_data.TRAIN_SIZE, auto=False)
        ##Image.fromarray(np.asarray(den) * 255 * 100).show()
        return img, den



    def get_num_samples(self):
        return self.num_samples       
            
        