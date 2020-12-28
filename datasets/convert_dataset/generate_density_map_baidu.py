#-*-encoding: utf-8 -*-

import glob
import os
import os.path as path
from PIL import Image
from PIL import ImageDraw
import scipy.io as scio
import numpy as np
import scipy.ndimage
import pickle
from tqdm import tqdm
import pdb
import json
import cv2

# gauss kernel
def gen_gauss_kernels(kernel_size=15, sigma=4):
    kernel_shape  = (kernel_size, kernel_size)
    kernel_center = (kernel_size // 2, kernel_size // 2)

    arr = np.zeros(kernel_shape).astype(float)
    arr[kernel_center] = 1

    arr = scipy.ndimage.filters.gaussian_filter(arr, sigma, mode='constant') 
    kernel = arr / arr.sum()
    return kernel

def gaussian_filter_density(non_zero_points, map_h, map_w):
    """
    Fast gaussian filter implementation : using precomputed distances and kernels
    """
    gt_count = non_zero_points.shape[0]
    density_map = np.zeros((map_h, map_w), dtype=np.float32)

    for i in range(gt_count):
        point_y, point_x = non_zero_points[i]
        #print(point_x, point_y)
        kernel_size = 15 // 2
        kernel = gen_gauss_kernels(kernel_size * 2 + 1, 4)
        min_img_x = int(max(0, point_x-kernel_size))
        min_img_y = int(max(0, point_y-kernel_size))
        max_img_x = int(min(point_x+kernel_size+1, map_h - 1))
        max_img_y = int(min(point_y+kernel_size+1, map_w - 1))
        #print(min_img_x, min_img_y, max_img_x, max_img_y)
        kernel_x_min = int(kernel_size - point_x if point_x <= kernel_size else 0)
        kernel_y_min = int(kernel_size - point_y if point_y <= kernel_size else 0)
        kernel_x_max = int(kernel_x_min + max_img_x - min_img_x)
        kernel_y_max = int(kernel_y_min + max_img_y - min_img_y)
        #print(kernel_x_max, kernel_x_min, kernel_y_max, kernel_y_min)

        density_map[min_img_x:max_img_x, min_img_y:max_img_y] += kernel[kernel_x_min:kernel_x_max, kernel_y_min:kernel_y_max]
    return density_map

mod = 16
dataset = ['SHHA', 'SHHB', 'UCF-QNRF', 'UCF-CC-50', 'GCC',"NWPU","JHU","BAIDUCROWD"][-1]
if dataset == 'SHHA':
    # ShanghaiTech_A
    root, nroot = path.join('ShanghaiTech_Crowd_Detecting', 'partA'), 'SHHA16'
elif dataset == 'SHHB':
    # ShanghaiTech_B
    root, nroot = path.join('ShanghaiTech_Crowd_Detecting', 'partB'), 'SHHB16'
elif dataset == 'UCF-QNRF':
    # UCF-QNRF
    root, nroot = '/home/lishuang/Disk/download/UCF-QNRF_ECCV18', 'UCF-QNRF_16'
elif dataset == 'UCF-CC-50':
    # UCF-CC-50
    root, nroot = 'UCF-CC-50', 'UCF-CC-50_16'
elif dataset == 'GCC':
    root, nroot = path.join('GCC', 'GCC-scene'), path.join('GCC-16')
elif dataset == 'NWPU':
    root, nroot = "/home/lishuang/Disk/download/NWPU-Crowd","/home/lishuang/Disk/download/NWPU-Crowd1024"
elif dataset == 'JHU':
    root, nroot = "/home/lishuang/Disk/download/jhu_crowd_v2.0", "/home/lishuang/Disk/download/jhu_crowd_v2.0_1024"
elif dataset == 'BAIDUCROWD':
    root, nroot = "/home/lishuang/Disk/download/Crowd Surveillance", "/home/lishuang/Disk/download/Crowd_Surveillance1024"

if 'SHH' in dataset:
    # ShanghiTech A and B
    imgps = glob.glob(path.join(root, '*', 'img', '*.jpg'))
elif 'UCF' in dataset:
    #UCF-QNRF and UCF-CC-50
    imgps = glob.glob(path.join(root, '*', '*.jpg'))
elif 'GCC' in dataset:
    imgps = glob.glob(path.join(root, 'scene_*', 'pngs', '*.png'))
elif 'NWPU' in dataset:
    imgps=glob.glob(path.join(root, 'images', '*.jpg'))
elif 'JHU' in dataset:
    imgps=glob.glob(path.join(root, '*', 'images', '*.jpg'))
elif 'BAIDUCROWD' in dataset:
    imgps_train=glob.glob(path.join(root,'images', 'train',  '*.jpg'))
    imgps_test = glob.glob(path.join(root, 'images', 'test', '*.jpg'))
    json_train_ann = json.load(open(os.path.join(root, 'annotations','train.json')))
    json_test_ann = json.load(open(os.path.join(root, 'annotations', 'test.json')))

a = 0
imgps_zip=[imgps_train,imgps_test]
jsons_ann_zip=[json_train_ann,json_test_ann]
for imgps,jsons_ann in zip(imgps_zip,jsons_ann_zip):
    for i, imgp in enumerate(imgps[a:]):
        if "BAIDUCROWD" in dataset:
            json_ann=jsons_ann["annotations"][i]
            imgp=path.join(root,json_ann['name'])
            print(f'[{i + a}]: {imgp}.')
            img = Image.open(imgp)
            img = img.convert('RGB')
            w, h = img.size
            mask_info=json_ann['ignore regions']
            if mask_info != '[]':
                mask_tg = np.ones((h, w), dtype='uint8')
                mask_pts = []
                region_num = len(mask_info)
                drawObject = ImageDraw.Draw(img)
                for j in range(region_num):
                    print('Processing ignore region')
                    mask_pts = np.array(mask_info[j], np.int32)
                    mask_pts = mask_pts.reshape(-1, 2)
                    drawObject.polygon([tuple(x) for x in mask_pts.tolist()], fill="black", outline="black")
                    # img.show()
                    # cv2.fillPoly(mask_tg, [mask_pts], 0)  # Fill 0 in the mask

            imgNo = path.basename(imgp).replace('.jpg', '')
            gt=np.array(json_ann['locations']).reshape(-1,2)

            nimgfold = path.join(nroot, 'train' if 'train' in imgp else 'test', 'img')

            if max(w, h) > 1024:
                if w == max(w, h):
                    nw, nh = 1024, round(h * 1024 / w / mod) * mod
                else:
                    nh, nw = 1024, round(w * 1024 / h / mod) * mod
            else:
                nw, nh = round((w / mod)) * mod, round((h / mod)) * mod

            # new resized image save
            if not path.exists(nimgfold):
                os.makedirs(nimgfold)
            img.resize((nw, nh), Image.BILINEAR).save(path.join(nimgfold, imgNo + ('.jpg' if 'GCC' != dataset else '.png')))
            if len(gt) > 0:
                gt[:, 0] = gt[:, 0].clip(0, w - 1)
                gt[:, 1] = gt[:, 1].clip(0, h - 1)
                gt[:, 0] = (gt[:, 0] / w * nw).round().astype(int)
                gt[:, 1] = (gt[:, 1] / h * nh).round().astype(int)



            # new den csv save
            csvfold = nimgfold.replace('img', 'den')
            if not path.exists(csvfold):
                os.makedirs(csvfold)
            den = gaussian_filter_density(gt, nh, nw)
            np.savetxt(path.join(csvfold, f'{imgNo}.csv'), den, fmt="%.6f", delimiter=",")

            SHOW_MASK = False
            if SHOW_MASK:
                heatmapshow = None
                heatmapshow = cv2.normalize(den, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
                mgtfold = nimgfold.replace('img', 'mask')
                if not path.exists(mgtfold):
                    os.makedirs(mgtfold)
                cv2.imwrite(path.join(mgtfold, f'{imgNo}.jpg'), heatmapshow)

            print(f'-- OK --')