import argparse
import torchvision.datasets as dset



import pdb
from PIL import Image
import numpy as np
import os
import glob

# TODO 

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainDataPath', type=str, default='/media/D/DataSet/UCF-QNRF_ECCV18/train_img', 
                        help='absolute path to your data path')
    return parser

if __name__ == '__main__':
    # args = make_parser().parse_args()

    R_means = []
    G_means = []
    B_means = []
    R_stds = []
    G_stds = []
    B_stds = []
    data_path='../../ProcessedData/MULDATASET'
    data_files = glob.glob(os.path.join(data_path, '*', "train", 'img', '*.jpg'))

    for i_img, img_name in enumerate(data_files):
        if i_img % 100 == 0:
            print( i_img )
        img = Image.open(img_name)
        if img.mode == 'L':
            img = img.convert('RGB')

        # img = np.array(img.resize((1024,768),Image.BILINEAR))

        imgs = np.array(img).astype(np.float32) / 255.
        im_R = imgs[ :, :, 0]
        im_G = imgs[ :, :, 1]
        im_B = imgs[ :, :, 2]
        im_R_mean = np.mean(im_R)
        im_G_mean = np.mean(im_G)
        im_B_mean = np.mean(im_B)
        im_R_std = np.std(im_R)
        im_G_std = np.std(im_G)
        im_B_std = np.std(im_B)
        R_means.append(im_R_mean)
        G_means.append(im_G_mean)
        B_means.append(im_B_mean)
        R_stds.append(im_R_std)
        G_stds.append(im_G_std)
        B_stds.append(im_B_std)


    a = [R_means, G_means, B_means]
    b = [R_stds, G_stds, B_stds]
    mean = [0, 0, 0]
    std = [0, 0, 0]
    mean[0] = np.mean(a[0])
    mean[1] = np.mean(a[1])
    mean[2] = np.mean(a[2])
    std[0] = np.mean(b[0])
    std[1] = np.mean(b[1])
    std[2] = np.mean(b[2])


    print("means: [{}, {}, {}]".format(mean[0],mean[1],mean[2]))
    print("stdevs: [{}, {}, {}]".format(std[0] ,std[1] ,std[2] ))
