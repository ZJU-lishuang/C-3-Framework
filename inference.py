from matplotlib import pyplot as plt

import matplotlib
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd

from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

exp_name = '../test_results'
if not os.path.exists(exp_name):
    os.mkdir(exp_name)

if not os.path.exists(exp_name + '/pred'):
    os.mkdir(exp_name + '/pred')

if not os.path.exists(exp_name + '/gt'):
    os.mkdir(exp_name + '/gt')

mean_std = ([0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611])
img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])
restore = standard_transforms.Compose([
    own_transforms.DeNormalize(*mean_std),
    standard_transforms.ToPILImage()
])
pil_to_tensor = standard_transforms.ToTensor()

dataRoot = '../ProcessedData/'

model_path = './exp/12-21_03-16_SHHB_Res50_1e-05/all_ep_41_mae_8.2_mse_13.9.pth'
model_path = '/home/lishuang/Disk/gitlab/traincode/crowd_counting/PyTorch_Pretrained/all_ep_93_mae_68.8_mse_402.4.pth'


def main():
    file_list = [filename for root, dirs, filename in os.walk(dataRoot + '/testimg/')]

    inference(file_list[0], model_path)


def inference(file_list, model_path):
    # net = CrowdCounter(cfg.GPU_ID, cfg.NET)
    # net.load_state_dict(torch.load(model_path))
    # net.cuda()
    # net.eval()

    net = CrowdCounter(cfg.GPU_ID, cfg.NET)
    net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
    net.cuda()
    net.eval()

    # f1 = plt.figure(1)
    #
    # gts = []
    # preds = []

    for filename in file_list:
        print(filename)
        imgname = dataRoot + '/testimg/' + filename
        # imgname = dataRoot + '/img/' + filename
        filename_no_ext = filename.split('.')[0]


        img = Image.open(imgname)

        if img.mode == 'L':
            img = img.convert('RGB')

        img = img_transform(img)

        with torch.no_grad():
            img = Variable(img[None, :, :, :]).cuda()
            pred_map = net.test_forward(img)

        sio.savemat(exp_name + '/pred/' + filename_no_ext + '.mat', {'data': pred_map.squeeze().cpu().numpy() / 100.})

        pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]

        pred = np.sum(pred_map) / 100.0
        pred_map = pred_map / np.max(pred_map + 1e-20)


        pred_frame = plt.gca()
        plt.imshow(pred_map, 'jet')
        pred_frame.axes.get_yaxis().set_visible(False)
        pred_frame.axes.get_xaxis().set_visible(False)
        pred_frame.spines['top'].set_visible(False)
        pred_frame.spines['bottom'].set_visible(False)
        pred_frame.spines['left'].set_visible(False)
        pred_frame.spines['right'].set_visible(False)
        plt.savefig(exp_name + '/' + filename_no_ext + '_pred_' + str(float(pred)) + '.png', \
                    bbox_inches='tight', pad_inches=0, dpi=150)

        plt.close()

        # sio.savemat(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.mat',{'data':pred_map})




if __name__ == '__main__':
    main()




