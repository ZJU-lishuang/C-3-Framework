from PIL import Image
import os
import glob

data_path='../ProcessedData/MULDATASET'
mode="train"
data_files=glob.glob(os.path.join(data_path, '*', mode,'img', '*.jpg'))
max_height=0
max_width=0
for fname in data_files:
    img = Image.open(fname)
    max_width=max(max_width,img.width)
    max_height=max(max_height,img.height)
    print("fname=",fname)
    print("max_width=",max_width,"max_height=",max_height)
