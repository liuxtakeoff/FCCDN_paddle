
import copy
import os.path as osp
from pathlib import Path
from paddle.io import Dataset
import paddle
import albumentations
import PIL.Image as Image
import scipy
import random
from paddle.vision.transforms import functional as F
import paddle.vision.transforms as pptransforms
import numpy as np
from cv2 import cv2

def random_flip(imgs,prob = 0.5):
    if random.random()<=prob:
        imgs[0] = albumentations.hflip(imgs[0])
        imgs[1] = albumentations.hflip(imgs[1])
        imgs[2] = albumentations.hflip(imgs[2])
    if random.random()<=prob:
        imgs[0] = albumentations.vflip(imgs[0])
        imgs[1] = albumentations.vflip(imgs[1])
        imgs[2] = albumentations.vflip(imgs[2])
    return imgs

def random_transpose(imgs,prob=0.5):
    if random.random()<=prob:
        imgs[0] = albumentations.transpose(imgs[0])
        imgs[1] = albumentations.transpose(imgs[1])
        imgs[2] = albumentations.transpose(imgs[2])
    return imgs

def random_rotate(imgs,prob=0.3):
    deg = random.uniform(-45,45)
    if random.random() <= prob:
        imgs[0] = albumentations.rotate(imgs[0],angle=deg)
        imgs[1] = albumentations.rotate(imgs[1],angle=deg)
        imgs[2] = albumentations.rotate(imgs[2],angle=deg)
    return imgs

def random_zoom(imgs,prob=0.3,maxscale=0.1):
    if random.random() <= prob:
        scale = random.uniform(1 - maxscale, 1 + maxscale)
        origin_H, origin_W = imgs[0].shape[0], imgs[0].shape[1]
        new_H,new_W = int(origin_H*scale),int(origin_W*scale)
        imgs[0] = cv2.resize(imgs[0],(new_H,new_W))
        imgs[1] = cv2.resize(imgs[2],(new_H,new_W))
        imgs[2] = cv2.resize(imgs[2],(new_H,new_W))
        if scale > 1:
            err_H = new_H - origin_H
            err_W = new_W - origin_W
            loc_h = random.randint(0,err_H)
            loc_w = random.randint(0,err_W)
            imgs[0] = imgs[0][loc_h:loc_h+origin_H,loc_w:loc_w+origin_W]
            imgs[1] = imgs[1][loc_h:loc_h + origin_H, loc_w:loc_w + origin_W]
            imgs[2] = imgs[2][loc_h:loc_h + origin_H, loc_w:loc_w + origin_W]
        elif scale < 1:
            #TODO:实现 zoom out
            pass
    return imgs

def random_shiftHSV(imgs,prob=0.3,H_shift=10,S_shift=5,V_shift=10):
    # TODO:测试哪种增强方式更好：分别变换，同步变换
    # if random.random()<=prob:
        #TODO:测试一下randomint会不会输出上限和下限
        # H_shift = random.randint(-H_shift,H_shift)
        # V_shift = random.randint(-V_shift,V_shift)
        # S_shift = random.randint(-S_shift,S_shift)
        # imgs[0] = albumentations.shift_hsv(imgs[0],H_shift,S_shift,V_shift)
    trans = albumentations.augmentations.transforms.HueSaturationValue(H_shift, S_shift, V_shift, p=prob)
    imgs[0] = trans(imgs[0])
    imgs[1] = trans(imgs[1])
    return imgs

def random_gaussiannoise(imgs,prob=0.3,mean=0,var=(10.0,50.0)):
    # TODO:测试哪种增强方式更好：分别变换，同步变换
    # if random.random()<=prob:
    #     _var = random.uniform(10,50)
    #     imgs[0] = np.array(imgs[0],dtype="float")
    #     noise = np.random.normal(mean,var**0.5,imgs[0].shape)
    #     imgs[0]+=noise
    #     np.clip(imgs[0],0,255)
    #     imgs[0].astype("uint8")
    #     imgs[0] = Image.fromarray(imgs[0])
    #
    #     imgs[1] = np.array(imgs[1], dtype="float")
    #     noise = np.random.normal(mean, var ** 0.5, imgs[1].shape)
    #     imgs[1] += noise
    #     np.clip(imgs[1], 0, 255)
    #     imgs[1].astype("uint8")
    #     imgs[1] = Image.fromarray(imgs[1])
    trans = albumentations.augmentations.transforms.GaussNoise(var_limit=var,mean=mean,p=prob)
    imgs[0] = trans(imgs[0])
    imgs[1] = trans(imgs[1])
    return imgs
def normalize_tensor(imgs,mean_value = (0.37772245912313807, 0.4425350597897193, 0.4464795300397427),
                  std_value = (0.1762166286060892, 0.1917139949806914, 0.20443966020731438)):
    normalize = paddle.vision.transforms.Normalize(mean=mean_value, std=std_value)
    imgs[0] = normalize(paddle.to_tensor(imgs[0].transpose(2,0,1)/255,dtype=paddle.float32))
    imgs[1] = normalize(paddle.to_tensor(imgs[1].transpose(2,0,1)/255,dtype=paddle.float32))
    imgs[2] = paddle.to_tensor(imgs[2],dtype=paddle.int64)
    return imgs
class LEVIRCD_DATASET(Dataset):
    def __init__(self,
                 data_dir,
                 transforms=None,
                 shuffle=False,
                 mode = "train"):
        super(LEVIRCD_DATASET, self).__init__()
        self.transforms = transforms
        self.shuffle = shuffle
        self.data_dir = Path(data_dir)
        self.image_names_A = list((self.data_dir / "A").glob("*.png"))
        self.mode = mode

    def __getitem__(self, idx):
        image_path_A = str(self.image_names_A[idx])
        image_path_B = image_path_A.replace("A","B")
        label_path = image_path_A.replace("A","label")

        img_A = cv2.imread(image_path_A)
        img_A = cv2.cvtColor(img_A,cv2.COLOR_BGR2RGB)
        img_B = cv2.imread(image_path_B)
        img_B = cv2.cvtColor(img_B,cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path,0)
        imgs = [img_A, img_B, label]
        if self.mode == "train":
            imgs = random_flip(imgs,1)
            imgs = random_transpose(imgs,1)
            imgs = random_rotate(imgs,1)
            imgs = random_zoom(imgs,1)
            imgs = random_shiftHSV(imgs,1)
            imgs = random_gaussiannoise(imgs,1)
        imgs = normalize_tensor(imgs)
        return imgs

    def __len__(self):
        return len(self.image_names_A)

    def _binarize(self, mask, threshold=127):
        return (mask > threshold).astype('int64')

if __name__ == '__main__':
    img = Image.open(r"F:\FCCDN_paddle\Data\test\label\test_1_0_0.png")
    img = img.convert("L")
    img = F.rotate(img,-18.5)
    # img = img.convert("L")
    img = np.asarray(img)
    print(img.shape)