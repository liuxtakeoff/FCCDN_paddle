# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from paddle.vision.transforms import functional as F
from paddle.io import Dataset
from pathlib import Path
import numpy as np
import paddle
import random
import cv2

def shift_hsv_uint8(img, hue_shift, sat_shift, val_shift):
    """copy from
    https://github.com/albumentations-team/albumentations/blob/46e280f2240bfc0fc5d859a9d7cd7a5950437c0d/albumentations/augmentations/functional.py#L277"""
    dtype = img.dtype
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(img)

    if hue_shift != 0:
        lut_hue = np.arange(0, 256, dtype=np.int16)
        lut_hue = np.mod(lut_hue + hue_shift, 180).astype(dtype)
        hue = cv2.LUT(hue, lut_hue)

    if sat_shift != 0:
        lut_sat = np.arange(0, 256, dtype=np.int16)
        lut_sat = np.clip(lut_sat + sat_shift, 0, 255).astype(dtype)
        sat = cv2.LUT(sat, lut_sat)

    if val_shift != 0:
        lut_val = np.arange(0, 256, dtype=np.int16)
        lut_val = np.clip(lut_val + val_shift, 0, 255).astype(dtype)
        val = cv2.LUT(val, lut_val)

    img = cv2.merge((hue, sat, val)).astype(dtype)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img

def GaussNoise(img,var_limit=(10.0,50.0),mean=0):
    var = random.uniform(var_limit[0],var_limit[1])
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,img.shape)
    img = img.astype("float32")
    return img + gauss

def random_flip(imgs,prob = 0.5):
    if random.random() <= prob:
        _r = random.random()
        if _r < 0.666667:
            imgs[0] = F.hflip(imgs[0])
            imgs[1] = F.hflip(imgs[1])
            imgs[2] = F.hflip(imgs[2])
        if _r >= 0.333333 :
            imgs[0] = F.vflip(imgs[0])
            imgs[1] = F.vflip(imgs[1])
            imgs[2] = F.vflip(imgs[2])
    return imgs

def random_transpose(imgs,prob=0.5):
    if random.random() <= prob:
        imgs[0] = imgs[0].transpose((1,0,2))
        imgs[1] = imgs[1].transpose((1,0,2))
        imgs[2] = imgs[2].transpose((1,0))
    return imgs

def random_rotateandscale(imgs,prob=0.3,degs=(-45,45),scale_limit=0.1):
    if random.random() <= prob:
        deg = random.uniform(degs[0],degs[1])
        scale = random.uniform(1-scale_limit,1+scale_limit)
        origin_H, origin_W = imgs[0].shape[0], imgs[0].shape[1]
        new_H,new_W = int(origin_H*3*scale),int(origin_W*3*scale)
        # 先填充一圈
        imgs[0] = cv2.copyMakeBorder(imgs[0], origin_H, origin_H, origin_W, origin_W, cv2.BORDER_REFLECT_101)
        imgs[1] = cv2.copyMakeBorder(imgs[1], origin_H, origin_H, origin_W, origin_W, cv2.BORDER_REFLECT_101)
        imgs[2] = cv2.copyMakeBorder(imgs[2], origin_H, origin_H, origin_W, origin_W, cv2.BORDER_REFLECT_101)
        #进行zoom in/out
        imgs[0] = cv2.resize(imgs[0], (new_H, new_W))
        imgs[1] = cv2.resize(imgs[1], (new_H, new_W))
        imgs[2] = cv2.resize(imgs[2], (new_H, new_W))
        #进行旋转
        imgs[0] = F.rotate(imgs[0], angle=deg)
        imgs[1] = F.rotate(imgs[1], angle=deg)
        imgs[2] = F.rotate(imgs[2], angle=deg)
        #中心截取
        imgs[0] = F.center_crop(imgs[0], (origin_H, origin_W))
        imgs[1] = F.center_crop(imgs[1], (origin_H, origin_W))
        imgs[2] = F.center_crop(np.expand_dims(imgs[2], axis=-1), (origin_H, origin_W))
        imgs[2] = np.squeeze(imgs[2],axis=2)
    return imgs

def random_shiftHSV(imgs,prob=0.3,H_shift=10,S_shift=5,V_shift=10):
    # TODO:测试哪种增强方式更好：分别变换，同步变换
    if random.random() <= prob:
        h = random.randint(-H_shift,H_shift)
        v = random.randint(-V_shift,V_shift)
        s = random.randint(-S_shift,S_shift)
        imgs[0] = shift_hsv_uint8(imgs[0],hue_shift=h,sat_shift=s,val_shift=v)
    if random.random() <= prob:
        h = random.randint(-H_shift, H_shift)
        v = random.randint(-V_shift, V_shift)
        s = random.randint(-S_shift, S_shift)
        imgs[1] = shift_hsv_uint8(imgs[1],hue_shift=h,sat_shift=s,val_shift=v)
    return imgs

def random_gaussiannoise(imgs,prob=0.3,mean=0,var=(10.0,50.0)):
    # TODO:测试哪种增强方式更好：分别变换，同步变换
    if random.random() <= prob:
        imgs[0] = GaussNoise(imgs[0],var,mean)
    if random.random() <= prob:
        imgs[1] = GaussNoise(imgs[1],var,mean)
    return imgs
def random_exchange(imgs,prob=0.5):
    if random.random() <= prob:
        imgs[0],imgs[1] = imgs[1],imgs[0]
    return imgs
def normalize_tensor(imgs):
    normalize = paddle.vision.transforms.Normalize(mean=(0.37772245912313807, 0.4425350597897193, 0.4464795300397427),
                                                   std=(0.1762166286060892, 0.1917139949806914, 0.20443966020731438))
    imgs[0] = normalize(paddle.to_tensor(imgs[0].transpose(2,0,1)/255,dtype=paddle.float32))
    imgs[1] = normalize(paddle.to_tensor(imgs[1].transpose(2,0,1)/255,dtype=paddle.float32))
    imgs[2] = paddle.to_tensor(imgs[2],dtype=paddle.int64)
    imgs[3] = paddle.to_tensor(imgs[3], dtype=paddle.int64)
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
        # img_A = cv2.cvtColor(img_A,cv2.COLOR_BGR2RGB)
        img_B = cv2.imread(image_path_B)
        # img_B = cv2.cvtColor(img_B,cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path,0)
        imgs = [img_A, img_B, label]
        if self.mode == "train":
            imgs = random_flip(imgs,prob=0.5)
            imgs = random_transpose(imgs,prob=0.5)
            imgs = random_rotateandscale(imgs,prob=0.3,degs=(-45,45),scale_limit=0.1)
            imgs = random_shiftHSV(imgs,prob=0.3,H_shift=10,S_shift=5,V_shift=10)
            imgs = random_gaussiannoise(imgs,prob=0.3,mean=0,var=(10.0,50.0))
            imgs = random_exchange(imgs,prob=0.5)
        #添加尺寸与输出相同的label,用于计算损失
        _,imgs[2] = cv2.threshold(imgs[2],0,1,cv2.THRESH_BINARY)
        _h,_w = imgs[2].shape[0],imgs[2].shape[1]
        label_down2x = cv2.resize(imgs[2],(_h//2,_w//2))
        imgs.append(label_down2x)
        imgs = normalize_tensor(imgs)
        return imgs

    def __len__(self):
        return len(self.image_names_A)
