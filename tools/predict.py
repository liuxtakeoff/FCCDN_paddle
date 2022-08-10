# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
import paddle
from paddle.vision import transforms as T
from tqdm import tqdm
import cv2
import numpy as np
from models.FCCDN import FCCDN



print("**********warning**********")
test_in_path = "images/test"
test_out_path = "images/eval"
pretrained_weights = "wt_paddle.pdparams"
mean_value = [0.37772245912313807, 0.4425350597897193, 0.4464795300397427]
std_value = [0.1762166286060892, 0.1917139949806914, 0.20443966020731438]

"make paths"
if not os.path.exists(test_out_path):
    os.makedirs(test_out_path)
if not os.path.exists(os.path.join(test_out_path, "change")):
    os.makedirs(os.path.join(test_out_path, "change"))
if not os.path.exists(os.path.join(test_out_path, "seg1")):
    os.makedirs(os.path.join(test_out_path, "seg1"))
if not os.path.exists(os.path.join(test_out_path, "seg2")):
    os.makedirs(os.path.join(test_out_path, "seg2"))

basename_list = []
files = os.listdir(os.path.join(test_in_path, "A"))
for file in files:
    if file[-3:] == "png":
        basename_list.append(file)

model = FCCDN(num_band=3, use_se=True)
# pretrained_dict = paddle.load(pretrained_weights, map_location="cpu")
# module_model_state_dict = {}
# for item, value in pretrained_dict['model_state_dict'].items():
#     if item[0:7] == 'module.':
#         item = item[7:]
#     module_model_state_dict[item] = value
# model.load_state_dict(module_model_state_dict, strict=True)
# model.cuda()
model_dict = paddle.load(pretrained_weights)
model.set_dict(model_dict)
model.eval()
# paddle.save(model.state_dict(),"wt_paddle.pth")
normalize = T.Normalize(mean=mean_value, std=std_value)

"""This is a simple inference code. Users can speed up the inference with paddle.utils.data.DataLoader"""
with tqdm(total=len(basename_list)) as pbar:
    pbar.set_description("Test")
    with paddle.no_grad():
        for basename in basename_list:
            pre = cv2.imread(os.path.join(test_in_path, "A", basename))
            post = cv2.imread(os.path.join(test_in_path, "B", basename))
            pre = normalize(paddle.to_tensor(pre.transpose(2,0,1)/255,dtype=paddle.float32))[None]
            post = normalize(paddle.to_tensor(post.transpose(2,0,1)/255,dtype=paddle.float32))[None]
            print(post)
            # pre = pre[0].cpu().detach().numpy().transpose(1,2,0)
            # post = post[0].cpu().detach().numpy().transpose(1,2,0)
            # plt.figure(0)
            # plt.subplot(1, 2, 1)
            # plt.title("imgA")
            # plt.imshow(pre)
            # plt.subplot(1, 2, 2)
            # plt.title("imgB")
            # plt.imshow(post)
            # # plt.figure(1)
            # # plt.title("label")
            # # plt.imshow(label)
            # plt.show()
            pre = pre.unsqueeze(1)
            post = post.unsqueeze(1)
            imgs = paddle.concat([pre,post])
            pred = model(imgs)
            # change_mask = paddle.sigmoid(pred[0]).cpu().numpy()[0,0]
            out = paddle.round(paddle.nn.functional.sigmoid(pred[0])).cpu().numpy()
            seg1 = paddle.round(paddle.nn.functional.sigmoid(pred[1])).cpu().numpy()
            seg2 = paddle.round(paddle.nn.functional.sigmoid(pred[2])).cpu().numpy()

            cv2.imwrite(os.path.join(test_out_path, "change", basename), out[0,0]*255)
            cv2.imwrite(os.path.join(test_out_path, "seg1", basename), seg1[0,0]*255)
            cv2.imwrite(os.path.join(test_out_path, "seg2", basename), seg2[0,0]*255)
            pbar.update()
