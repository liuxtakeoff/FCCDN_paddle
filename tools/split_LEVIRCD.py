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
import os.path as osp
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
import PIL.Image as Image
from tqdm import tqdm
import argparse

def split_data(image_path, block_size, save_path,overlap_ratio=0.5):
    img = Image.open(image_path)
    H,W = img.size[0],img.size[1]
    rows = int(H/block_size/(1-overlap_ratio)) - 1
    cols = int(W/block_size/(1-overlap_ratio)) - 1
    total_number = rows*cols
    for r in range(rows):
        for c in range(cols):
            loc_start = (c * (1 - overlap_ratio) * block_size, r * (1 - overlap_ratio) * block_size)
            _img = img.crop((loc_start[0],loc_start[1],loc_start[0]+block_size,loc_start[1]+block_size))
            _save_path = save_path+"_%d_%d.png"%(r,c)
            _img.save(_save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split LEVIR-CD as described in the FCCDN Paper.')
    # 定义读取及保存地址
    parser.add_argument('--data_dir', default="LEVIR-CD", help='origin LEVIR-CD dataset dir (default: LEVIR-CD)')
    parser.add_argument('--out_dir', default="Data", help='data dir for training (default: Data)')
    # 定义分割参数
    parser.add_argument('--split_size', default=512, type=int, help='img size after split (default: 512)')
    parser.add_argument('--overlap_ratio', default=0.5, type=float, help='overlap ratio while split (default: 0.5)')
    args = parser.parse_args()
    print(args)

    img_root = args.data_dir    #定义原始数据集地址
    img_saveroot = args.out_dir #定义分割数据集地址

    with tqdm(total=(445+64+128)*3) as pbar:
        for _dir in ["test","val","train"]:
            for _type in ["A","B","label"]:
                img_dir = osp.join(img_root,_dir,_type)
                img_savedir = osp.join(img_saveroot,_dir,_type)
                if not osp.exists(img_savedir):
                    os.makedirs(img_savedir)
                imgnames = os.listdir(img_dir)
                #测试集不需要分割，直接复制粘贴即可
                if _dir in ["test"]:
                    for imgname in imgnames:
                        imgpath = osp.join(img_dir, imgname)
                        imgsavepath = osp.join(img_savedir, imgname)
                        img = Image.open(imgpath)
                        img.save(imgsavepath)
                        pbar.update(1)
                else:
                    for imgname in imgnames:
                        imgpath = osp.join(img_dir,imgname)
                        imgsavepath = osp.join(img_savedir,imgname[:-4])
                        split_data(imgpath,args.split_size,imgsavepath,args.overlap_ratio)
                        pbar.update(1)

