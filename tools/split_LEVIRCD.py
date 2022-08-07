import os.path as osp
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
from tqdm import tqdm
import math
import PIL.Image as Image
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
    # img_path = r"F:\FCCDN_paddle\images\A\test_1.png"
    # save_path = r"F:\FCCDN_paddle\images\A_512\test_1"
    # split_data(img_path,512,save_path,0.5)
    img_root = "LEVIR-CD"
    img_saveroot = "Data"
    with tqdm(total=(445+64+128)*3) as pbar:
        for _dir in ["test","val","train"]:
            # dataset_dir = osp.join(img_root,_dir)
            # dataset_savedir = osp.join(img_saveroot,_dir)
            for _type in ["A","B","label"]:
                img_dir = osp.join(img_root,_dir,_type)
                img_savedir = osp.join(img_saveroot,_dir,_type)
                if not osp.exists(img_savedir):
                    os.makedirs(img_savedir)
                imgnames = os.listdir(img_dir)
                for imgname in imgnames:
                    imgpath = osp.join(img_dir,imgname)
                    imgsavepath = osp.join(img_savedir,imgname[:-4])
                    split_data(imgpath,512,imgsavepath,0.5)
                    pbar.update(1)

