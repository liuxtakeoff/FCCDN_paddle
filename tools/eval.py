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
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
from tools.LEVIRCD import LEVIRCD_DATASET
from paddle.io import DataLoader
from models.FCCDN import FCCDN
from tqdm import tqdm
import paddle
import argparse
import cv2

def eval_model(model,dataloader_val,test_out_path=None,save_plot=False):
    with paddle.no_grad():
        model.eval()
        m_precision = paddle.metric.Precision()
        m_recall = paddle.metric.Recall()
        count = 0
        for batch in tqdm(dataloader_val):
            img_A,img_B,label = batch[0],batch[1],batch[2]
            inp = paddle.concat([img_A.unsqueeze(0), img_B.unsqueeze(0)], axis=0)
            out = model(inp)
            if save_plot:
                #输出看看到底预测正确咩
                change_mask = paddle.round(paddle.nn.functional.sigmoid(out[0])).cpu().numpy()
                seg1 = paddle.round(paddle.nn.functional.sigmoid(out[1])).cpu().numpy()
                seg2 = paddle.round(paddle.nn.functional.sigmoid(out[2])).cpu().numpy()
                cv2.imwrite(os.path.join(test_out_path, "change", "%d.png"%count), change_mask[0,0]*255)
                cv2.imwrite(os.path.join(test_out_path, "seg1", "%d.png"%count), seg1[0,0]*255)
                cv2.imwrite(os.path.join(test_out_path, "seg2", "%d.png"%count), seg2[0,0]*255)
                count+=1
            pre_change = paddle.flatten(out[0])
            label = paddle.flatten(label)
            pre_change = paddle.round(paddle.nn.functional.sigmoid(pre_change)).astype(paddle.int32)
            m_precision.update(pre_change,label)
            m_recall.update(pre_change,label)
        precision = m_precision.accumulate()
        recall = m_recall.accumulate()
        f1 = 2*(precision*recall)/(precision+recall+1e-8)
        return f1,precision,recall
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evalution change detection as described in the FCCDN Paper.')
    #定义验证参数
    parser.add_argument('--cuda', default=True, help='use cuda for training (default: False)')
    parser.add_argument('--pretrained', default=None, help='use pretrained model for training (default: None)')
    parser.add_argument('--workers', default=0, type=int, help="number of workers to use for data loading (default:0)")
    parser.add_argument('--save_plot', default=False, help="save result pictures while eval (default:False)")
    parser.add_argument('--batch_size', default=1, help="batch_size while eval (default:1)")
    #定义读取及保存地址
    parser.add_argument('--data_dir', default="Data/test", help='data dir for training (default: Data/test)')
    parser.add_argument('--out_dir', default="logs/eval", help='data dir for training (default: logs/eval)')
    parser.add_argument('--pretrained_model', default="logs/train/best.pdparams", help='pretrained model for evalution')
    # 定义模型参数
    parser.add_argument('--num_band', default=3, type=int, help='num_band param for building model (default: 3)')
    parser.add_argument('--os', default=16, type=int, help='os param for building model (default: 16)')
    parser.add_argument('--use_se', default=True, help='use se while building model (default: True)')
    args = parser.parse_args()

    args.save_plot = True if args.save_plot in ["True","true","1",True] else False
    args.cuda = True if args.cuda in ["True","true","1",True] else False
    args.use_se = True if args.use_se in ["True","true","1",True] else False
    print(args)

    test_in_path = args.data_dir                #定义测试集地址
    test_out_path = args.out_dir                #定义测试结果输出地址
    pretrained_weights = args.pretrained_model  #定义预训练权重地址

    #定义模型参数
    num_band = args.num_band
    use_se = args.use_se
    model_os = args.os

    #make paths
    if not os.path.exists(test_out_path):
        os.makedirs(test_out_path)
    if args.save_plot:
        args.batch_size = 1
        if not os.path.exists(os.path.join(test_out_path, "change")):
            os.makedirs(os.path.join(test_out_path, "change"))
        if not os.path.exists(os.path.join(test_out_path, "seg1")):
            os.makedirs(os.path.join(test_out_path, "seg1"))
        if not os.path.exists(os.path.join(test_out_path, "seg2")):
            os.makedirs(os.path.join(test_out_path, "seg2"))
    #载入模型
    model = FCCDN(num_band=num_band,os=model_os, use_se=use_se)
    model_dict = paddle.load(pretrained_weights)
    model.set_dict(model_dict)
    print("load model:%s"%pretrained_weights)
    model.eval()

    dataset_test = LEVIRCD_DATASET(data_dir=test_in_path,mode="test")
    loader_test = DataLoader(dataset=dataset_test,
                             batch_size=args.batch_size,
                             num_workers=args.workers,
                             shuffle=False,
                             )
    f1_score,precision,recall = eval_model(model,loader_test,test_out_path,args.save_plot)
    print("F1-score:%f precision:%f recall:%f"%(f1_score,precision,recall))
    with open(os.path.join(test_out_path,"eval_result.txt"),"w") as f:
        f.write("F1-score:%f precision:%f recall:%f"%(f1_score,precision,recall))
