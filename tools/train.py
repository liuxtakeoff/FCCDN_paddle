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
from tools.loss import FCCDN_loss_BCD
from tools.eval import eval_model
from paddle.io import DataLoader
import matplotlib.pyplot as plt
from models.FCCDN import FCCDN
from paddle import optimizer
import numpy as np
import argparse
import random
import paddle
import time
seed = 0
np.random.seed(seed)
paddle.seed(seed)
random.seed(seed)

def test_dataset(args):
    #加载并测试数据集,看看数据增强能不能正常运行
    data_test_dir = os.path.join(args.data_dir,"train")
    dataset_test = LEVIRCD_DATASET(data_dir=data_test_dir,mode="train")
    dataloader_test = DataLoader(dataset_test,
                                 batch_size=2,
                                 drop_last=True,
                                 shuffle=False,
                                 num_workers=0,
                                 )
    for ind,batch in enumerate(dataloader_test):
        print(len(batch)) #[bs*imgA bs*imgB bs*label]
        imgA,imgB,label = batch[0][0],batch[1][0],batch[2][0]

        imgA = imgA.cpu().detach().numpy()
        imgA = imgA.transpose(1,2,0)
        imgB = imgB.cpu().detach().numpy()
        imgB = imgB.transpose(1, 2, 0)
        label = label.cpu().detach().numpy()
        plt.figure(0)
        plt.subplot(1,2,1)
        plt.title("imgA")
        plt.imshow(imgA)
        plt.subplot(1, 2, 2)
        plt.title("imgB")
        plt.imshow(imgB)
        plt.figure(1)
        plt.title("label")
        plt.imshow(label*255)
        plt.show()
        print(imgA.shape)
        print(label.shape)



if __name__ == '__main__':
    def str2bool(v):
        return v.lower() in ("true", "t", "1")
    parser = argparse.ArgumentParser(description='Training change detection as described in the FCCDN Paper.')
    #训练参数
    parser.add_argument('--cuda', default=True,type=str2bool,help='use cuda for training (default: False)')
    parser.add_argument('--pretrained', default=None, help='use pretrained model for training (default: None)')
    parser.add_argument('--data_dir', default="Data", help='data dir for training (default: Data)')
    parser.add_argument('--batch_size', default=16,type=int,help='batch size for training (default: 16 for 512 ,64 for 256)')
    parser.add_argument('--img_size', default=512,type=int, help='input image size for training (default: 512 for LEVIRCD)')
    parser.add_argument('--workers', default=0, type=int, help="number of workers to use for data loading (default:0)")
    parser.add_argument('--epochs', default=99999,type=int, help="max epochs for training")
    #模型建立参数
    parser.add_argument('--num_band', default=3,type=int, help='num_band param for building model (default: 3)')
    parser.add_argument('--os', default=16,type=int, help='os param for building model (default: 16)')
    parser.add_argument('--use_se', default=True,type=str2bool, help='use se while building model (default: True)')
    #学习率调度参数
    parser.add_argument('--warmup', default=200,type=float,help='warm up iters to reach original learning rate (default:200)')
    parser.add_argument('--lr', default=0.002,type=float,help='original learning rate for training (default:0.002)')
    parser.add_argument('--lr_reduce_ratio', default=0.3,type=float,help='learning rate reduce ratio when change the learning rate(default:0.3)')
    #优化器超参数
    parser.add_argument('--weight_decay', default=0.001,type=float,help='weight_decay param for optimzer (default:0.001)')
    #结果保存参数
    parser.add_argument('--val_epoch', default=30,type=int,help='the epoch when start validation in training(default: 30)')
    parser.add_argument('--max_reduce_time', default=4,type=int,help='max lr reduce times (default: 4)')
    parser.add_argument('--min_reduce_epochs', default="80,100,140,160",help='min train epochs to follow the paper')
    parser.add_argument('--save_interval', default=10, type=int, help="number of epochs between each model save (default:10)")
    parser.add_argument('--log_interval', default=1, type=int, help="number of step between each log print (default:1)")
    parser.add_argument('--log_dir', default="logs", help="path to save outputs (default: logs)")
    parser.add_argument('--output', default=None, help="no sense")
    args = parser.parse_args()
    print(args)

    # 优化器超参数
    weight_decay = args.weight_decay

    #学习率调度超参数
    lr_reduce_ratio = args.lr_reduce_ratio  # 学习率衰减比例
    max_reduce_time = args.max_reduce_time  # 第几次调整学习率时，结束训练

    lr = args.lr            #初始化学习率变量
    min_reduce_epochs = [int(n) for n in args.min_reduce_epochs.split(",")]
    last_improve_epoch = 0  # 初始化距离上次提升的轮次数
    lr_reduce_times = 0  # 初始化学习率调整次数
    max_f1 = 0              #初始化最佳f1-score
    val_flag = False        # 初始化验证开关

    #建立保存模型的位置
    wt_save_path = os.path.join(args.log_dir, "train")
    if not os.path.exists(wt_save_path):
        os.makedirs(wt_save_path)

    #建立训练数据集
    data_train_dir = os.path.join(args.data_dir,"train")
    dataset_train = LEVIRCD_DATASET(data_dir=data_train_dir, mode="train")
    dataloader_train = DataLoader(dataset_train,
                                 batch_size=args.batch_size,
                                 drop_last=True,
                                 shuffle=True,
                                 num_workers=args.workers,
                                 persistent_workers=True, prefetch_factor=5
                                 )
    len_train = len(dataloader_train)
    args.warmup = min(len_train,args.warmup)
    warmup_lrs = np.arange(1e-7, args.lr, (args.lr - 1e-7) / args.warmup)
    # 建立验证数据集
    data_val_dir = os.path.join(args.data_dir, "val")
    dataset_val = LEVIRCD_DATASET(data_dir=data_val_dir, mode="val")
    dataloader_val = DataLoader(dataset_val,
                                  batch_size=1,
                                  drop_last=False,
                                  shuffle=False,
                                  num_workers=args.workers,
                                  persistent_workers=True, prefetch_factor=5
                                )
    #建立模型
    model = FCCDN(num_band=args.num_band,os=args.os,use_se=args.use_se)
    #初始化优化器
    optim = optimizer.AdamW(learning_rate=args.lr,
                            weight_decay=weight_decay,
                            parameters=model.parameters())
    time0 = time.time() #记录当前时间
    total_except_reader_cost = 0 #记录除了数据读取以外花的时间，从而迂回地计算数据读取时间
    print("start training!")
    with open(os.path.join(wt_save_path,"train_log.txt"),"w") as f:
        f.write("start training!\n")
    for epoch in range(1,args.epochs+1):#会早停的，实际并不会走这么多轮
        total_loss = 0
        #到达一定轮次后，开启验证环节
        if epoch == args.val_epoch:
            val_flag = True
        model.train()
        for ind,batch in enumerate(dataloader_train):
            optim.clear_grad()
            if epoch == 1 and ind < args.warmup:
                #先进行warm_up
                optim.set_lr(warmup_lrs[ind])
            else:
                optim.set_lr(lr)
            imgs_A,imgs_B,labels,labels_down2x = batch[0],batch[1],batch[2],batch[3]
            inp = paddle.concat([imgs_A.unsqueeze(0), imgs_B.unsqueeze(0)], axis=0)

            time1 = time.time()
            out = model(inp)
            #计算loss值
            loss = FCCDN_loss_BCD(out,[labels.unsqueeze(1).astype(paddle.float32),labels_down2x.unsqueeze(1).astype(paddle.float32)])
            #更新参数
            loss.backward()
            optim.step()

            total_loss += loss
            total_except_reader_cost += time.time() - time1
        with paddle.no_grad():
            avg_batch_cost = (time.time() - time0)/epoch/len_train
            total_reader_cost = (time.time() - time0) - total_except_reader_cost
            print("epoch:%d loss:%.4f avg_reader_cost:%.4f avg_batch_cost:%.3f avg_ips:%.4f lr:%.6f" % (
                epoch, total_loss, total_reader_cost / epoch / len_train, avg_batch_cost, avg_batch_cost / args.batch_size, optim.get_lr()))
            with open(os.path.join(wt_save_path,"train_log.txt"),"a") as f:
                f.write("epoch:%d loss:%.4f avg_reader_cost:%.4f avg_batch_cost:%.3f avg_ips:%.4f lr:%.6f\n" % (
                epoch, total_loss, total_reader_cost / epoch / len_train, avg_batch_cost, avg_batch_cost / args.batch_size, optim.get_lr()))
        #如果没到时候，什么也不做
        if not val_flag:
            continue
        #时机已到！
        model.eval()
        f1_score,precision,recall = eval_model(model=model,
                                                dataloader_val=dataloader_val,)
        if f1_score > max_f1:
            last_improve_epoch = 0
            max_f1 = f1_score
            #保存该模型
            paddle.save(model.state_dict(),os.path.join(wt_save_path,"best.pdparams"))
        else:
            last_improve_epoch += 1
            if last_improve_epoch == 10:
                # 已经十个轮次没有提升性能了，衰减学习率
                lr_reduce_times += 1
                #保存一次模型
                paddle.save(model.state_dict(),os.path.join(wt_save_path,"reduce%d.pdparams"%lr_reduce_times))
                if lr_reduce_times == max_reduce_time:
                    #当要第四次衰减时，结束训练
                    print("训练结束，共训练%d轮，最终验证集F1-score:%f"%(epoch,max_f1))
                    break
                #衰减学习率并重置计数器
                lr *= lr_reduce_ratio
                optim.set_lr(lr)
                last_improve_epoch = 0
        print("epoch:%d(lr_reduce：%d no_improve:%d) F1-score:%.4f(best:%.4f)" % (
        epoch, lr_reduce_times, last_improve_epoch, f1_score, max_f1))
        with open(os.path.join(wt_save_path,"train_log.txt"),"a") as f:
            f.write("epoch:%d(lr_reduce：%d no_improve:%d) F1-score:%.4f(best:%.4f)\n" % (
            epoch, lr_reduce_times, last_improve_epoch, f1_score, max_f1))


