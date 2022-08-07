import argparse
import os

from tools.LEVIRCD import LEVIRCD_DATASET
from paddle.io import DataLoader
import matplotlib.pyplot as plt



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training change detection as described in the FCCDN Paper.')
    parser.add_argument('--cuda', default=True,type=bool,help='use cuda for training (default: False)')
    parser.add_argument('--pretrained', default=None, help='use pretrained model for training (default: None)')
    parser.add_argument('--data_dir', default="Data", help='data dir for training (default: Data)')
    parser.add_argument('--batch_size', default=16,type=int,help='batch size for training (default: 16 for 512 ,64 for 256)')
    parser.add_argument('--img_size', default=512,type=int, help='input image size for training (default: 512 for LEVIRCD)')
    parser.add_argument('--lr', default=0.002,type=float,help='original learning rate for training (default:0.002)')
    parser.add_argument('--val_epoch', default=30,type=int,help='the epoch when start validation in training(default: 30)')
    parser.add_argument('--workers', default=0, type=int, help="number of workers to use for data loading (default:8)")
    parser.add_argument('--save_interval', default=10, type=int, help="number of epochs between each model save (default:10)")
    parser.add_argument('--log_interval', default=1, type=int, help="number of step between each log print (default:1)")
    parser.add_argument('--log_dir', default="logs", help="path to save outputs (default: logs)")
    args = parser.parse_args()

    #加载数据集
    data_test_dir = os.path.join(args.data_dir,"test")
    dataset_test = LEVIRCD_DATASET(data_dir=data_test_dir)
    dataloader_test = DataLoader(dataset_test,
                                 batch_size=1,
                                 drop_last=True,
                                 shuffle=True,
                                 num_workers=0,
                                 )
    for ind,batch in enumerate(dataloader_test):
        print(len(batch))
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
        plt.imshow(label)
        plt.show()

        print(imgA.shape)
        print(label.shape)