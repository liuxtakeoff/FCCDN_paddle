import cv2
import torch
import paddle
import transform_wt
from torchvision import transforms as T
from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger
from networks.FCCDN import FCCDN as FCCDN_torch
from models.FCCDN import FCCDN as FCCDN_paddle
import numpy as np
import random

seed = 2333
paddle.seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
mean_value = [0.37772245912313807, 0.4425350597897193, 0.4464795300397427]
std_value = [0.1762166286060892, 0.1917139949806914, 0.20443966020731438]
normalize_torch = T.Normalize(mean=mean_value, std=std_value)
normalize_paddle = paddle.vision.transforms.Normalize(mean=mean_value, std=std_value)
def check_model():
    """
    检查模型前向转播是否对齐，并返回对齐的模型
    """
    print("======start check model...=============")
    # write log
    reprod_log_1 = ReprodLogger()
    reprod_log_2 = ReprodLogger()
    data_1 = np.random.rand(1,3,512,512).astype(np.float32)
    data_2 = np.random.rand(1,3,512,512).astype(np.float32)
    datap_1 = paddle.to_tensor(data_1, dtype=paddle.float32)
    datap_2 = paddle.to_tensor(data_2, dtype=paddle.float32)
    datat_1 = torch.Tensor(data_1)
    datat_2 = torch.Tensor(data_2)

    # data_1 = cv2.imread("images/A/test_1.png")
    # data_2 = cv2.imread("images/B/test_1.png")
    # datap_1 = normalize_paddle(paddle.to_tensor(data_1.transpose(2,0,1)/255,dtype=paddle.float32))[None]
    # datap_2 = normalize_paddle(paddle.to_tensor(data_2.transpose(2,0,1)/255,dtype=paddle.float32))[None]
    # datat_1 = normalize_torch(torch.Tensor(data_1.transpose(2,0,1)/255))[None]
    # datat_2 = normalize_torch(torch.Tensor(data_2.transpose(2,0,1)/255))[None]
    wt_torch = "wt_torch.pth"
    wt_paddle = "wt_paddle.pdparams"
    # wt_paddle_dict = paddle.load(wt_paddle)
    # print(wt_paddle_dict)
    model_paddle = FCCDN_paddle(num_band=3, use_se=True)
    model_torch = FCCDN_torch(num_band=3, use_se=True)
    model_torch.load_state_dict(torch.load(wt_torch))
    model_paddle.load_dict(paddle.load(wt_paddle))
    model_paddle.eval()
    model_torch.eval()

    datat = model_torch([datat_1,datat_2])[0][0]
    datap = model_paddle([datap_1,datap_2])[0][0]
    print(datap.shape)
    print(datap.shape)
    reprod_log_1.add("result_model", datap.cpu().detach().numpy())
    reprod_log_1.save("diff_log/result_model_paddle.npy")

    reprod_log_2.add("result_model", datat.cpu().detach().numpy())
    reprod_log_2.save("diff_log/result_model_torch.npy")

    # check_diff
    diff_helper = ReprodDiffHelper()

    info1 = diff_helper.load_info("diff_log/result_model_paddle.npy")
    info2 = diff_helper.load_info("diff_log/result_model_torch.npy")

    diff_helper.compare_info(info1, info2)

    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path="diff_log/diff_model.txt")
    return model_paddle, model_torch

def check_loss():
    """
    检查损失函数，并返回两个损失函数
    """
    print("======start check loss...=============")
    loss_pp = paddle.nn.CrossEntropyLoss()
    loss_torch = torch.nn.CrossEntropyLoss()
    # write log
    reprod_log_1 = ReprodLogger()
    reprod_log_2 = ReprodLogger()
    data_1 = np.random.rand(96, 3).astype(np.float32) #随机生成输出数据
    data_2 = np.random.randint(low=0,high=2,size=96).astype(np.int64) #随机生成标签数据
    datap = paddle.to_tensor(data_1,place=paddle.CUDAPlace(0))
    datat = torch.tensor(data_1)
    labelp = paddle.to_tensor(data_2,place=paddle.CUDAPlace(0))
    labelt = torch.tensor(data_2)

    lossp = loss_pp(datap,labelp)
    losst = loss_torch(datat,labelt)

    # reprod_log_1.add("demo_test_1", data_1)
    reprod_log_1.add("result_loss", lossp.cpu().detach().numpy())
    reprod_log_1.save("diff_log/result_loss_paddle.npy")

    # reprod_log_2.add("demo_test_1", data_1)
    reprod_log_2.add("result_loss", losst.cpu().detach().numpy())
    reprod_log_2.save("diff_log/result_loss_torch.npy")

    # check_diff
    diff_helper = ReprodDiffHelper()

    info1 = diff_helper.load_info("diff_log/result_loss_paddle.npy")
    info2 = diff_helper.load_info("diff_log/result_loss_torch.npy")

    diff_helper.compare_info(info1, info2)

    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path="diff_log/diff_loss.txt")
    return loss_pp,loss_torch


def check_optim(model_pp, model_torch, test=True):
    """
    检查优化器（学习率是否一致），并返回两个优化器和调度器
    """
    print("======start check optim...=============")
    # 定义超参数
    learning_rate = 3e-2
    weight_decay = 0.00003
    momentum = 0.9
    epochs = 100
    # 定义优化器及学习率时间表
    optim_torch = torch.optim.SGD(model_torch.parameters(), lr=learning_rate, momentum=momentum,
                                  weight_decay=weight_decay)
    scheduler_torch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_torch, epochs)

    scheduler_pp = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=learning_rate, T_max=epochs)
    optim_pp = paddle.optimizer.Momentum(parameters=model_pp.parameters(), learning_rate=scheduler_pp,
                                         momentum=momentum, weight_decay=weight_decay)

    if test:
        # write log
        reprod_log_1 = ReprodLogger()
        reprod_log_2 = ReprodLogger()
        lr_pps = []
        lr_torchs = []
        for step in range(epochs):
            scheduler_torch.step()
            scheduler_pp.step()
            lr_pp = optim_pp.get_lr()
            lr_pps.append(lr_pp)
            lr_torch = optim_torch.param_groups[0]["lr"]
            lr_torchs.append(lr_torch)
        lr_pps = np.array(lr_pps)
        lr_torchs = np.array(lr_torchs)

        reprod_log_1.add("result_lr", lr_pps)
        reprod_log_1.save("diff_log/result_lr_paddle.npy")

        reprod_log_2.add("result_lr", lr_torchs)
        reprod_log_2.save("diff_log/result_lr_torch.npy")

        # check_diff
        diff_helper = ReprodDiffHelper()

        info1 = diff_helper.load_info("diff_log/result_lr_paddle.npy")
        info2 = diff_helper.load_info("diff_log/result_lr_torch.npy")

        diff_helper.compare_info(info1, info2)

        diff_helper.report(
            diff_method="mean", diff_threshold=1e-6, path="diff_log/diff_LearningRate.txt")
    else:
        return optim_pp, optim_torch, scheduler_pp, scheduler_torch
    from networks.utils import double_conv as dc_torch
    from models.utils import double_conv as dc_paddle

if __name__ == '__main__':
    model_paddle,model_torch = check_model()





    # wt_torch_path = "conv_torch.pth"
    # wt_paddle_path = "conv_paddle.pdparams"
    # # conv_torch = dc_torch(32, 64)
    # conv_torch = torch.nn.Conv2d(32, 32,3,stride=1,dilation=1,padding=1,padding_mode="zeros")
    # # conv_paddle = dc_paddle(32,64)
    # conv_paddle = paddle.nn.Conv2D(32, 32,3,stride=1,dilation=1,padding=1,padding_mode="zeros")
    # torch.save(conv_torch.state_dict(),wt_torch_path)
    # transform_wt.torch2paddle(wt_torch_path,wt_paddle_path)
    #
    # # conv_torch.load_state_dict(torch.load(wt_torch_path))
    # conv_paddle.set_dict(paddle.load(wt_paddle_path))
    # # write log
    # reprod_log_1 = ReprodLogger()
    # reprod_log_2 = ReprodLogger()
    # data = np.random.rand(1,32,512,512).astype(np.float32)
    # data_t = torch.Tensor(data)
    # data_p = paddle.Tensor(data)
    #
    # datat = conv_torch(data_t)
    # datap = conv_paddle(data_p)
    # print(datat[0,2,2,2].cpu().detach().numpy(),datap[0,2,2,2])
    # print(datat.size(),datap.shape)
    #
    # reprod_log_1.add("result_doubleconv", datap.cpu().detach().numpy())
    # reprod_log_1.save("diff_log/result_doubleconv_paddle.npy")
    #
    # reprod_log_2.add("result_doubleconv", datat.cpu().detach().numpy())
    # reprod_log_2.save("diff_log/result_doubleconv_torch.npy")
    #
    # # check_diff
    # diff_helper = ReprodDiffHelper()
    #
    # info1 = diff_helper.load_info("diff_log/result_doubleconv_paddle.npy")
    # info2 = diff_helper.load_info("diff_log/result_doubleconv_torch.npy")
    #
    # diff_helper.compare_info(info1, info2)
    #
    # diff_helper.report(
    #     diff_method="mean", diff_threshold=1e-6, path="diff_log/diff_conv.txt")