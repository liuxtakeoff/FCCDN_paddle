import paddle
import paddle.nn as nn
# from models.sync_batchnorm import SynchronizedBatchNorm2d
bn_mom = 0.0003
"""Implemention of non-local feature pyramid network"""


class NL_Block(nn.Layer):
    def __init__(self, in_channels):
        super(NL_Block, self).__init__()
        self.conv_v = nn.Sequential(
            nn.Conv2D(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            # SynchronizedBatchNorm2d(in_channels),
            nn.BatchNorm2D(in_channels,momentum=0.1),
        )
        self.W = nn.Sequential(
            nn.Conv2D(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            # SynchronizedBatchNorm2d(in_channels),
            nn.BatchNorm2D(in_channels, momentum=0.1),
            nn.ReLU()
        )

    def forward(self, x):
        batch_size, c, h, w = x.shape[0],x.shape[1], x.shape[2], x.shape[3]
        value = self.conv_v(x)
        value =value.reshape([batch_size, c, value.shape[2]*value.shape[3]])
        value = value.transpose([0, 2, 1])                                             #B * (H*W) * value_channels
        key = x.reshape([batch_size, c, h*w])                #B * key_channels * (H*W)
        query = x.reshape([batch_size, c,h*w])
        query = query.transpose([0, 2, 1])
        sim_map = paddle.matmul(query, key)                                         #B * (H*W) * (H*W)
        sim_map = (c**-.5) * sim_map                               #B * (H*W) * (H*W)
        sim_map = paddle.nn.functional.softmax(sim_map, axis=-1)                    #B * (H*W) * (H*W)
        context = paddle.matmul(sim_map, value)
        #TODO:how to contiguous() in paddle
        # context = context.transpose([0, 2, 1]).contiguous()
        context = context.transpose([0, 2, 1])
        context = context.reshape([batch_size, c, *x.shape[2:]])
        context = self.W(context)
        
        return context


class NL_FPN(nn.Layer):
    """ non-local feature parymid network"""
    def __init__(self, in_dim, reduction=True):
        super(NL_FPN, self).__init__()
        if reduction:
            self.reduction = nn.Sequential(
                        nn.Conv2D(in_dim, in_dim//4, kernel_size=1, stride=1, padding=0),
                        # SynchronizedBatchNorm2d(in_dim//4, momentum=bn_mom),
                        nn.BatchNorm2D(in_dim//4, momentum=bn_mom),
                        nn.ReLU(),
                        )
            self.re_reduction = nn.Sequential(
                        nn.Conv2D(in_dim//4, in_dim, kernel_size=1, stride=1, padding=0),
                        # SynchronizedBatchNorm2d(in_dim, momentum=bn_mom),
                        nn.BatchNorm2D(in_dim, momentum=bn_mom),
                        nn.ReLU(),
                        )
            in_dim = in_dim//4
        else:
            self.reduction = None
            self.re_reduction = None
        self.conv_e1 = nn.Sequential(
                        nn.Conv2D(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                        # SynchronizedBatchNorm2d(in_dim, momentum=bn_mom),
                        nn.BatchNorm2D(in_dim, momentum=bn_mom),
                        nn.ReLU(),
                        )
        self.conv_e2 = nn.Sequential(
                        nn.Conv2D(in_dim, in_dim*2,kernel_size=3, stride=1, padding=1),
                        # SynchronizedBatchNorm2d(in_dim*2, momentum=bn_mom),
                        nn.BatchNorm2D(in_dim*2, momentum=bn_mom),
                        nn.ReLU(),
                        )
        self.conv_e3 = nn.Sequential(
                        nn.Conv2D(in_dim*2,in_dim*4,kernel_size=3, stride=1, padding=1),
                        # SynchronizedBatchNorm2d(in_dim*4, momentum=bn_mom),
                        nn.BatchNorm2D(in_dim*4, momentum=bn_mom),
                        nn.ReLU(),
                        )
        self.conv_d1 = nn.Sequential(
                        nn.Conv2D(in_dim,in_dim,kernel_size=3, stride=1, padding=1),
                        # SynchronizedBatchNorm2d(in_dim, momentum=bn_mom),
                        nn.BatchNorm2D(in_dim, momentum=bn_mom),
                        nn.ReLU(),
                        )
        self.conv_d2 = nn.Sequential(
                        nn.Conv2D(in_dim*2,in_dim,kernel_size=3, stride=1, padding=1),
                        # SynchronizedBatchNorm2d(in_dim, momentum=bn_mom),
                        nn.BatchNorm2D(in_dim, momentum=bn_mom),
                        nn.ReLU(),
                        )
        self.conv_d3 = nn.Sequential(
                        nn.Conv2D(in_dim*4,in_dim*2,kernel_size=3, stride=1, padding=1),
                        # SynchronizedBatchNorm2d(in_dim*2, momentum=bn_mom),
                        nn.BatchNorm2D(in_dim*2, momentum=bn_mom),
                        nn.ReLU(),
                        )
        self.nl3 = NL_Block(in_dim*2)
        self.nl2 = NL_Block(in_dim)
        self.nl1 = NL_Block(in_dim)

        self.downsample_x2 = nn.MaxPool2D(stride=2, kernel_size=2)
        self.upsample_x2 = nn.UpsamplingBilinear2D(scale_factor=2)

    def forward(self, x):
        if self.reduction is not None:
            x = self.reduction(x)
        e1 = self.conv_e1(x)                            #C,H,W
        e2 = self.conv_e2(self.downsample_x2(e1))       #2C,H/2,W/2
        e3 = self.conv_e3(self.downsample_x2(e2))       #4C,H/4,W/4
        
        d3 = self.conv_d3(e3)                           #2C,H/4,W/4
        nl = self.nl3(d3)
        d3 = self.upsample_x2(paddle.multiply(d3, nl))                  ##2C,H/2,W/2
        d2 = self.conv_d2(e2+d3)                        #C,H/2,W/2
        nl = self.nl2(d2)
        d2 = self.upsample_x2(paddle.multiply(d2, nl))                 #C,H,W
        d1 = self.conv_d1(e1+d2)
        nl = self.nl1(d1)
        d1 = paddle.multiply(d1, nl)          #C,H,W
        if self.re_reduction is not None:
            d1 = self.re_reduction(d1)

        return d1