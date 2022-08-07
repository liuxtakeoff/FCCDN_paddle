import paddle
import paddle.nn as nn
# from models.sync_batchnorm import SynchronizedBatchNorm2d
bn_mom = 0.0003
"""Implemention of dense fusion module"""

class densecat_cat_add(nn.Layer):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_add, self).__init__()

        self.conv1 = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_chn, in_chn, kernel_size=3, padding=1),
            paddle.nn.ReLU(),
        )
        self.conv2 = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_chn, in_chn, kernel_size=3, padding=1),
            paddle.nn.ReLU(),
        )
        self.conv3 = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_chn, in_chn, kernel_size=3, padding=1),
            paddle.nn.ReLU(),
        )
        self.conv_out = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_chn, out_chn, kernel_size=1, padding=0),
            # SynchronizedBatchNorm2d(out_chn, momentum=bn_mom),
            nn.BatchNorm2D(out_chn,momentum=bn_mom),
            paddle.nn.ReLU(),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2+x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2+y1)

        return self.conv_out(x1 + x2 + x3 + y1 + y2 + y3)


class densecat_cat_diff(nn.Layer):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_diff, self).__init__()
        self.conv1 = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_chn, in_chn, kernel_size=3, padding=1),
            paddle.nn.ReLU(),
        )
        self.conv2 = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_chn, in_chn, kernel_size=3, padding=1),
            paddle.nn.ReLU(),
        )
        self.conv3 = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_chn, in_chn, kernel_size=3, padding=1),
            paddle.nn.ReLU(),
        )
        self.conv_out = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_chn, out_chn, kernel_size=1, padding=0),
            # SynchronizedBatchNorm2d(out_chn, momentum=bn_mom),
            nn.BatchNorm2D(out_chn,momentum=bn_mom),
            paddle.nn.ReLU(),
        )

    def forward(self, x, y):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2+x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2+y1)
        out = self.conv_out(paddle.abs(x1 + x2 + x3 - y1 - y2 - y3))
        return out


class DF_Module(nn.Layer):
    def __init__(self, dim_in, dim_out, reduction=True):
        super(DF_Module, self).__init__()
        if reduction:
            self.reduction = paddle.nn.Sequential(
                paddle.nn.Conv2D(dim_in, dim_in//2, kernel_size=1, padding=0),
                # SynchronizedBatchNorm2d(dim_in//2, momentum=bn_mom),
                nn.BatchNorm2D(dim_in//2, momentum=bn_mom),
                paddle.nn.ReLU(),
            )
            dim_in = dim_in//2
        else:
            self.reduction = None
        self.cat1 = densecat_cat_add(dim_in, dim_out)
        self.cat2 = densecat_cat_diff(dim_in, dim_out)
        self.conv1 = nn.Sequential(
            nn.Conv2D(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            # SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
            nn.BatchNorm2D(dim_out, momentum=bn_mom),
            nn.ReLU(),
        )

    def forward(self, x1, x2):
        if self.reduction is not None:
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)
        x_add = self.cat1(x1, x2)
        x_diff = self.cat2(x1, x2)
        y = self.conv1(x_diff) + x_add
        return y
