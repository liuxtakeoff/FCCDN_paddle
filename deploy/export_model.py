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
from paddle.static import InputSpec
import paddle
import os
import sys
paddle.set_device("cpu")
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
from models.FCCDN import FCCDN


def get_args(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description='PaddlePaddle Classification Training', add_help=add_help)
    #模型权重地址
    parser.add_argument('--model_path', default='logs', help='model_weight')
    #模型搭建参数
    parser.add_argument('--num_band', default=3, type=int, help='num_band param for building model (default: 3)')
    parser.add_argument('--os', default=16, type=int, help='os param for building model (default: 16)')
    parser.add_argument('--use_se', default=True, type=bool, help='use se while building model (default: True)')
    parser.add_argument('--img_size', default=512, help='image size to export')
    #训练参数
    parser.add_argument('--device', default='gpu', help='device')
    parser.add_argument('--save-inference-dir', default='deploy', help='path where to save')
    parser.add_argument('--pretrained', default=None, help='pretrained model')

    args = parser.parse_args()
    return args


def export(args):
    # 建立模型
    model = FCCDN(num_band=args.num_band, os=args.os, use_se=args.use_se)
    # model = nn.Sequential(model, nn.Softmax())
    model.eval()
    # print("%s/%s/final.pdparmas"%(args.model_path,args.data_type))
    model_dict = paddle.load("%s/train/best.pdparams"%(args.model_path))
    model.set_dict(model_dict)

    model = paddle.jit.to_static(
        model,
        input_spec=[
            InputSpec(
                shape=[2,None, 3, args.img_size, args.img_size], dtype='float32')
        ])
    paddle.jit.save(model, os.path.join(args.save_inference_dir, "inference"))
    print(f"inference model has been saved into {args.save_inference_dir}")


if __name__ == "__main__":
    args = get_args()
    export(args)
