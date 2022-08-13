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
import sys
import os
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
from paddle import inference
from PIL import Image
import numpy as np
import paddle
import cv2


class InferenceEngine(object):
    """InferenceEngine

    Inference engina class which contains preprocess, run, postprocess
    """

    def __init__(self, args):
        """
        Args:
            args: Parameters generated using argparser.
        Returns: None
        """
        super().__init__()
        self.args = args

        # init inference engine
        self.predictor, self.config, self.input_tensor, self.output_tensor = self.load_predictor(
            os.path.join(args.model_dir, "inference.pdmodel"),
            os.path.join(args.model_dir, "inference.pdiparams"))

        # build transforms
        self.normalize = paddle.vision.transforms.Normalize(mean=[0.37772245912313807, 0.4425350597897193, 0.4464795300397427],
                                                             std=[0.1762166286060892, 0.1917139949806914, 0.20443966020731438])

        # wamrup
        if self.args.warmup > 0:
            for idx in range(args.warmup):
                print(idx)
                x = np.random.rand(2,1, 3, self.args.resize_size,
                                   self.args.resize_size).astype("float32")
                self.input_tensor.copy_from_cpu(x)
                self.predictor.run()
                self.output_tensor.copy_to_cpu()
        return

    def load_predictor(self, model_file_path, params_file_path):
        """load_predictor
        initialize the inference engine
        Args:
            model_file_path: inference model path (*.pdmodel)
            model_file_path: inference parmaeter path (*.pdiparams)
        Return:
            predictor: Predictor created using Paddle Inference.
            config: Configuration of the predictor.
            input_tensor: Input tensor of the predictor.
            output_tensor: Output tensor of the predictor.
        """
        args = self.args
        config = inference.Config(model_file_path, params_file_path)
        if args.use_gpu:
            config.enable_use_gpu(1000, 0)
        else:
            config.disable_gpu()
            # The thread num should not be greater than the number of cores in the CPU.
            config.set_cpu_math_library_num_threads(4)

        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)

        # get input and output tensor property
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])

        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])

        return predictor, config, input_tensor, output_tensor

    def preprocess(self, img_path):
        """preprocess
        Preprocess to the input.
        Args:
            img_path: Image path.
        Returns: Input data after preprocess.
        """
        img = cv2.imread(img_path)
        img = self.normalize(paddle.to_tensor(img.transpose(2,0,1)/255,dtype=paddle.float32))[None]
        return img

    def postprocess(self, x):
        """postprocess
        Postprocess to the inference engine output.
        Args:
            x: Inference engine output.
        Returns: Output data after argmax.
        """
        out0 = x.astype("uint8")
        mask = out0[0,0]*255

        img_savepath = os.path.join(self.args.result_savepath,"result.png")
        mask_img = Image.fromarray(mask,mode="L")
        mask_img.save(img_savepath)
        # cv2.imwrite(img_savepath, mask[0, 0] * 255)
        return mask,img_savepath

    def run(self, x):
        """run
        Inference process using inference engine.
        Args:
            x: Input data after preprocess.
        Returns: Inference engine output
        """
        self.input_tensor.copy_from_cpu(x)
        self.predictor.run()
        output = self.output_tensor.copy_to_cpu()
        return output


def get_args(add_help=True):
    """
    parse args
    """
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(description="PaddlePaddle Infer", add_help=add_help)
    parser.add_argument("--model-dir", default="deploy", help="inference model dir")
    parser.add_argument("--use-gpu", default=False, type=str2bool, help="use_gpu")
    parser.add_argument("--max-batch-size", default=16, type=int, help="max_batch_size")
    parser.add_argument("--batch-size", default=1, type=int, help="batch size")
    parser.add_argument("--resize-size", default=512, type=int, help="resize_size")
    parser.add_argument("--crop-size", default=512, type=int, help="crop_szie")
    parser.add_argument("--imgA-path", default="images/demoA.png",help="imgA path")
    parser.add_argument("--imgB-path", default="images/demoB.png",help="imgB path")
    parser.add_argument("--result_savepath", default="images",help="path to save infer result")
    parser.add_argument("--benchmark", default=False, type=str2bool, help="benchmark")
    parser.add_argument("--warmup", default=0, type=int, help="warmup iter")

    args = parser.parse_args()
    return args


def infer_main(args):
    """infer_main
    Main inference function.
    Args:
        args: Parameters generated using argparser.
    Returns:
        class_id: Class index of the input.
        prob: : Probability of the input.
    """
    inference_engine = InferenceEngine(args)

    # init benchmark
    if args.benchmark:
        import auto_log
        autolog = auto_log.AutoLogger(
            model_name="classification",
            batch_size=args.batch_size,
            inference_config=inference_engine.config,
            gpu_ids="auto" if args.use_gpu else None)

    assert args.batch_size == 1, "batch size just supports 1 now."

    # enable benchmark
    if args.benchmark:
        autolog.times.start()

    # preprocess
    imgA = inference_engine.preprocess(args.imgA_path)
    imgB = inference_engine.preprocess(args.imgB_path)
    if args.benchmark:
        autolog.times.stamp()

    pre = imgA.unsqueeze(1)
    post = imgB.unsqueeze(1)
    imgs = paddle.concat([pre, post])
    output = inference_engine.run(imgs.cpu().numpy())
    if args.benchmark:
        autolog.times.stamp()

    # postprocess
    mask, img_savepath = inference_engine.postprocess(output)

    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        autolog.report()

    print(f"image_name: {args.imgA_path,args.imgB_path}, result was saved in : {img_savepath}")
    return mask, img_savepath


if __name__ == "__main__":
    args = get_args()
    mask, img_savepath = infer_main(args)