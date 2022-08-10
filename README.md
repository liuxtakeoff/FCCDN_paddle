# FCCDN-paddle

## 目录

- [1. 简介]()
- [2. 数据集和复现精度]()
- [3. 准备数据与环境]()
    - [3.1 准备环境]()
    - [3.2 准备数据]()
- [4. 开始使用]()
    - [4.1 模型训练]()
    - [4.2 模型评估]()
    - [4.3 模型预测]()
- [5. 模型推理部署]()
- [6. 自动化测试脚本]()
- [7. LICENSE]()
- [8. 参考链接与文献]()

## 1. 简介
FCCDN是一种基于 的变化检测网络。其主要结构如下图所示。

**论文:** [FCCDN: Feature Constraint Network for VHR Image Change Detection](https://arxiv.org/pdf/2105.10860.pdf)

**参考repo:** [FCCDN_pytorch](https://github.com/chenpan0615/FCCDN_pytorch)

在此非常感谢`chenpan0615`等人贡献的[FCCDN_pytorch](https://github.com/chenpan0615/FCCDN_pytorch) ，提高了本repo复现论文的效率。

**aistudio体验教程:** [FCCDN_paddle](https://aistudio.baidu.com/aistudio/projectdetail/4411651)


## 2. 数据集和复现精度

- 数据集大小：共包含15个物品类别，解压分割后总大小在5.18G左右
- 数据集下载链接：[LEVIR-CD](https://justchenhao.github.io/LEVIR/)
- 训练权重下载链接：[logs]() (提取码：)
# 复现精度（Comparison to Li et al.）
|                  |    环境(env)     |    F1-score(%)    | precision(%)      |recall(%)          |
| ---------------  | --------------- | -------------- | -------------- | -------------- |
| FCCDN-torch(LEVIR-CD)  | RTX TITAN 24GB  |   92.29   |   93.07        |   91.52        |
| FCCDN-paddle(LEVIR-CD) |    V100 32GB  |      |         |           |


## 3. 准备数据与环境


### 3.1 准备环境

首先介绍下支持的硬件和框架版本等环境的要求：

- 硬件：GPU显存建议在24G以上
- 框架：
  - PaddlePaddle >= 2.2.0
- 环境配置： 
  - 首先使用`pip install -r requirements.txt`安装依赖库
  - 卸载旧版本opencv-python`pip uninstall opencv-python`
  - 安装新版本opencv-python`pip install opencv-python`，这是因为使用的albumentations库不兼容旧版本opencv-python
### 3.2 准备数据

#### 全量数据训练：
  - 下载好 [LEVIR-CD](https://justchenhao.github.io/LEVIR/) 数据集
  - 将其解压到 根目录下的**LEVIR-CD** 文件夹下,文件格式如下所示：
```
    │LEVIR-CD/
    ├──train/
    │  ├── A
    │  ├── B
    │  ├── label
    │     
    ├──val/
    │  ├── A
    │  ├── B
    │  ├── label
    │     
    ├──test/
    │  ├── A
    │  ├── B
    │  ├── label
```
- 运行分割代码`python tools/split_LEVIRCD.py --data_dir LEVIR-CD --out_dir Data`将训练数据集划分为512*512的小块,得到最终数据集
  
####少量数据训练：
  - 无需下载数据集，使用lite_data里的数据即可
  

## 4. 开始使用


### 4.1 模型训练

- 全量数据训练：
  - 下载好 [LEVIR-CD](https://justchenhao.github.io/LEVIR/) 数据集后，将其解压到 **./Data** 文件夹下
  - 运行指令`python tools/train.py --epochs 7000 --batch_size 32 --workers 4 --log_interval 10 --pretrained True`
- 少量数据训练：
  - 运行指令`python tools/train.py --data_dir lite_data --type lite --epochs 5 --batch_size 4`
- 部分训练日志如下所示：
```
> python tools/train.py --data_dir lite_data --type lite --epochs 5 --batch_size 4 --cuda False
Namespace(batch_size=4, cuda='False', data_dir='lite_data', epochs=5, freeze_resnet=20, head_layer=1, lr=0.03, model_dir='logs', optim='sgd', pretrained=False, save_interval=500, test_epochs=-1, type='l
ite', variant='3way', workers=0)
using device: cpu
training bottle
loading images
loaded 209 images
epoch:1/5 loss:1.2578 avg_reader_cost:0.05 avg_batch_cost:3.01 avg_ips:0.75
epoch:2/5 loss:1.6850 avg_reader_cost:0.02 avg_batch_cost:2.81 avg_ips:0.70
epoch:3/5 loss:1.5016 avg_reader_cost:0.02 avg_batch_cost:2.75 avg_ips:0.69
...
``` 


### 4.2 模型评估

- 全量数据模型评估：`python eval.py`
- 少量数据模型评估：`python tools/eval.py --data_dir lite_data --type lite`
```
> python tools/eval.py --data_dir lite_data --type lite --cuda False
Namespace(cuda='False', data_dir='lite_data', density='sklearn', head_layer=1, model_dir='logs', save_plots=True, type='lite')
evaluating bottle
loading model logs/bottle/final.pdparams
loading images
loaded 8 images
using density estimation GaussianDensitySklearn
bottle AUC: 0.875
average auroc:0.8750
``` 

### 4.3 模型预测（需要预先完成4.1训练及4.2验证）

- 基于原始代码的模型预测：`python tools/predict.py --data_type bottle --img-path images/demo0.png --dist_th 1.0`


## 5. 模型推理部署

- 基于推理引擎的模型预测：
```
python deploy/export_model.py
python deploy/infer.py --data_type bottle --img-path images/demo0.png --dist_th 1.0
```
部分结果如下：
```
> python deploy/export_model.py
inference model has been saved into deploy

> python deploy/infer.py --data_type bottle --img-path images/demo0.png --dist_th 1.0
image_name: images/demo0.png, class_id: 0, prob: 0.07689752858017344
``` 


## 6. 自动化测试脚本
- tipc创建指南请见[tipc创建及基本使用方法。](https://github.com/PaddlePaddle/models/blob/release/2.2/tutorials/tipc/train_infer_python/test_train_infer_python.md)
- 本项目TIPC脚本测试命令详见[Linux GPU/CPU 基础训练推理测试](test_tipc/docs/test_train_inference_python.md)
```bash
#测试环境准备脚本
bash test_tipc/prepare.sh test_tipc/configs/resnet18/train_infer_python.txt lite_train_lite_infer
```

```bash
#测试训练验证推理一体化脚本
bash test_tipc/test_train_inference_python.sh test_tipc/configs/resnet18/train_infer_python.txt lite_train_lite_infer
```

输出结果如下，表示命令运行成功。

```bash
 Run successfully with command - python3.7 tools/train.py --type lite --model_dir logs --output=./log/resnet18/lite_train_lite_infer/norm_train_gpus_0 --epochs=2   --batch_size=1!  
......
 Run successfully with command - python3.7 deploy/infer.py --use-gpu=True --model-dir=./log/resnet18/lite_train_lite_infer/norm_train_gpus_0 --batch-size=1   --benchmark=False > ./log/resnet18/lite_train_lite_infer/python_infer_gpu_batchsize_1.log 2>&1 !  
```

## 7. LICENSE

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。

## 8. 参考链接与文献
**参考论文:** [CutPaste: Self-Supervised Learning for Anomaly Detection and Localization](https://arxiv.org/pdf/2104.04015v1.pdf)

**参考repo:** [pytorch-cutpaste](https://github.com/Runinho/pytorch-cutpaste)
