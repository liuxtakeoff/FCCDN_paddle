# Linux GPU/CPU 基础训练推理测试

Linux GPU/CPU 基础训练推理测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能。

## 1. 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称 | 单机单卡 | 单机多卡 |
|  :----: |   :----:  |    :----:  |  :----:   |
|  FCCDN  | FCCDN | 正常训练 | - |


- 推理相关：

| 算法名称 | 模型名称 | device_CPU | device_GPU | batchsize |
|  :----:   |  :----: |   :----:   |  :----:  |   :----:   |
|  FCCDN   |  FCCDN |  支持 | 支持 | 1 |


## 2. 测试流程

### 2.1 准备数据

用于基础训练推理测试的数据位于`lite_data`，直接使用即可。

### 2.2 准备环境


- 安装PaddlePaddle：如果您已经安装了2.2或者以上版本的gpu版paddlepaddle，那么无需运行下面的命令安装paddlepaddle。
    ```
    # 需要安装2.2及以上版本的Paddle
    # 安装GPU版本的Paddle
    pip install paddlepaddle-gpu==2.2.0
    # 安装CPU版本的Paddle
    pip install paddlepaddle==2.2.0
    ```

- 安装依赖  
依赖都比较常见，建议大家直接运行，再查漏补缺

### 2.3 功能测试


测试方法如下所示，希望测试不同的模型文件，只需更换为自己的参数配置文件，即可完成对应模型的测试。

```bash
bash test_tipc/test_train_inference_python.sh ${your_params_file} lite_train_lite_infer
```

以`FCCDN`的`Linux GPU/CPU 基础训练推理测试`为例，命令如下所示。

```bash
#测试环境准备脚本
bash test_tipc/prepare.sh test_tipc/configs/FCCDN/train_infer_python.txt lite_train_lite_infer
```

```bash
#测试训练验证推理一体化脚本
bash test_tipc/test_train_inference_python.sh test_tipc/configs/FCCDN/train_infer_python.txt lite_train_lite_infer
```

输出结果如下，表示命令运行成功。

```bash
 Run successfully with command - python3.7 tools/train.py --data_dir lite_data --val_epoch 1 --output=./log/FCCDN/lite_train_lite_infer/norm_train_gpus_0 --epochs=2   --batch_size=1!  
......
 Run successfully with command - python3.7 tools/eval.py --data_dir lite_data/test --pretrained=./log/FCCDN/lite_train_lite_infer/norm_train_gpus_0/best.pdparams!  
......
 Run successfully with command - python3.7 deploy/export_model.py  --pretrained=./log/FCCDN/lite_train_lite_infer/norm_train_gpus_0/best.pdparams --save-inference-dir=./log/FCCDN/lite_train_lite_infer/norm_train_gpus_0!  
......
 Run successfully with command - python3.7 deploy/infer.py --imgA-path images/demoA.png --imgB-path images/demoB.png --use-gpu=True --model-dir=./log/FCCDN/lite_train_lite_infer/norm_train_gpus_0 --batch-size=1   --benchmark=False > ./log/FCCDN/lite_train_lite_infer/python_infer_gpu_batchsize_1.log 2>&1 !  
```
