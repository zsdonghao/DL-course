# 深度学习作业

## 第一课 神经网络基础
### 代码文件说明
* `ch1_nn/examples.py` 课堂演示代码，主要用来学习参考 
* `ch1_nn/bp_np.py` 课后作业代码, 使用numpy实现BP算法
* `ch1_nn/mnist_mlp.py` 课后作业代码, MNIST手写数字分类  
### 数据集文件夹结构
解压`ch1_nn/mnist.zip`后，文件夹结构如下：
```
ch1_nn/mnist/
├── X_train.npy
├── X_test.npy
├── X_val.npy
├── y_train.npy
├── y_test.npy
└── y_val.npy

```
### 运行说明
```
cd ch1_nn
unzip mnist.zip # 解压数据集, 如果是windows系统, 可以直接解压
python examples.py # 运行课堂演示代码
python bp_np.py # 运行课后作业代码，使用numpy实现BP算法
python mnist_mlp.py # 运行课后作业代码，使用pytorch实现MNIST手写数字分类
```
