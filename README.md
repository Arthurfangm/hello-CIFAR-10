# hello-CIFAR-10
## Introduction
This is an implementatoin of deep neural network for the image classification task on the [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset. Although CIFAR-10 seem a entry-level dataset for image classificatoin, it's not easy to gain a test precision over 90%. [This page](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130) summarize the SOTA methods on this task.</br>

To gain a higher test accuracy, Technology like Xavier(Init method) weight decay, batch normalization， dropout(to avoid overfitting) and [factional max-pooling](https://arxiv.org/abs/1412.6071)(alternative for 2x2 max-pooling) are involved in this implementatoin. And as for the net arcihtecture of my implementation, generally I imitate the net architecture illustrated in the paper of [Fractional Max-pooling](https://arxiv.org/abs/1412.6071), but have less layers.</br>

To have a good coding style and documents management, I refer to [pointNet](https://github.com/charlesq34/pointnet) when I code and construct the architecture of this project.</br>

## Installation and Run
The code is tested in:

    Python 3.6, Tensorflow 1.0.1, CUDA 8.0 and cuDNN 5.1 on Ubuntu 14.04.</br>
    
A new model can be trained using simply a line of code:

    python train.py
    
like the following image shows:
[image]()
