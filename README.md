tensorflow-CoViAR
===
This repository reimplement [CoViAI](https://github.com/chaoyuaw/pytorch-coviar) on TensorFlow, modifying some architectural design from their original pyTorch version. The model can be trained on both [UCF101](http://crcv.ucf.edu/data/UCF101.php) and [HMDB-51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) as the original implementation in their paper.


## Prerequisites
The model is implemented in the following environment:
* python3.5+
* CUDA 9.0
* cuDNN 7.2+
* TensorFlow 1.11
* FFmpeg (Please follow [this description](https://github.com/chaoyuaw/pytorch-coviar/blob/master/GETTING_STARTED.md) from the author to install FFmpeg for using the data loader)

## Datasets
As mentioned in the paper, original .avi input videos need to be encoded into the .mpeg4 format with the GOP structure described in this paper. For convenience, we provide converted training data in the following links. 
* UCF101 (15.35 GB)
* [HMDB-51](https://drive.google.com/file/d/1J_jvE57bAP0HM8wvlg9gMFNuqcxqlGm4/view?usp=sharing) (3.57 GB)

## Data Loader
The author provides an excellent data loader to extract the compressed represnetation (i.e., I-frames, motion vectors, and residuals as described in their paper) from .mpeg4 videos. Based on their implementation, we integrate this loader with the TensorFlow dataset API to train our model more efficiently.

To use the data loader, please follow this [instruction](https://github.com/chaoyuaw/pytorch-coviar/blob/master/GETTING_STARTED.md#data-loader) from the author. Before running our code, make sure you have built the data loader in the folder without any error message.

## Pretrained Weights
Loading existing pretrained models and training their weights in TensorFlow could be an arduous and tedious task. We provide the weights of ResNet152 and ResNet50 for initializing our model.
* [PretrainedResNet](https://drive.google.com/file/d/1ZGaswKgj8yKE6DAwpibl2LbuvU8-zGRH/view?usp=sharing) (791 MB)

## Train the Model
After prepaing the above data, it is now ready to train the CoViAR model. One can simply type
```
python3 tf-coviar.py
```
to run the training scripts. On the top of the script, there are some hyperparameters and dataset directory defined by tf.app.flags. Please adjust the setting to fit your training environment.

Note that different from the original pyTorch implementation which train each model separately, we can train the three prediction models end-to-end and update the fusion parameters accordingly. 

## Visualization
It is also possible to visualize the classication loss via tensorboard. Here is the example and result:
```
tensorboard --logdir=logs/coviar
```
#### Visualization of classification loss

![](https://i.imgur.com/fnqb9S4.png)

#### Visualization of validation accuracy

![](https://i.imgur.com/yt3FeBz.png)

Note taht the i-frames model is based on ResNet 152, therefore it takes more time to converge during training.

# Reference

Chao-Yuan Wu, et al., "Compressed Video Action Recognition", CVPR 2018.

"[pytorch-coviar](https://github.com/chaoyuaw/pytorch-coviar)", original implemetaion from the author
