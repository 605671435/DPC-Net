# DSNet-ImageNet-COCO

### Architecture

| **Method** | **Backbone** | **Plugin of every stage** |**Position of plugins** | **Ratio** |
| :--------- | :----------: | :-----------------------: | :--------------------: | :-------: |
| DSNet50-A  | ResNet50     | None,  None, None, DSM\* | after conv2   |2       |
| DSNet50-B  | ResNet50     | DSM, DSM, DSM, DSM   | after conv1       |16       |
| FcaNet50  | ResNet50     | MSCA\*\*, MSCA, MSCA, MSCA   | after conv3 | 16 | 

-   我们在论文中提出的主要方法是DSNet50-A，我们的出发点是为了更好的解决语义分割问题，所以DSNet50-A在语义分割上的实验优于FcaNet50，但是分类和检测上稍弱于FcaNet50。
-   为了更好分类和检测的结果，我们提出了DSNet50-B，其架构更像FcaNet50，但是参数量均少于FcaNet50和DSNet50-A，且分类和检测性能位于这两者之间。

\*DSM: Decoupled self-attention module

\*\*MSCA: Multi-spectral channel attention

### ImageNet1K

| **Method** | **Parameters** | **FLOPS** | **Train FPS** | **top1**  | **top5** |
| :--------- | :------------: | :-------: | :-----------: | :-------: | :------: |
| ResNet50   | 25.557M        | 4.109G    | 1677.51       | 76.48     | 93.17    |
| FcaNet50   | 30.121M        | 4.112G    | 1036.70       | 77.50     | 93.79    |
| DSNet50-A  | 27.369M        | 4.357G    | 1873.95       | **76.51** | 93.12    |
| DSNet50-B  | 29.106M        | 4.208G    | 1424.30       | **77.28** | 93.60    |

-   以上实验均是在mmpretrain \[1]中进行实验的，所以与FcaNet论文报告的有差异（FcaNet论文使用的是Nvidia APEX mixed precision training toolkit，而我们没有使用混合精度mixed precision）。

### COCO2018&#x20;

| **Model**   | **Method** | **Parameters** | **FLOPS** | Train FPS | $AP$ | $AP_{50}$ | $AP_{75}$ | $AP_{S}$ | $AP_{M}$ | $AP_{L}$ |
| :---------- | :--------: | :------------: | :-------: | :-------: | :----: | :-------: | :-------: | :------: | :------: | :------: |
| Faster-RCNN | ResNet50   | 41.750M        | 187.20G   | 46.75     | 37.4   | 58.3      | 40.5      | 21.9     | 40.7     | 48.1     |
| Faster-RCNN | FcaNet50   | 44.268M        | 187.31G   | 28.90     | 38.9   | 60.2      | 42.4      | 23.1     | 42.5     | 49.9     |
| Faster-RCNN | DSNet50-A  | 45.302M        | 188.97G   | 44.95     | 37.8   | 59.4      | 40.8      | 23       | 41.6     | 48.1     |
| Faster-RCNN | DSNet50-B  | 43.565M        | 194.27G   | **64.89** | 38.2   | 59.6      | 41.5      | 22.8     | 42.1     | 48.5     |

-   我们的DSNet50-B的Parameters相比FcaNet较少，训练的FPS两倍于FcaNet50.





-   以下是两个**语义分割实验**，注意，我们的DSNet50-A的Parameters和FLOPS较大，是由于我们的DSM保留了较高的空间分辨率和通道维度，也因此获得更好的分割效果。

### Synapse

| **Method** | **Parameters** | **FLOPS** | **Train FPS** | **mDice** | **mIoU** |
| :--------- | :------------: | :-------: | :-----------: | :-------: | :------: |
| ResNet50   | 47.13M         | 197.86G   | 7.52\*        | 82.92     | 74.72    |
| FcaNet50   | 49.62M         | 196.67G   | 81.60         | 84.06     | 76.10    |
| DSNet50-A  | 50.67M         | 205.93G   | 60.4          | **85.07** | 77.61    |
| DSNet50-B  | 48.94M         | 204.44G   | 95.72         | **83.8**  | 75.76    |

\*\*在单卡A5000上训练，其他都是8卡4090训练。

### Cityscapes

| **Decoder** | **Method**       | **Pretrained weights**\* | **Parameters** | **FLOPS** | **Train FPS** | **mIoU**      | **mAcc**  | **aAcc**  |
| :---------- | :--------------: | :----------------------: | :------------: | :-------: | :-----------: | :-----------: | :-------: | :-------: |
| FCN         | ResNet50 V1C\*\* | -                        | 47.13M         | 395.76G   | -             | 72.25\*\*\*\* | -         | -         |
| FCN         | FcaNet50         | FcaNet50                 | 49.65M         | 395.91G   | 48.24         | 75.63         | 82.93     | 95.78     |
| FCN         | DSNet50-A        | DSNet50-A                | 50.68M         | 411.91G   | 46.00         | **76.01**     | 83.59     | 95.86     |
| FCN         | DSNet50-B        | DSNet50-B                | 48.95M         | 408.92G   | 49.44         | **75.25**     | 83.00     | 95.62     |
| FCN         | ResNet50 V1C     | ResNet50 V1C             | 47.13M         | 395.76G   | 45.6          | 75.51         | 83.19     | 95.89     |
| FCN         | FcaNet50         | ResNet50 V1C             | 49.65M         | 395.91G   | 69.76         | 76.45         | 83.38     | 95.93     |
| FCN         | DSNet50-A        | ResNet50 V1C             | 50.68M         | 411.91G   | 70.96         | **77.25**     | **84.48** | **96.83** |

\*由于时间关系，我们没有在ImageNet1K预训练基于ResNet V1C的FcaNet50和DSNet50，而使用了两种训练策略：（1）使用基于ResNet(即ResNet V1B)的FcaNet50和DSNet50的预训练权重；（2）都使用ResNet50 V1C的训练权重。

\*\*Compare to ResNet, ResNet V1C replace the 7x7 conv in the stem with three 3x3 convs.

\*\*\*由于时间关系，我们没有亲自训练，ResNet50的mIoU结果使用的是mmseg \[2]的official results \[3].



## Reference

\[1] [https://github.com/open-mmlab/mmpretrain](https://github.com/open-mmlab/mmpretrain "https://github.com/open-mmlab/mmpretrain")

\[2] [https://github.com/open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation "https://github.com/open-mmlab/mmsegmentation")

\[3] [https://github.com/open-mmlab/mmsegmentation/tree/main/configs/fcn](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/fcn "https://github.com/open-mmlab/mmsegmentation/tree/main/configs/fcn")
