# Does the Attention Mechanism Act As a Correction Factor?

A pytorch implenment for Does the Attention Mechanism Act As a Correction Factor?

# Directory

-   [Requirement](#Requirement)
-   [Installation for MMEngine and MMSegmentation](#Installation-for-MMEngine-and-MMSegmentation)
-   [Datasets](#Datasets)
-   [Training and testing](#Training-and-testing)
-   [Results](#Results)
    -   [Synapse](#Synapse)
    -   [ACDC](#ACDC)
-   [Acknowledgement](#Acknowledgement)
-   [Citation](#Citation)
-   [License](#License)

# Requirement
## MMEngine and MMSegmentation
This repo is required the MMEngine and MMSegmentation:

-   Pytorch >= 1.12.0
-   MMEngine >= 0.7.0
-   MMSegmentation >= 1.0.0rc6

You can run the following scripts to install MMEngine and MMSegmentation.

```pycon
pip install -U openmim
mim install mmengine>=0.7.0
mim install mmsegmentation>=1.0.0rc6
```

## Other required package:
- SimpleITK
- nibabel
- WandB

You can run the following script to install requirement package:

```pycon
pip install -r requirements.txt
```

# Datasets

You counld get Synapse raw data from [https://www.synapse.org/#!Synapse:syn3193805/wiki/217752](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752 "https://www.synapse.org/#!Synapse:syn3193805/wiki/217752").

Or email us to get our proprcessed data.

## Training and testing

-   Training on one GPU:

```pycon
python ./train.py {config}
```

-   Testing on one GPU:

```pycon
python ./test.py {config}
```
{config} means the config path. The config path can be found in [configs](configs "configs").
# Results

### Synapse

| Method                     | Backbone   | Crop Size | Lr schd | mDice ↑    | mHD95 ↓    | Config                                                                           |
| -------------------------- | ---------- | --------- | ------- | ---------- | ---------- | -------------------------------------------------------------------------------- |
| FCN                        | ResNet50   | 512x512   | 40000   | 81.86±0.68 | 26.57±5.22 | [config](configs/resnet/fcn_r50-d8_1xb2-40k_synapse-512x512.py "config")         |
| FCN+SE                     | ResNet50   | 512x512   | 40000   | 82.47±0.69 | 24.87±3.69 | [config](configs/se/fcn_r50-se-d8_1xb2-40k_synapse-512x512.py "config")          |
| FCN+EncNet                 | ResNet50   | 512x512   | 40000   | 82.08±0.26 | 25.00±2.41 | [config](configs/encnet/fcn_r50-encnet-d8_1xb2-40k_synapse-512x512.py "config")  |
| FCN+ECANet                 | ResNet50   | 512x512   | 40000   | 81.72±0.52 | 23.19±1.70 | [config](configs/ecanet/fcn_r50-ecanet-d8_1xb2-40k_synapse-512x512.py "config")  |
| FCN+CBAM                   | ResNet50   | 512x512   | 40000   | 81.82±1.01 | 24.38±3.46 | [config](configs/cbam/fcn_r50-cbam-d8_1xb2-40k_synapse-512x512.py "config")      |
| DANet                      | ResNet50   | 512x512   | 40000   | 82.23±0.67 | 26.24±1.55 | [config](configs/danet/danet_r50-d8_1xb2-40k_synapse-512x512.py "config")        |
| CCNet                      | ResNet50   | 512x512   | 40000   | 81.51±0.85 | 27.73±4.31 | [config](configs/ccnet/ccnet_r50-d8_1xb2-40k_synapse-512x512.py "config")        |
| GCNet                      | ResNet50   | 512x512   | 40000   | 81.83±0.98 | 25.52±3.38 | [config](configs/gcnet/gcnet_r50-d8_1xb2-40k_synapse-512x512.py "config")        |
| HamNet                     | ResNet50   | 512x512   | 40000   | 82.37±0.59 | 24.36±1.48 | [config](configs/hamnet/hamnet_r50-d8_1xb2-40k_synapse-512x512.py "config")      |
| EANet                      | ResNet50   | 512x512   | 40000   | 81.77±0.25 | 27.00±2.50 | [config](configs/eanet/eanet_r50-d8_1xb2-40k_synapse-512x512.py "config")        |
| FCN+PSA (p)                | ResNet50   | 512x512   | 40000   | 82.66±0.64 | 21.99±0.88 | [config](configs/psa/fcn_r50-psa-d8_1xb2-40k_synapse-512x512.py "config")        |
| FCN+PSA (s)                | ResNet50   | 512x512   | 40000   | 82.48±0.56 | 22.33±1.79 | [config](configs/psa/fcn_r50-psa_s-d8_1xb2-40k_synapse-512x512.py "config")      |
| UPerNet                    | ConvNeXt-B | 512x512   | 80000   | 83.24±0.46 | 28.16±3.19 | [config](configs/convnext/conv_next_b-synapse-80k.py "config")                   |
| UPerNet                    | SegNeXt-B  | 512x512   | 80000   | 83.86±0.38 | 21.98±1.83 | [config](configs/segnext/upernet-segnext-b_1xb2-80k_synapse-512x512.py "config") |
| HamNet                     | MSCAN-B    | 512x512   | 80000   | 84.72±0.51 | 20.68±3.57 | [config](configs/segnext/upernet-segnext-b_1xb2-80k_synapse-512x512.py "config") |
| **DSNet (ours)**           | ResNet50   | 512x512   | 40000   | 82.78±0.91 | 22.75±3.39 | [config](configs/dsa/dsnet_r50-d8_1xb2-40k_synapse-512x512.py "config")          |
| **FCN+DSM (ours)**         | ResNet50   | 512x512   | 40000   | 83.25±0.56 | 20.55±3.56 | [config](configs/dsa/fcn_r50-ex-d8_1xb2-40k_synapse-512x512.py "config")         |
| **DSNet (ours)** | MSCAN-B    | 512x512   | 80000   | 84.69±0.50 | 18.28±3.55 | [config](configs/segnext/upernet-segnext-b_1xb2-80k_synapse-512x512.py "config") |

### ACDC

| Method             | Backbone | Crop Size | Lr schd | mDice ↑    | Config                                                                       |
| ------------------ | -------- | --------- | ------- | ---------- | ---------------------------------------------------------------------------- |
| FCN                | ResNet50 | 256x256   | 40000   | 87.87±0.43 | [config](configs/resnet/fcn_r50-d8_1xb2-40k_acdc-256x256.py "config")        |
| FCN+SE             | ResNet50 | 256x256   | 40000   | 87.75±0.59 | [config](configs/se/fcn_r50-se-d8_1xb2-40k_acdc-256x256.py "config")         |
| FCN+EncNet         | ResNet50 | 256x256   | 40000   | 87.63±0.39 | [config](configs/encnet/fcn_r50-encnet-d8_1xb2-40k_acdc-256x256.py "config") |
| FCN+ECANet         | ResNet50 | 256x256   | 40000   | 87.66±0.43 | [config](configs/ecanet/fcn_r50-ecanet-d8_1xb2-40k_acdc-256x256.py "config") |
| FCN+CBAM           | ResNet50 | 256x256   | 40000   | 87.39±0.53 | [config](configs/cbam/fcn_r50-cbam-d8_1xb2-40k_acdc-256x256.py "config")     |
| DANet              | ResNet50 | 256x256   | 40000   | 87.71±0.45 | [config](configs/danet/danet_r50-d8_1xb2-40k_acdc-256x256.py "config")       |
| CCNet              | ResNet50 | 256x256   | 40000   | 87.52±0.61 | [config](configs/ccnet/ccnet_r50-d8_1xb2-40k_acdc-256x256.py "config")       |
| GCNet              | ResNet50 | 256x256   | 40000   | 87.81±0.59 | [config](configs/gcnet/gcnet_r50-d8_1xb2-40k_acdc-256x256.py "config")       |
| HamNet             | ResNet50 | 256x256   | 40000   | 87.73±0.48 | [config](configs/hamnet/hamnet_r50-d8_1xb2-40k_acdc-256x256.py "config")     |
| EANet              | ResNet50 | 256x256   | 40000   | 87.72±0.39 | [config](configs/eanet/eanet_r50-d8_1xb2-40k_acdc-256x256.py "config")       |
| FCN+PSA (p)        | ResNet50 | 256x256   | 40000   | 87.45±0.48 | [config](configs/psa/fcn_r50-psa-d8_1xb2-40k_acdc-256x256.py "config")       |
| FCN+PSA (s)        | ResNet50 | 256x256   | 40000   | 87.92±0.37 | [config](configs/psa/fcn_r50-psa_s-d8_1xb2-40k_acdc-256x256.py "config")     |
| **DSNet (ours)**   | ResNet50 | 256x256   | 40000   | 87.42±0.38 | [config](configs/dsa/dsnet_r50-d8_1xb2-40k_acdc-512x512.py "config")         |
| **FCN+DSM (ours)** | ResNet50 | 256x256   | 40000   | 87.97±0.46 | [config](configs/dsa/fcn_r50-ex-d8_1xb2-40k_acdc-256x256.py "config")        |

# Acknowledgement

Specially thanks to [MMSegmentation](https://github.com/open-mmlab/mmsegmentation "MMSegmentation"), [MMEngine](https://github.com/open-mmlab/mmengine "MMEngine").

# Citation

```bash
@misc{mmseg2020,
  title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
  author={MMSegmentation Contributors},
  howpublished = {\url{[https://github.com/open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation)}},
  year={2020}
}
```

```bash
@article{mmengine2022,
  title   = {{MMEngine}: OpenMMLab Foundational Library for Training Deep Learning Models},
  author  = {MMEngine Contributors},
  howpublished = {\url{https://github.com/open-mmlab/mmengine}},
  year={2022}
}
```

# License

This project is released under the [Apache 2.0 license](https://github.com/open-mmlab/mmsegmentation/blob/main/LICENSE "Apache 2.0 license") of mmsegmentation.
