
# INPA
This is official Pytorch implementation of [" Incrementally Adapting Pretrained Model Using
 Network Prior for Multi-Focus Image Fusion"](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10568542) published in IEEE TIP 2024.


## Setup

### Environment

 - [x] python 3.9
 - [x] torch 1.12.1
 - [x] cudatoolkit 11.3
 - [x] torchvision 0.13.1
 - [x] numpy 1.23.5
 - [x] opencv-python 4.7.0
 - [x] ...

### Evaluation dataset

- [x] [Real-MFF](https://github.com/Zancelot/Real-MFF)
- [x] [MFI-WHU](https://github.com/HaoZhang1018/MFI-WHU)
- [x] Lytro


## INPA Results

INPA results are available in [google drive](https://drive.google.com/drive/folders/1FxGmkj6DFScijblbtQMF80L2EjpeWAuP?usp=sharing).

## INPA Test
Download the pretrained backbone weight from [google drive](https://drive.google.com/drive/folders/1s3VLrxNRYJX_Ec03ukO_Mo2XQ2j0xqQC?usp=sharing), and put it under `./pretrained_backbone/`. Change the dataset name and path in the configuration file (test_inpa.yaml).

    python test_inpa.py


## Dataset Construct
Create multi-focus dataset by simulating continuously changing defocus blur based on the depth dataset [NYUDv2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html):

    python synthesize_df_dataset.py


## Backbone Pretrain
The unet-based is trained on the synthesized MFIF-SYNDoF datasert:

    python -m torch.distributed.launch --nproc_per_node=2 --master_port=3210 train_backbone.py


## Citation
```
@article{hu2024incrementally,
  title={Incrementally adapting pretrained model using network prior for multi-focus image fusion},
  author={Hu, Xingyu and Jiang, Junjun and Wang, Chenyang and Liu, Xianming and Ma, Jiayi},
  journal={IEEE Transactions on Image Processing},
  year={2024},
  publisher={IEEE}
}
```