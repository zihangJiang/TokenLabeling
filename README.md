# Token Labeling: Training an 85.4% Top-1 Accuracy Vision Transformer with 56M Parameters on ImageNet ([arxiv](https://arxiv.org/abs/2104.10858))

This is a Pytorch implementation of our technical report. 



![Compare](Compare.png)

Comparison between the proposed LV-ViT and other recent works based on transformers. Note that we only show models whose model sizes are under 100M.

#### Training Pipeline

![Pipeline](Pipeline.png)

Our codes are based on the [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) by [Ross Wightman](https://github.com/rwightman).

#### LV-ViT Models

| Model                           | layer | dim  | Image resolution |  Param  | Top 1 |Download |
| :------------------------------ | :---- | :--- | :--------------: |-------: | ----: |   ----: |
| LV-ViT-S                        | 16    | 384  |       224        |  26.15M |  83.3 |[link](https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_s-224-83.3.pth.tar) |
| LV-ViT-S                        | 16    | 384  |       384        |  26.30M |  84.4 |[link](https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_s-26M-384-84.4.tar) |
| LV-ViT-M                        | 20    | 512  |       224        |  55.83M |  84.0 |[link](https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_m-56M-224-84.0.tar) |
| LV-ViT-M                        | 20    | 512  |       384        |  56.03M |  85.4 |[link](https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_m-56M-384-85.4.tar) |
| LV-ViT-L                        | 24    | 768  |       448        | 150.47M |  86.2 |[link](https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_l-150M-448-86.2.tar) |

#### Requirements

torch>=1.4.0
torchvision>=0.5.0
pyyaml
timm==0.4.5

data prepare: ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

#### Validation
Replace DATA_DIR with your imagenet validation set path and MODEL_DIR with the checkpoint path
```
CUDA_VISIBLE_DEVICES=0 bash eval.sh /path/to/imagenet/val /path/to/checkpoint
```

#### Label data

We provide NFNet-F6 generated dense label map [here](https://drive.google.com/file/d/1Cat8HQPSRVJFPnBLlfzVE0Exe65a_4zh/view?usp=sharing). As NFNet-F6 are based on pure ImageNet data, no extra training data is involved.


#### Training

Coming soon

#### Reference
If you use this repo or find it useful, please consider citing:
```
@misc{jiang2021token,
      title={Token Labeling: Training an 85.4% Top-1 Accuracy Vision Transformer with 56M Parameters on ImageNet}, 
      author={Zihang Jiang and Qibin Hou and Li Yuan and Daquan Zhou and Xiaojie Jin and Anran Wang and Jiashi Feng},
      year={2021},
      eprint={2104.10858},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

#### Related projects
[T2T-ViT](https://github.com/yitu-opensource/T2T-ViT/), [Re-labeling ImageNet](https://github.com/naver-ai/relabel_imagenet).
