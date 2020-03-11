# TorchSeg
This project aims at providing a fast, modular reference implementation for semantic segmentation models using PyTorch.

![demo image](demo/cityscapes_demo_img.png)

## Highlights
- **Modular Design:** easily construct customized semantic segmentation models by combining different components.
- **Distributed Training:** **>60%** faster than the multi-thread parallel method([nn.DataParallel](https://pytorch.org/docs/stable/nn.html#dataparallel)), we use the multi-processing parallel method.
- **Multi-GPU training and inference:** support different manners of inference.
- Provides pre-trained models and implement different semantic segmentation models.

## Prerequisites
- PyTorch 1.0
  - `pip3 install torch torchvision`
- Easydict
  - `pip3 install easydict`
- [Apex](https://nvidia.github.io/apex/index.html)
- Ninja
  - `sudo apt-get install ninja-build`
- tqdm
  - `pip3 install tqdm`
  
## Updates
v0.1.1 (05/14/2019)
- Release the pre-trained models and all trained models
- Add PSANet for ADE20K
- Add support for CamVid, PASCAL-Context datasets
- Start only supporting the distributed training manner

## Model Zoo
### Pretrained Model
- [ResNet18](https://drive.google.com/file/d/1PP7mEMvqW6vBMDeNhrS6l13RU5TwI6cA/view?usp=sharing)
- [ResNet50](https://drive.google.com/file/d/1iEshXXzI3tCexo2CH92TNNOyizf2R_db/view?usp=sharing)
- [ResNet101](https://drive.google.com/file/d/1iELk6WeQ1smockQJGKU_DEG6slcqw6Mu/view?usp=sharing)

### Supported Model
- FCN
- [DFN](https://arxiv.org/abs/1804.09337) 
- [BiSeNet](https://arxiv.org/abs/1808.00897)
- PSPNet
- PSANet

### Performance and Benchmarks
SS:Single Scale MSF:Multi-scale + Flip

### PASCAL VOC 2012
 Methods | Backbone | TrainSet | EvalSet | Mean IoU(ss) | Mean IoU(msf) | Model 
:--:|:--:|:--:|:--:|:--:|:--:|:--:
 FCN-32s     | R101_v1c | *train_aug*  | *val*  | 71.26 | -                 |  
 DFN(paper)  | R101_v1c | *train_aug*  | *val*  | 79.67 | 80.6<sup>*</sup>  |  
 DFN(ours)   | R101_v1c | *train_aug*  | *val*  | 79.40 | 81.40             | [GoogleDrive](https://drive.google.com/file/d/1dK5v1oakTMP1UKMARfYf5kdBP15mEgiL/view?usp=sharing) 

80.6<sup>*</sup>: this result reported in paper is further finetuned on *train* dataset. 

### Cityscapes
#### Non-real-time Methods
 Methods | Backbone |OHEM| TrainSet | EvalSet | Mean IoU(ss) | Mean IoU(msf) | Model 
:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:
 DFN(paper)                 | R101_v1c |✗| *train_fine* | *val*  | 78.5  | 79.3   | 
 DFN(ours)                  | R101_v1c |✗| *train_fine* | *val*  | 79.09 | 80.41  | [GoogleDrive](https://drive.google.com/file/d/1QGM652rWQWZx83oe2A5r48HGYtfZCRrm/view?usp=sharing)
 DFN(ours)                  | R101_v1c |✓| *train_fine* | *val*  | 79.16 | 80.53  | [GoogleDrive](https://drive.google.com/file/d/1KEX5g5dXF2cNpCh1NUe9iKvVye9g9JaD/view?usp=sharing) 
 BiSeNet(paper)             | R101_v1c |✓| *train_fine* | *val*  |  -    | 80.3   |  
 BiSeNet(ours)              | R101_v1c |✓| *train_fine* | *val*  | 79.09 | 80.39  | [GoogleDrive](https://drive.google.com/file/d/1yTbozInCLGiCklJ8plNTl5GgeGnW_fC0/view?usp=sharing) 
 BiSeNet(paper)             | R18      |✓| *train_fine* | *val*  | 76.21 | 78.57  |  
 BiSeNet(ours)              | R18      |✓| *train_fine* | *val*  | 76.28 | 78.00  | [GoogleDrive](https://drive.google.com/file/d/1hFF-J9qoXlbVRRUr29aWeQpL4Lwn45mU/view?usp=sharing)  
 BiSeNet(paper)             | X39      |✓| *train_fine* | *val*  | 70.1  | 72     | 
 BiSeNet(ours)<sup>*</sup>  | X39      |✓| *train_fine* | *val*  | 70.32 | 72.06  | [GoogleDrive](https://drive.google.com/file/d/1hb_qk3QLwZtQUmevZUFOHNiRHhRr0IQB/view?usp=sharing)  
 
 #### Real-time Methods
  Methods | Backbone |OHEM| TrainSet | EvalSet | Mean IoU | Model 
:--:|:--:|:--:|:--:|:--:|:--:|:--:
 BiSeNet(paper)             | R18      |✓| *train_fine* | *val*  | 74.8  | 
 BiSeNet(ours)              | R18      |✓| *train_fine* | *val*  | 74.83 | [GoogleDrive](https://drive.google.com/file/d/1bLc7YC0qePcKZQTLrqsrNRnzx3cmC-1C/view?usp=sharing) 
 BiSeNet(paper)             | X39      |✓| *train_fine* | *val*  | 69    | 
 BiSeNet(ours)<sup>*</sup>  | X39      |✓| *train_fine* | *val*  | 68.51 | [GoogleDrive](https://drive.google.com/file/d/1xZEQLtJR-FSYt6ri7kfSTJ7UT9fXZtxc/view?usp=sharing) 
 
BiSeNet(ours)<sup>*</sup>: because we didn't pre-train the Xception39 model on ImageNet in PyTorch, we train this experiment from scratch. We will release the pre-trained Xception39 model in PyTorch and the corresponding experiment.

### ADE
  Methods | Backbone | TrainSet | EvalSet | Mean IoU(ss) | Accuracy(ss)|  Model 
:--:|:--:|:--:|:--:|:--:|:--:|:--:
 PSPNet(paper)             | R50_v1c  | *train* | *val*  | 41.68 | 80.04 |  
 PSPNet(ours)              | R50_v1c  | *train* | *val*  | 41.65 | 79.74 | [GoogleDrive](https://drive.google.com/file/d/1jDj3UJCAAffmPQ4ckTOFxQdvFWxThCoy/view?usp=sharing)
 PSPNet(paper)             | R101_v1c | *train* | *val*  | 41.96 | 80.64 |  
 PSPNet(ours)              | R101_v1c | *train* | *val*  | 42.89 | 80.55 | [GoogleDrive](https://drive.google.com/file/d/1Y6OErgb9F1qc2cJmGhW6VlUfaSdFfmfu/view?usp=sharing)
 PSANet(paper)             | R50_v1c  | *train* | *val*  | 41.92 | 80.17 |  
 PSANet(ours)<sup>*</sup>  | R50_v1c  | *train* | *val*  | 41.67 | 80.09 | [GoogleDrive](https://drive.google.com/file/d/1bHD1NBgJUY1PSDgyuXH2FUXPXUB7DlLj/view?usp=sharing)
 PSANet(paper)             | R101_v1c | *train* | *val*  | 42.75 | 80.71 |  
 PSANet(ours)              | R101_v1c | *train* | *val*  | 43.04 | 80.56 | [GoogleDrive](https://drive.google.com/file/d/1nnq0-pDNTttvSgtqiSo-QZ4HeYVaZl5T/view?usp=sharing)

PSANet(ours)<sup>*</sup>: The original PSANet in the paper constructs the 
attention map with over-parameters, while we only predict the attention map with 
the same size of the feature map. The performance is almost similar to the 
original one.

### To Do
- [ ] offer comprehensive documents 
- [ ] support more semantic segmentation models
  - [ ] Deeplab v3 / Deeplab v3+
  - [ ] DenseASPP
  - [ ] EncNet
  - [ ] OCNet


## Training
1. create the config file of dataset:`train.txt`, `val.txt`, `test.txt`   
    file structure：(split with `tab`)
    ```txt
    path-of-the-image   path-of-the-groundtruth
    ```
2. modify the `config.py` according to your requirements
3. train a network:

### Distributed Training
We use the official `torch.distributed.launch` in order to launch multi-gpu training. 
This utility function from PyTorch spawns as many Python processes as the number 
of GPUs we want to use, and each Python process will only use a single GPU.

For each experiment, you can just run this script:
```bash
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py
```

## Inference
In the evaluator, we have implemented the multi-gpu inference base on the multi-process. In the inference phase, the function will spawns as many Python processes as the number of GPUs we want to use, and each Python process will handle a subset of the whole evaluation dataset on a single GPU.
1. evaluate a trained network on the validation set:
    ```bash
    python3 eval.py
    ```
2. input arguments:
    ```bash
    usage: -e epoch_idx -d device_idx [--verbose ] 
    [--show_image] [--save_path Pred_Save_Path]
    ```


## Disclaimer
This project is under active development. So things that are currently working might break in a future release. However, feel free to open issue if you get stuck anywhere.

## Citation
The following are BibTeX references. The BibTeX entry requires the url LaTeX package.

Please consider citing this project in your publications if it helps your research. 
```
@misc{torchseg2019,
  author =       {Yu, Changqian},
  title =        {TorchSeg},
  howpublished = {\url{https://github.com/ycszen/TorchSeg}},
  year =         {2019}
}
```

Please consider citing the [DFN](https://arxiv.org/abs/1804.09337) in your publications if it helps your research. 

```
@inproceedings{yu2018dfn,
  title={Learning a Discriminative Feature Network for Semantic Segmentation},
  author={Yu, Changqian and Wang, Jingbo and Peng, Chao and Gao, Changxin and Yu, Gang and Sang, Nong},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
```

Please consider citing the [BiSeNet](https://arxiv.org/abs/1808.00897) in your publications if it helps your research. 

```
@inproceedings{yu2018bisenet,
  title={Bisenet: Bilateral segmentation network for real-time semantic segmentation},
  author={Yu, Changqian and Wang, Jingbo and Peng, Chao and Gao, Changxin and Yu, Gang and Sang, Nong},
  booktitle={European Conference on Computer Vision},
  pages={334--349},
  year={2018},
  organization={Springer}
}
```

## Why this name, Furnace?
Furnace means the **Alchemical Furnace**. We all are the **Alchemist**, so I hope everyone can have a good alchemical furnace to practice the **Alchemy**. Hope you can be a excellent alchemist. 

