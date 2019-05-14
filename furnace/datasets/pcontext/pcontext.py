#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2017/12/16 下午8:41
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : mclane.py

from datasets.BaseDataset import BaseDataset


class PascalContext(BaseDataset):
    @classmethod
    def get_class_colors(*args):
        return [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128],
                [128, 0, 0], [128, 0, 128], [128, 128, 0],
                [128, 128, 128],
                [0, 0, 64], [0, 0, 192], [0, 128, 64],
                [0, 128, 192],
                [128, 0, 64], [128, 0, 192], [128, 128, 64],
                [128, 128, 192], [0, 64, 0], [0, 64, 128],
                [0, 192, 0],
                [0, 192, 128], [128, 64, 0], ]

    @classmethod
    def get_class_names(*args):
        return ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                "cat", "chair", "cow", "table", "dog", "horse", "motorbike",
                "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
                "bag", "bed", "bench", "book", "building", "cabinet", "ceiling",
                "cloth", "computer", "cup", "door", "fence", "floor", "flower",
                "food", "grass", "ground", "keyboard", "light", "mountain",
                "mouse", "curtain", "platform", "sign", "plate", "road", "rock",
                "shelves", "sidewalk", "sky", "snow", "bedclothes", "track",
                "tree", "truck", "wall", "water", "window", "wood"]


if __name__ == "__main__":
    data_setting = {
        'img_root': '/unsullied/sharefs/g:research_detection/GeneralDetection/VOC/VOC/VOC2012_AUG/',
        'gt_root': '/unsullied/sharefs/g:research_detection/GeneralDetection/VOC/VOC/VOC2012_AUG',
        'train_source': '/unsullied/sharefs/g:research_detection/GeneralDetection/VOC/VOC/VOC2012_AUG/config/train.txt',
        'eval_source': '/unsullied/sharefs/g:research_detection/GeneralDetection/VOC/VOC/VOC2012_AUG/config/val.txt'}
    voc = VOC(data_setting, 'train', None)
    print(voc.get_class_names())
    print(voc.get_length())
    print(next(iter(voc)))
