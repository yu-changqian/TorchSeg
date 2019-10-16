#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2017/12/16 下午8:41
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : mclane.py

import cv2
import numpy as np
import scipy.io as sio

from datasets.BaseDataset import BaseDataset


class Stuff10K(BaseDataset):
    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        # cv2: B G R
        # h w c

        if 'mat' in filepath:
            img = sio.loadmat(filepath)['S']
            img = img.astype(np.float32)
        else:
            img = np.array(cv2.imread(filepath, mode), dtype=dtype)

        return img

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
        return ["person", "bicycle", "car", "motorcycle,", "airplane,", "bus,",
                "train,", "truck,", "boat,", "traffic_light,", "fire_hydrant,",
                "street_sign,", "stop_sign,", "parking_meter,", "bench,",
                "bird,", "cat,", "dog,",
                "horse,", "sheep,", "cow,", "elephant,", "bear,", "zebra,",
                "giraffe,", "hat,", "backpack", "umbrella", "shoe",
                "eyeglasses", "handbag", "tie", "suitcase", "frisbee", "skis",
                "snowboard", "sportsball", "kite", "baseballbat",
                "baseballglove", "skateboard", "surfboard", "tennisracket",
                "bottle", "plate", "wineglass", "cup", "fork", "knife", "spoon",
                "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                "carrot", "hotdog", "pizza", "donut", "cake", "chair", "couch",
                "pottedplant", "bed", "mirror", "diningtable", "window", "desk",
                "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard",
                "cellphone", "microwave", "oven", "toaster", "sink",
                "refrigerator", "blender", "book", "clock", "vase", "scissors",
                "teddybear", "hairdrier", "toothbrush", "hairbrush", "banner",
                "blanket", "branch", "bridge", "building-other", "bush",
                "cabinet", "cage", "cardboard", "carpet", "ceiling-other",
                "ceiling-tile", "cloth", "clothes", "clouds", "counter",
                "cupboard", "curtain", "desk-stuff", "dirt", "door-stuff",
                "fence", "floor-marble", "floor-other", "floor-stone",
                "floor-tile", "floor-wood", "flower", "fog", "food-other",
                "fruit", "furniture-other", "grass", "gravel", "ground-other",
                "hill", "house", "leaves", "light", "mat", "metal",
                "mirror-stuff", "moss", "mountain", "mud", "napkin", "net",
                "paper", "pavement", "pillow", "plant-other", "plastic",
                "platform", "playingfield", "railing", "railroad", "river",
                "road", "rock", "roof", "rug", "salad", "sand", "sea", "shelf",
                "sky-other", "skyscraper", "snow", "solid-other", "stairs",
                "stone", "straw", "structural-other", "table", "tent",
                "textile-other", "towel", "tree", "vegetable", "wall-brick",
                "wall-concrete", "wall-other", "wall-panel", "wall-stone",
                "wall-tile", "wall-wood", "water-other", "waterdrops",
                "window-blind", "window-other", "wood"]
