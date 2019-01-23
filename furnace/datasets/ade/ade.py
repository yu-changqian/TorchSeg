#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2017/12/16 下午8:41
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : ade.py

import os.path as osp
import numpy as np
import scipy.io as sio
import time
import cv2

import torch

from datasets.BaseDataset import BaseDataset


class ADE(BaseDataset):
    def _fetch_data(self, img_path, gt_path, dtype=np.float32):
        img = self._open_image(img_path)
        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=dtype)

        return img, gt

    @staticmethod
    def _process_item_names(item):
        item = item.strip()
        img_name = item
        gt_name = item.split('.')[0] + ".png"

        return img_name, gt_name

    @classmethod
    def get_class_colors(*args):
        color_list = sio.loadmat(osp.join('.', 'color150.mat'))
        color_list = color_list['colors']
        color_list = color_list[:, ::-1, ]
        color_list = np.array(color_list).astype(int).tolist()
        color_list.insert(0, [0, 0, 0])
        return color_list

    @classmethod
    def get_class_names(*args):
        return ['wall', 'building, edifice', 'sky',
                'floor, flooring', 'tree', 'ceiling', 'road, route',
                'bed ', 'windowpane, window ',
                'grass', 'cabinet', 'sidewalk, pavement',
                'person, individual, someone, somebody, mortal, soul',
                'earth, ground', 'door, double door', 'table',
                'mountain, mount', 'plant, flora, plant life',
                'curtain, drape, drapery, mantle, pall', 'chair',
                'car, auto, automobile, machine, motorcar', 'water',
                'painting, picture', 'sofa, couch, lounge', 'shelf',
                'house',
                'sea', 'mirror', 'rug, carpet, carpeting', 'field',
                'armchair', 'seat', 'fence, fencing', 'desk',
                'rock, stone',
                'wardrobe, closet, press', 'lamp',
                'bathtub, bathing tub, bath, tub', 'railing, rail',
                'cushion', 'base, pedestal, stand', 'box',
                'column, pillar',
                'signboard, sign',
                'chest of drawers, chest, bureau, dresser',
                'counter', 'sand', 'sink', 'skyscraper',
                'fireplace, hearth, open fireplace',
                'refrigerator, icebox', 'grandstand, covered stand',
                'path',
                'stairs, steps', 'runway',
                'case, display case, showcase, vitrine',
                'pool table, billiard table, snooker table',
                'pillow',
                'screen door, screen', 'stairway, staircase',
                'river',
                'bridge, span', 'bookcase', 'blind, screen',
                'coffee table, cocktail table',
                'toilet, can, commode, crapper, pot, potty, stool, throne',
                'flower', 'book', 'hill', 'bench', 'countertop',
                'stove, kitchen stove, range, kitchen range, cooking stove',
                'palm, palm tree', 'kitchen island',
                'computer, computing machine, computing device, data processor, electronic computer, information processing system',
                'swivel chair', 'boat', 'bar', 'arcade machine',
                'hovel, hut, hutch, shack, shanty',
                'bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle',
                'towel', 'light, light source', 'truck, motortruck',
                'tower',
                'chandelier, pendant, pendent',
                'awning, sunshade, sunblind',
                'streetlight, street lamp',
                'booth, cubicle, stall, kiosk',
                'television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box',
                'airplane, aeroplane, plane', 'dirt track',
                'apparel, wearing apparel, dress, clothes', 'pole',
                'land, ground, soil',
                'bannister, banister, balustrade, balusters, handrail',
                'escalator, moving staircase, moving stairway',
                'ottoman, pouf, pouffe, puff, hassock',
                'bottle', 'buffet, counter, sideboard',
                'poster, posting, placard, notice, bill, card',
                'stage', 'van', 'ship', 'fountain',
                'conveyer belt, conveyor belt, conveyer, conveyor, transporter',
                'canopy',
                'washer, automatic washer, washing machine',
                'plaything, toy',
                'swimming pool, swimming bath, natatorium',
                'stool', 'barrel, cask', 'basket, handbasket',
                'waterfall, falls', 'tent, collapsible shelter',
                'bag',
                'minibike, motorbike', 'cradle', 'oven', 'ball',
                'food, solid food', 'step, stair',
                'tank, storage tank',
                'trade name, brand name, brand, marque',
                'microwave, microwave oven', 'pot, flowerpot',
                'animal, animate being, beast, brute, creature, fauna',
                'bicycle, bike, wheel, cycle ', 'lake',
                'dishwasher, dish washer, dishwashing machine',
                'screen, silver screen, projection screen',
                'blanket, cover', 'sculpture', 'hood, exhaust hood',
                'sconce',
                'vase', 'traffic light, traffic signal, stoplight',
                'tray',
                'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin',
                'fan', 'pier, wharf, wharfage, dock', 'crt screen',
                'plate', 'monitor, monitoring device',
                'bulletin board, notice board', 'shower',
                'radiator',
                'glass, drinking glass', 'clock', 'flag']
