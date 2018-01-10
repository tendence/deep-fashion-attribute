#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 23:02:33 2018

@author: jack
"""

import os
import os.path as ops
import numpy as np
from easydict import EasyDict as edict

_C=edict()

cfg=_C

_C.GPU_ID = 1
_C.TRAIN_BATCH_SIZE = 8
_C.TEST_BATCH_SIZE = 8
_C.TRIPLET_BATCH_SIZE = 32
_C.EXTRACT_BATCH_SIZE = 64
_C.TEST_BATCH_COUNT = 30
_C.NUM_WORKERS = 4
_C.LR = 0.001
_C.MOMENTUM = 0.5
_C.EPOCH = 10
_C.DUMPED_MODEL = "model_10_final.pth.tar"

_C.LOG_INTERVAL = 10
_C.DUMP_INTERVAL = 500
_C.TEST_INTERVAL = 100

_C.DATASET_BASE = r'../deep-fashion-dataset'
_C.ENABLE_INSHOP_DATASET = False
_C.INSHOP_DATASET_PRECENT = 0.8
_C.IMG_SIZE = 256
_C.CROP_SIZE = 224
_C.INTER_DIM = 512
_C.CATEGORIES = 20
_C.ATTRIBUTES = 1000
_C.N_CLUSTERS = 50
_C.COLOR_TOP_N = 10
_C.TRIPLET_WEIGHT = 2.0
_C.ENABLE_TRIPLET_WITH_COSINE = False  # Buggy when backward...
_C.COLOR_WEIGHT = 0.1
_C.DISTANCE_METRIC = ('euclidean', 'euclidean')
#_C.DISTANCE_METRIC = ('cosine', 'cosine')
_C.MODELS_FLODER= r'models_attr'
_C.FREEZE_PARAM = True

