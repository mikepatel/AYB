"""
Michael Patel
July 2020

Project description:

File description:
"""
################################################################################
# Imports
import os
import pandas as pd
from PIL import Image

import tensorflow as tf


################################################################################
# Image dimension
IMAGE_WIDTH = 128
IMAGE_HEIGH = 128
IMAGE_CHANNELS = 3

NUM_EPOCHS = 1
BATCH_SIZE = 32

# directories and files
DATA_DIR = os.path.join(os.getcwd(), "data")
CELEB_DATASET_CSV = os.path.join(DATA_DIR, "list_attr_celeba.csv")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
IMAGES_DIR = os.path.join(DATA_DIR, "celeba")
TRAIN_DIR = os.path.join(DATASETS_DIR, "Train")
TEST_DIR = os.path.join(DATASETS_DIR, "Test")
