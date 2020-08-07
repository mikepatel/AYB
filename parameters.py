"""
Michael Patel
July 2020

Project description:

File description:
"""
################################################################################
# Imports
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf


################################################################################
# Image dimension
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNELS = 3

NUM_EPOCHS = 100
BATCH_SIZE = 64

# directories and files
DATA_DIR = os.path.join(os.getcwd(), "data")
CELEB_DATASET_CSV = os.path.join(DATA_DIR, "list_attr_celeba.csv")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
IMAGES_DIR = os.path.join(DATA_DIR, "celeba")

TRAIN_DIR = os.path.join(DATASETS_DIR, "Train")
TEST_DIR = os.path.join(DATASETS_DIR, "Test")
VAL_DIR = os.path.join(DATASETS_DIR, "Validation")
SAVED_MODEL_DIR = os.path.join(os.getcwd(), "saved_model")

TEMP_DIR = os.path.join(os.getcwd(), "temp")
