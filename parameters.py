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

import tensorflow as tf


################################################################################
# Image dimension
IMAGE_WIDTH = 128
IMAGE_HEIGH = 128
IMAGE_CHANNELS = 3

NUM_EPOCHS = 1
BATCH_SIZE = 32

# directories
DATASET_DIR = os.path.join(os.getcwd(), "data\\datasets")
