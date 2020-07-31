"""
Michael Patel
June 2020

Project description:

File description:
"""
################################################################################
# Imports
from parameters import *


################################################################################
def build_model(num_classes):
    m = tf.keras.Sequential()

    # Convolution 1
    m.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
        activation=tf.keras.activations.relu
    ))

    # Convolution 2
    m.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation=tf.keras.activations.relu
    ))

    # Flatten
    m.add(tf.keras.layers.Flatten())

    # Dense
    m.add(tf.keras.layers.Dense(
        units=128,
        activation=tf.keras.activations.relu
    ))

    # Output
    m.add(tf.keras.layers.Dense(
        units=num_classes
    ))

    return m
