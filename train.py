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
# Main
if __name__ == "__main__":
    # TF version
    print(f'TF version: {tf.__version__}')

    # read in labels
    classes = []
    int2class = {}
    directories = os.listdir(DATASETS_DIR)  # Test, Training
    for i in range(len(directories)):
        name = directories[i]
        classes.append(name)
        int2class[i] = name

    num_classes = len(classes)

    print(f'Classes: {classes}')
    print(f'Number of classes: {num_classes}')

    # read in CSV data
    df = pd.read_csv(CELEB_DATASET_CSV)
    df = df[:12000]  # first 12k images
    df = df[["image_id", "Black"]]
