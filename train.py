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

    # how many rows have Black=1?
    print(f'Number of rows with Black=1: {len(df.loc[df["Black"] == 1])}')

    # use df to copy images to appropriate directories
    #image_files = os.listdir(IMAGES_DIR)
    for index, row in df.iterrows():
        if row["Black"] == 1:
            image_filepath = os.path.join(IMAGES_DIR, row["image_id"])
            print(image_filepath)
            image = Image.open(image_filepath)
            
