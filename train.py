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

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    # read in labels
    classes = []
    int2class = {}
    directories = os.listdir(TRAIN_DIR)  # Train
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

    """
    # use df to copy images to appropriate directories
    # populate Training directory
    for index, row in df.iterrows():
        image_filename = row["image_id"]
        image_filepath = os.path.join(IMAGES_DIR, image_filename)
        image = Image.open(image_filepath)

        if row["Black"] == 1:  # save in "Black"
            image.save(os.path.join(TRAIN_DIR, "Black\\"+image_filename))

        else:  # save in "Not Black"
            image.save(os.path.join(TRAIN_DIR, "Not Black\\"+image_filename))
    """

    # image generator
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255  # [0, 255] --> [0, 1]
    )

    train_data_gen = image_generator.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        color_mode="rgb",
        class_mode="binary",
        classes=classes,
        batch_size=BATCH_SIZE,
        shuffle=True
        #save_to_dir=TEMP_DIR  # temp
    )

    #next(train_data_gen)
    #quit()

    # ----- MODEL ----- #
    vgg16 = tf.keras.applications.vgg16.VGG16(
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
        include_top=False
    )

    for layer in vgg16.layers[:-4]:
        layer.trainable = False

    model = tf.keras.models.Sequential()
    model.add(vgg16)

    # ----- TRAIN ----- #

    # ----- EVALUATE ----- #
