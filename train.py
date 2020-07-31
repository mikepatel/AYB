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
        validation_split=0.1,
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

    #for layer in vgg16.layers[:-4]:
    #    layer.trainable = False
    vgg16.trainable = False

    model = tf.keras.models.Sequential()
    model.add(vgg16)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        units=1024,
        activation=tf.keras.activations.relu
    ))
    model.add(tf.keras.layers.Dense(
        units=num_classes,
        activation=tf.keras.activations.sigmoid
    ))

    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    model.summary()

    # ----- TRAIN ----- #
    history = model.fit(
        x=train_data_gen,
        epochs=NUM_EPOCHS
    )

    # plot accuracy
    plt.scatter(range(1, NUM_EPOCHS+1), history.history["accuracy"], label="accuracy", s=500)
    #plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(os.getcwd(), "training"))

    # ----- EVALUATE ----- #
    # test black
    test_black_dir = os.path.join(TEST_DIR, "Black")

    # test not black
    test_not_black_dir = os.path.join(TEST_DIR, "Not Black")

    # convert image to array
    dirs = [test_black_dir, test_not_black_dir]
    for d in dirs:
        images = os.listdir(d)
        for image in images:
            image_filepath = os.path.join(d, image)

            image = Image.open(image_filepath)

            image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            image = np.array(image)
            image = image.reshape(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
            image = image / 255.0
            image = np.expand_dims(image, 0)

            prediction = model.predict(image)
            print()
            #print(image_filepath)
            print(int2class[int(np.argmax(prediction))])
