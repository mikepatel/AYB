"""
Michael Patel
July 2020

Project description:

File description:
"""
################################################################################
# Imports
from parameters import *


################################################################################
# Main
if __name__ == "__main__":
    # data labels
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

    # load trained model
    model = tf.keras.models.load_model(SAVED_MODEL_DIR)
    #model.summary()

    # open webcam
    capture = cv2.VideoCapture(0)
    while True:
        # capture frame by frame
        ret, frame = capture.read()

        # preprocess image
        image = frame
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # save frame image
        filepath = os.path.join(os.getcwd(), "predicted.jpg")
        cv2.imwrite(filepath, image)

        # crop webcam image
        y, x, channels = image.shape
        left_x = int(x*0.25)
        right_x = int(x*0.75)
        top_y = int(y*0.25)
        bottom_y = int(y*0.75)
        image = image[top_y:bottom_y, left_x:right_x]

        # resize image
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        mod_image = image

        # array and rescale
        image = np.array(image)
        image = image / 255.0
        image = np.expand_dims(image, 0)

        # make prediction
        prediction = model.predict(image)
        prediction = int(np.argmax(prediction))
        prediction = int2class[prediction]
        print(prediction)

        # display frame
        #cv2.imshow("", frame)
        cv2.imshow("", mod_image)

        # label webcam image with predicted label
        webcam_image = Image.open(filepath)
        draw = ImageDraw.Draw(webcam_image)
        font = ImageFont.truetype("arial.ttf", 32)
        draw.text((0, 0), prediction, font=font)
        webcam_image.save(filepath)

        # continuous stream, escape key
        if cv2.waitKey(1) == 27:
            break

    # release capture
    capture.release()
    cv2.destroyAllWindows()
