import numpy as np
import pickle
import cv2 as cv
import time
from tensorflow import keras
from keras.callbacks import ReduceLROnPlateau
from keras_preprocessing.image import ImageDataGenerator

LOAD_DATA = False
SAVE_PICKLE = False
LOAD_PICKLE = True

SAVE_SUBMISSION = True

TRAIN_IMAGES_PATH = './data/train/'
VALIDATION_IMAGES_PATH = './data/validation/'
TEST_IMAGES_PATH = './data/test/'

train_load_images = []
train_load_labels = []

validation_load_images = []
validation_load_labels = []


if LOAD_DATA:

    # Read train images and labels
    with open("./data/train.txt", 'r') as file:
        file.readline()
        for line in file.readlines():
            img = cv.imread(TRAIN_IMAGES_PATH + line.split(',')[0])
            img = (img - np.mean(img)) / np.std(img)
            train_load_images.append(img)
            train_load_labels.append(int(line.split(',')[1]))

    # Read validation images and labels
    with open("./data/validation.txt", 'r') as file:
        file.readline()
        for line in file.readlines():
            img = cv.imread(VALIDATION_IMAGES_PATH + line.split(',')[0])
            img = (img - np.mean(img)) / np.std(img)
            validation_load_images.append(img)
            validation_load_labels.append(int(line.split(',')[1]))

    # Save loaded data to pickle
    if SAVE_PICKLE:
        with open("./data/pickle/train_images", 'wb') as pickle_save_file:
            pickle.dump(train_load_images, pickle_save_file)

        with open("./data/pickle/validation_images", 'wb') as pickle_save_file:
            pickle.dump(validation_load_images, pickle_save_file)

        with open("./data/pickle/train_labels", 'wb') as pickle_save_file:
            pickle.dump(train_load_labels, pickle_save_file)

        with open("./data/pickle/validation_labels", 'wb') as pickle_save_file:
            pickle.dump(validation_load_labels, pickle_save_file)


# Initialize arrays
loaded_train_images = []
loaded_validation_images = []
loaded_test_images = []

train_images = []
validation_images = []
test_images = []

train_images_labels = []
validation_images_labels = []
test_images_predictions = []

# Load data from pickle if setting is set to True
if LOAD_PICKLE:
    with open("./data/pickle/train_images", 'rb') as pickle_save_file:
        loaded_train_images = np.array(pickle.load(pickle_save_file))

    with open("./data/pickle/validation_images", 'rb') as pickle_save_file:
        loaded_validation_images = np.array(pickle.load(pickle_save_file))

    with open("./data/pickle/train_labels", 'rb') as pickle_save_file:
        train_images_labels = np.array(pickle.load(pickle_save_file))

    with open("./data/pickle/validation_labels", 'rb') as pickle_save_file:
        validation_images_labels = np.array(pickle.load(pickle_save_file))


# Load images from pickle array to normal list
for image in loaded_train_images:
    train_images.append(image)

for image in loaded_validation_images:
    validation_images.append(image)

# Transform arrays from normal lists to numpy arrays
train_images = np.array(train_images)
validation_images = np.array(validation_images)

# Initialize Image augmentation generator
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True,
    zoom_range=0.1,
)

# Train the image augmentation
datagen.fit(train_images)

# Define model
classifier = keras.Sequential([
    keras.layers.Conv2D(64, (5, 5), input_shape=(16, 16, 3), activation='relu'),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Flatten(),
    keras.layers.Dense(240, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(120, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(7, activation='softmax')
])


# Hyperparameter tuning - Reducing learning rate on plateau, optimizer Adam, low learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=5, factor=0.2, min_lr=0.000001)

# Compile model and fit on training data validate on validation images
classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
classifier.fit(datagen.flow(train_images, train_images_labels, batch_size=8), epochs=50, batch_size=8, validation_data=datagen.flow(validation_images, validation_images_labels, batch_size=8), verbose=1, callbacks=[reduce_lr])

# Read test images
test_images_names = []

# Read test file
with open("./data/test.txt", 'r') as file:
    file.readline()
    for line in file.readlines():
        img = cv.imread(TEST_IMAGES_PATH + line.split()[0])
        img = (img - np.mean(img)) / np.std(img)
        loaded_test_images.append(img)
        test_images_names.append(line.split()[0])

# Prepare test images data
for image in loaded_test_images:
    test_images.append(image)

# test_images = np.expand_dims(test_images, axis=3)
test_images = np.array(test_images)

# Save submission
if SAVE_SUBMISSION:
    with open("./submissions/submission" + str(time.time()) + ".txt", 'w') as submission_file:
        submission_file.write('id,label\n')
        predictions = classifier.predict(test_images)

        for i, prediction in enumerate(predictions):
            submission_file.write(test_images_names[i] + ',' + str(np.argmax(prediction)) + '\n')
