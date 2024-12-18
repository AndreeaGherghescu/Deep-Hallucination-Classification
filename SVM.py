import numpy as np
import pickle
import cv2 as cv
from sklearn import svm
import keras

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
            train_load_images.append(cv.imread(TRAIN_IMAGES_PATH + line.split(',')[0]))
            train_load_labels.append(line.split(',')[1])

    # Read validation images and labels
    with open("./data/validation.txt", 'r') as file:
        file.readline()
        for line in file.readlines():
            validation_load_images.append(cv.imread(VALIDATION_IMAGES_PATH + line.split(',')[0]))
            validation_load_labels.append(line.split(',')[1])

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


loaded_train_images = []
loaded_validation_images = []
loaded_test_images = []

train_images = []
validation_images = []
test_images = []

train_images_labels = []
validation_images_labels = []
test_images_predictions = []

if LOAD_PICKLE:
    with open("./data/pickle/train_images", 'rb') as pickle_save_file:
        loaded_train_images = np.array(pickle.load(pickle_save_file))

    with open("./data/pickle/validation_images", 'rb') as pickle_save_file:
        loaded_validation_images = np.array(pickle.load(pickle_save_file))

    with open("./data/pickle/train_labels", 'rb') as pickle_save_file:
        train_images_labels = np.array(pickle.load(pickle_save_file))

    with open("./data/pickle/validation_labels", 'rb') as pickle_save_file:
        validation_images_labels = np.array(pickle.load(pickle_save_file))


for image in loaded_train_images:
    train_images.append(cv.cvtColor(image, cv.COLOR_BGR2GRAY).flatten())

for image in loaded_validation_images:
    validation_images.append(cv.cvtColor(image, cv.COLOR_BGR2GRAY).flatten())

# 47%
classifier = svm.SVC()
classifier.fit(train_images, train_images_labels)

# Read test images
test_images_names = []

with open("./data/test.txt", 'r') as file:
    file.readline()
    for line in file.readlines():
        loaded_test_images.append(cv.imread(TEST_IMAGES_PATH + line.split()[0]))
        test_images_names.append(line.split()[0])

# Prepare test images data
for image in loaded_test_images:
    test_images.append(cv.cvtColor(image, cv.COLOR_BGR2GRAY).flatten())

# Save submission
if SAVE_SUBMISSION:
    with open("submission-andreea-initial.txt", 'w') as submission_file:
        submission_file.write('id,label\n')

        for i, image in enumerate(test_images):
            prediction = classifier.predict([image])
            submission_file.write(test_images_names[i] + ',' + str(prediction)[2] + '\n')
