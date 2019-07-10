
# Import libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# initializing the CNN, Convolution , pooling
classifier = Sequential()


# step-1-Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# step-2-Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))      # chooses max value from 2x2 matrix filter

#  Second layer
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# step-3-Flatten
classifier.add(Flatten())       # converts matrix into linear vector

# step-4-Full Connection
classifier.add(Dense(units=128, activation='relu'))         # relu helps to reduces the hidden layers
classifier.add(Dense(units=32, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))        # sigmoid helps to predict between 2 classed


# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('E:/dataset/training_set',         # Give your own path*****************************
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('E:/dataset/test_set',                  # Give your own path*********************
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=1,
                         validation_data=test_set,
                         validation_steps=2000)

# saving the Trained-Model
classifier.save("E:/dataset/cats_vs_dogs_V1.h5")                    # Give your own path******************************


# Testing a unknown i/p for Cat or Dog
import numpy as np
import keras
from keras.preprocessing import image
from keras.models import load_model

# ************************************************Give your own paths****************************************

classifier = load_model("E:/dataset/cats_vs_dogs_V1.h5")   # Load the trained model
test_image = image.load_img('E:/dataset/single_prediction/cat_or_dog_2.jpg',
                            target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
    print(prediction)
else:
    prediction = 'cat'
    print(prediction)



