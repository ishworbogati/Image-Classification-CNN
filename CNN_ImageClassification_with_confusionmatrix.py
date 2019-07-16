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

training_set = train_datagen.flow_from_directory('E:/dataset/dataset/training_set',         # Give your own path*****************************
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('E:/dataset/dataset/test_set',                  # Give your own path*********************
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=1,
                         validation_data=test_set,
                         validation_steps=2000)

# saving the Trained-Model
classifier.save("E:/dataset/cats_vs_dogs_V1.h5")



import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from keras.models import load_model


batch_size= 32

img_row, img_height, img_depth = 64,64,3
model = load_model('E:/dataset/cats_vs_dogs_V1.h5')

class_labels = test_set.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())

nb_train_samples = 8000
nb_validation_samples = 2000

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(test_set, nb_validation_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)

target_names = list(class_labels.values())

plt.figure(figsize=(20,20))
cnf_matrix = confusion_matrix(test_set.classes, y_pred)

plt.imshow(cnf_matrix, interpolation='nearest')
plt.colorbar()
tick_marks = np.arange(len(classes))
_ = plt.xticks(tick_marks, classes, rotation=90)
_ = plt.yticks(tick_marks, classes)                    # Give your own path******************************



# Displaying our Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib
#Confution Matrix and Classification Report
Y_pred = model.predict_generator(test_set, nb_validation_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_set.classes, y_pred))
print('Classification Report')
target_names = list(class_labels.values())
print(classification_report(test_set.classes, y_pred, target_names=target_names))



# Testing a unknown i/p for Cat or Dog
import numpy as np
import keras
from keras.preprocessing import image
from keras.models import load_model

# ************************************************Give your own paths****************************************

classifier = load_model("E:/dataset/cats_vs_dogs_V1.h5")   # Load the trained model
test_image = image.load_img('E:/dataset/dataset/single_prediction/1.jpg',
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



