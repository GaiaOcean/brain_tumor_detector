import cv2 #opencv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Dense, Flatten
from sklearn.model_selection import train_test_split

img_dir = 'dataset/'
dataset = []
labels = []
healthy_img = os.listdir(img_dir + 'no/') #creates a list of all the images labled as no
tumor_img = os.listdir(img_dir + 'yes/')

#note: opencv reads the image in BGR format so we need to convert the img to RGB
for index, img in enumerate(healthy_img):
    if img.split('.')[1] == 'jpg':
        image = cv2.imread(img_dir + 'no/' + img) #reading each image
        image = Image.fromarray(image) #convert the img to RBG
        image = image.resize((64,64)) 
        dataset.append(np.array(image))
        labels.append(0) #the healthy images will be catagorized as 0

for index, img in enumerate(tumor_img):
    if img.split('.')[1] == 'jpg':
        image = cv2.imread(img_dir + 'yes/' + img) #reading each image
        image = Image.fromarray(image) #convert the img to RBG
        image = image.resize((64,64)) 
        dataset.append(np.array(image))
        labels.append(1) #the non healthy images will be catagorized as 1

dataset = np.array(dataset)
labels = np.array(labels)

#spliting the dataset into 80% for the training and 20% for the test
x_train,x_test,y_train,y_test = train_test_split(dataset,labels, test_size=0.2,random_state=0)


x_train = normalize(x_train,axis=1)
x_test = normalize(x_test,axis=1)

model = Sequential()
model.add(Conv2D(32,(3,3),input_shape = (64,64,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),kernel_initializer="he_uniform"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),kernel_initializer="he_uniform"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=16,verbose=True,epochs=10,validation_data=(x_test,y_test),shuffle=False)

model.save('tumor_detector.h5')