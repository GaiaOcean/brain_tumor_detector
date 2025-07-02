import cv2 
from PIL import Image
from keras.models import load_model
import numpy as np

model = load_model('tumor_detector.h5')

val_img1 = cv2.imread('C:\\Users\\User\\Downloads\\tumor_dector\\dataset\\pred\\pred56.jpg')

img = Image.fromarray(val_img1)
img = img.resize((64,64))
img = np.array(img)
img = np.expand_dims(img,axis=0)
pred = model.predict(img)
print(pred)