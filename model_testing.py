import cv2 
from PIL import Image
from keras.models import load_model
import numpy as np
from grad_cam import grad_cam_generator, normalize_img, get_score
model = load_model('tumor_detector.h5')

img_path = 'C:\\Users\\User\\Downloads\\tumor_dector\\dataset\\pred\\pred58.jpg'
input_img = normalize_img(img_path)


heatmap_img = grad_cam_generator(model,input_img)




# img = Image.fromarray(val_img1)
# img = img.resize((64,64))
# img = np.array(img)
# img = np.expand_dims(img,axis=0)
# pred = model.predict(img)
# print(pred)