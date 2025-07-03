import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
import cv2 
import keras
from PIL import Image


model = keras.models.load_model('tumor_detector.h5')

def get_img_path(img_path: str) -> str:
    return img_path

def normalize_img(img_path: str) -> np.ndarray:
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # optional: convert BGR to RGB
    image = Image.fromarray(image)
    image = image.resize((64, 64))
    image = np.asarray(image).astype(np.float32) / 255.0  # normalize and convert dtype
    image = np.expand_dims(image, axis=0)
    return image

#gets the raw output of the model throught the paramiter and outputs tensor of floating-point numbers
def get_score(model_output):
    return model_output[0]


def grad_cam_generator(model,image):
    linear = ReplaceToLinear()
    gradcam = Gradcam(model, model_modifier= linear, clone=True)
    cam = gradcam(get_score, image, penultimate_layer=-1) 
    heatmap = np.uint8(255 * cam[0])
    img = image[0]
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_img = heatmap * 0.4 + (img * 255).astype(np.uint8)
    model_output = model.predict(image)
    score = get_score(model_output)

    # plt.imshow(heatmap_img.astype('uint8'))
    # plt.axis('off')
    # if score >= 0.5:
    #     plt.title("Tumor detected", fontsize=14, color='red')
    # else:
    #     plt.title("TNo tumor detected", fontsize=14, color='green')
    # plt.show()
    
    return heatmap_img.astype(np.uint8)