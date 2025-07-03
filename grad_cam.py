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

def normalize_img(img_path: str) -> str:
    if img_path.split('.')[1] == 'jpg':
        image = cv2.imread(img_path)
        image = Image.fromarray(image)
        image = image.resize((64,64))
        image = np.expand_dims(image,axis=0)

        return image
    else:
        raise ValueError("Unsupported image format. Convert the image to jpg")

#gets the raw output of the model throught the paramiter and outputs tensor of floating-point numbers
def get_score(model_output):
    return model_output[:,0]

def grad_cam_generator(model,model_output,image):
    score = get_score(model_output)
    linear = ReplaceToLinear()
    gradcam = Gradcam(model, model_modifier= linear, clone=True)
    cam = gradcam(score, image, penultimate_layer=-1) 
    heatmap = np.uint8(255 * cam[0])
    img = image[0]
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + (img * 255).astype(np.uint8)

    plt.imshow(superimposed_img.astype('uint8'))
    plt.axis('off')
    plt.title('Brain Scan')
    plt.show()
    
    return superimposed_img