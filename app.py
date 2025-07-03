from flask import Flask, request, render_template, jsonify
import os
import cv2
from keras.models import load_model
from grad_cam import grad_cam_generator, normalize_img, get_score
model = load_model('tumor_detector.h5')

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. Web route – user uploads an image through a form
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        img_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(img_path)

        preprocessed_image = normalize_img(img_path)
        gradcam_img = grad_cam_generator(model, preprocessed_image)

        # Save the gradcam image to serve on HTML
        gradcam_path = os.path.join(UPLOAD_FOLDER, 'gradcam_' + image.filename)
        cv2.imwrite(gradcam_path, cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR))

        prediction = model.predict(preprocessed_image)[0][0]
        label = "Tumor detected" if prediction >= 0.5 else "No tumor detected"

        return render_template('result.html',
                               prediction=label,
                               original=img_path,
                               gradcam=gradcam_path)

    return render_template('index.html')


# 2. REST API route – programmatic access
@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    img_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(img_path)

    preprocessed_image = normalize_img(img_path)
    gradcam_img = grad_cam_generator(model, preprocessed_image)

    gradcam_path = os.path.join(UPLOAD_FOLDER, 'gradcam_' + image.filename)
    cv2.imwrite(gradcam_path, cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR))

    prediction = model.predict(preprocessed_image)[0][0]
    label = "Tumor detected" if prediction >= 0.5 else "No tumor detected"

    return jsonify({
        'prediction': label,
        'original_image_url': img_path,
        'grad_cam_image_url': gradcam_path
    })

if __name__ == '__main__':
    app.run(debug=True)
