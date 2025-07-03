# Brain Tumor Detector with Grad-CAM

A deep learning web application built with Flask for detecting brain tumors in medical images. This project uses a custom-trained convolutional neural network (CNN) for binary classification (tumor vs. no tumor), trained from scratch on a labeled medical image dataset.

The app integrates Grad-CAM visualizations to highlight important regions of the image that influenced the model’s prediction, providing transparency and interpretability in a clinical context.

Image preprocessing (resizing and normalization) is applied to ensure consistent model input. Users can upload their own medical images via a simple web interface and receive both a prediction and a Grad-CAM heatmap overlay.
