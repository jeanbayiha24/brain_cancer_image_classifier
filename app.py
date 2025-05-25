from flask import Flask, render_template, request
from PIL import Image
import torch
import torchvision.transforms as transforms
import tensorflow as tf
import numpy as np
import os
from models.cnn import CNN, get_tensorflow_model

app = Flask(__name__)

# we load our 2 models
model_torch = CNN()
model_torch.load_state_dict(torch.load("jean_bayiha_model.torch", map_location = "cpu"))
model_torch.eval()

model_tf = get_tensorflow_model()
model_tf.load_weights("jean_bayiha_model.weights.h5")
print(model_tf.input_shape)

models = {
    "PyTorch Model": "pytorch",
    "Tensorflow Model": "tensorflow"
}

# dataset classes
classes = ["glioma","meningioma","notumor","pituitary"]

#image preprocessing for pytorch model
transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize to match ResNet input
        transforms.Grayscale(num_output_channels=3),  # convert to 3 channels
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)  # normalize RGB channels
])

#image preprocessing for tensorflow model
def transform_tf(image):
    image = image.convert("RGB")  # conert image in RGB format
    image = image.resize((96,96))
    img_array = np.array(image) / 255.0 #normalize
    return np.expand_dims(img_array, axis=0)


@app.route("/", methods=["GET","POST"])
def index():
    prediction = None
    if request.method == "POST":
        model_type = request.form["model"]
        file = request.files["image"]
        if file:
            img =Image.open(file)

            if model_type =="pytorch":
                img_tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    outputs = model_torch(img_tensor)
                    _ , predicted = outputs.max(1) # the maximum prob between the 4 classes
                    prediction = classes[predicted.item()]
           
            elif model_type == "tensorflow":
                img_tensor = transform_tf(img)
                outputs = model_tf.predict(img_tensor)
                predicted = np.argmax(outputs, axis=1)[0]
                prediction = classes[predicted]
    return render_template("index.html", prediction=prediction, model_names=models.keys(), models=models)


if __name__== "__main__":
    app.run()