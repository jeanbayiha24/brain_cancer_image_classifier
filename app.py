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
    "Tensorflow Model": "tensorflow",
}

# Historique des prédictions (max 5)
prediction_history = []  # liste de dicts

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
    

import json

@app.route("/metrics")
def metrics():
    # Charger les JSON générés
    try:
        with open("static/metrics_torch.json") as f:
            metrics_torch = json.load(f)
    except FileNotFoundError:
        metrics_torch = None

    try:
        with open("static/metrics_tf.json") as f:
            metrics_tf = json.load(f)
    except FileNotFoundError:
        metrics_tf = None

    return render_template(
        "metrics.html",
        metrics_torch=metrics_torch,
        metrics_tf=metrics_tf,
        classes=classes,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    global prediction_history

    #print("\n--- New request ---")
    prediction = None
    probs = None
    selected_model = None
    confidence = None

    if request.method == "POST":
        #print("Request is POST")
        selected_model = request.form.get("model")
        file = request.files.get("image")
        #print("Selected model:", selected_model)
        #print("File present:", bool(file))

        if file and selected_model:
            img = Image.open(file)

            if selected_model == "pytorch":
                img_tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    outputs = model_torch(img_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                    predicted_idx = int(np.argmax(probabilities))

            elif selected_model == "tensorflow":
                img_tensor = transform_tf(img)
                outputs = model_tf.predict(img_tensor)
                probabilities = outputs[0]
                predicted_idx = int(np.argmax(probabilities))

            confidence = float(probabilities[predicted_idx])
            threshold = 0.7
            #print("Predicted idx:", predicted_idx)
            #print("Confidence:", confidence)

            if confidence < threshold:
                prediction = (
                    f"Uncertain prediction "
                    f"(most likely class : {classes[predicted_idx]}, "
                    f"confidence {confidence:.2f})."
                )
            else:
                prediction = classes[predicted_idx]

            probs = sorted(
                [(cls, float(p)) for cls, p in zip(classes, probabilities)],
                key=lambda x: x[1],
                reverse=True,
            )

            prediction_entry = {
                "model": "PyTorch CNN" if selected_model == "pytorch" else "TF MobileNetV2",
                "prediction": prediction,
                "top_class": classes[predicted_idx],
                "confidence": confidence,
                "probs": {cls: float(p) for cls, p in zip(classes, probabilities)},
            }
            #print("New prediction entry:", prediction_entry)

            prediction_history.insert(0, prediction_entry)
            prediction_history = prediction_history[:5]
            #print("History length:", len(prediction_history))

    #print("History before render:", prediction_history)

    return render_template(
        "index.html",
        prediction=prediction,
        probs=probs,
        model_names=models.keys(),
        models=models,
        selected_model=selected_model,
        classes=classes,
        confidence=confidence,
        prediction_history=prediction_history,
    )
    
if __name__== "__main__":
    app.run()
