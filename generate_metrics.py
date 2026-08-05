import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# --- Config générale ---
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
N_CLASSES = len(CLASSES)

# Dossiers de données
TRAIN_DIR = "data/training/"
TEST_DIR = "data/testing/"


# --- PyTorch data & modèle ---
def get_torch_test_loader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return test_loader


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2_drop = torch.nn.Dropout2d(p=0.3)
        self.fc1 = torch.nn.Linear(64 * 56 * 56, 128)
        self.fc2 = torch.nn.Linear(128, N_CLASSES)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.conv2_drop(x)
        x = x.view(-1, 64 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def evaluate_torch():
    device = torch.device("cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load("jean_bayiha_model.torch", map_location=device))
    model.eval()

    test_loader = get_torch_test_loader()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=CLASSES, output_dict=True
    )

    return cm, report


# --- TensorFlow data & modèle ---
def get_tf_test_dataset(target_size=(96, 96), batch_size=32):
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        image_size=target_size,
        batch_size=batch_size,
        shuffle=False,
    )

    normalization_layer = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))
    return test_dataset


def get_tf_model():
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(96, 96, 3)
    )
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(56, activation="relu")(x)
    x = Dropout(0.3)(x)
    predictions = Dense(N_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights("jean_bayiha_model.weights.h5")
    return model


def evaluate_tf():
    model = get_tf_model()
    test_dataset = get_tf_test_dataset()

    all_preds = []
    all_labels = []

    for x_batch, y_batch in test_dataset:
        y_pred = model.predict(x_batch, verbose=0)
        preds = np.argmax(y_pred, axis=1)

        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=CLASSES, output_dict=True
    )

    return cm, report


# --- Fonctions de plot et sauvegarde ---
def plot_confusion_matrix(cm, classes, title, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annoter les cellules
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def main():
    # PyTorch
    cm_torch, report_torch = evaluate_torch()
    plot_confusion_matrix(
        cm_torch, CLASSES, "Confusion Matrix - PyTorch CNN", "static/confusion_torch.png"
    )
    with open("static/metrics_torch.json", "w") as f:
        json.dump(report_torch, f, indent=2)

    # TensorFlow
    cm_tf, report_tf = evaluate_tf()
    plot_confusion_matrix(
        cm_tf,
        CLASSES,
        "Confusion Matrix - TF MobileNetV2",
        "static/confusion_tf.png",
    )
    with open("static/metrics_tf.json", "w") as f:
        json.dump(report_tf, f, indent=2)

    print("Metrics and confusion matrices generated in static/.")


if __name__ == "__main__":
    main()
