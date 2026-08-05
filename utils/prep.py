from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize to match ResNet input
        transforms.Grayscale(num_output_channels=3),  # convert to 3 channels
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)  # normalize RGB channels
    ])
    training = datasets.ImageFolder('data/training/', transform=transform)
    test = datasets.ImageFolder('data/testing/', transform=transform)

    train_loader = DataLoader(training, batch_size=64, shuffle=True)
    test_loader = DataLoader(test, batch_size=64, shuffle=False)

    return train_loader, test_loader


def get_tf_data(batch_size=32, target_size=(96,96)):
    #data augmentation on training set and normalization
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "data/training/",
        labels='inferred',
        label_mode='int',
        image_size=target_size,
        batch_size=batch_size,
        shuffle=True
    )

    # Chargement des données de test
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "data/testing/",
        labels='inferred',
        label_mode='int',
        image_size=target_size,
        batch_size=batch_size,
        shuffle=False
    )

    # normalization
    normalization_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)

    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    return train_dataset, test_dataset