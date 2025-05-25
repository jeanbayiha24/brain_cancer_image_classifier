import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import optimizers, losses
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Mean

class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, lr, wd, epochs, device):
        self.epochs = epochs
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, save=False, plot=False):
        self.model.train()
        self.train_acc = []
        self.train_loss = []
        for epoch in range(self.epochs):
            total_loss = 0
            total_correct = 0
            total_samples = 0

            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False)

            for batch in progress_bar:
                input_datas, labels = batch
                input_datas, labels = input_datas.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_datas)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # === Accuracy Calculation ===
                _, preds = outputs.max(1)  # Get the predicted class indices
                correct = (preds == labels).sum().item()
                total = labels.size(0)

                total_correct += correct
                total_samples += total
                total_loss += loss.item()

                batch_accuracy = 100.0 * correct / total
                average_accuracy = 100.0 * total_correct / total_samples
                average_loss = total_loss / total_samples

                progress_bar.set_postfix({
                    'Batch Acc': f'{batch_accuracy:.2f}%',
                    'Avg Acc': f'{average_accuracy:.2f}%',
                    'Loss': f'{average_loss:.4f}'
                })
            self.train_acc.append(average_accuracy)
            self.train_loss.append(average_loss)
        if save:
            torch.save(self.model.state_dict(), "jean_bayiha_model.torch") 
        if plot:
            self.plot_training_history()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for inputs, labels in tqdm(self.test_dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            _, preds = outputs.max(1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)  # Multiply by batch size

        avg_loss = total_loss / total_samples
        accuracy = 100.0 * total_correct / total_samples

        print(f"\nTest Accuracy: {accuracy:.2f}%  |  Test Loss: {avg_loss:.4f}")
        return accuracy, avg_loss

    def plot_training_history(self):
        epochs = range(1, len(self.train_loss) + 1)

        fig, ax1 = plt.subplots(figsize=(8, 5))

        color_loss = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color_loss)
        ax1.plot(epochs, self.train_loss, color=color_loss, label='Loss')
        ax1.tick_params(axis='y', labelcolor=color_loss)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color_acc = 'tab:red'
        ax2.set_ylabel('Accuracy (%)', color=color_acc)
        ax2.plot(epochs, self.train_acc, color=color_acc, label='Accuracy')
        ax2.tick_params(axis='y', labelcolor=color_acc)

        plt.title('Training Loss and Accuracy')
        fig.tight_layout()  # to prevent overlap
        plt.show()


    

#Tensorflow trainer
class TFTrainer:
    def __init__(self,model,train_data, test_data, lr, epochs):
        self.epochs = epochs
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizers.Adam(learning_rate=lr)
        self.loss_fn = losses.SparseCategoricalCrossentropy(from_logits=False)
        self.train_acc_metric = SparseCategoricalAccuracy()
        self.train_loss_metric = Mean()
        self.train_acc = []
        self.train_loss = []

    #@tf.function
    def train_step(self, x_batch_train, y_batch_train):
        with tf.GradientTape() as tape:
            y_pred = self.model(x_batch_train, training=True)
            loss_value = self.loss_fn(y_batch_train, y_pred)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        self.train_acc_metric.update_state(y_batch_train, y_pred)
        self.train_loss_metric.update_state(loss_value)

    def train(self, save=False, plot=False, verbose=True):
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            pbar = tqdm(self.train_data, desc=f"Training", unit="batch", leave=False)

            for x_batch_train, y_batch_train in pbar:
                self.train_step(x_batch_train, y_batch_train)
                if verbose :
                    current_acc = self.train_acc_metric.result().numpy()*100
                    current_loss = self.train_loss_metric.result().numpy()
                    pbar.set_postfix(acc= f"{current_acc:.2f}%", loss= f"{current_loss:.4f}")
            
            train_acc = self.train_acc_metric.result().numpy()*100#to have it in %
            train_loss = self.train_loss_metric.result().numpy()
            self.train_acc.append(train_acc)
            self.train_loss.append(train_loss)

            print(f"Epoch {epoch+1} - Accuracy: {train_acc: .2f}% | Loss: {train_loss:.4f}")

            self.train_acc_metric.reset_state()
            self.train_loss_metric.reset_state()

        if save:
            self.model.save_weights("jean_bayiha_model.weights.h5")
        if plot:
            self.plot_training_history()

    def evaluate(self):
        acc_metric = SparseCategoricalAccuracy()
        loss_metric = Mean()

        
        for x_batch_test, y_batch_test in self.test_data:
            y_pred = self.model(x_batch_test, training=False)
            loss_value = self.loss_fn(y_batch_test, y_pred)
                
            acc_metric.update_state(y_batch_test, y_pred)
            loss_metric.update_state(loss_value)
            
        accuracy = acc_metric.result().numpy()*100#to have it in %
        avg_loss = loss_metric.result().numpy()

        print(f"\nTest Accuracy: {accuracy: .2f}% | Test Loss: {avg_loss:.4f}")
        return accuracy, avg_loss
    
    def plot_training_history(self):
        epochs = range(1, len(self.train_loss) + 1)

        fig, ax1 = plt.subplots(figsize=(8, 5))

        color_loss = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color_loss)
        ax1.plot(epochs, self.train_loss, color=color_loss, label='Loss')
        ax1.tick_params(axis='y', labelcolor=color_loss)

        ax2 = ax1.twinx()
        color_acc = 'tab:red'
        ax2.set_ylabel('Accuracy (%)', color=color_acc)
        ax2.plot(epochs, self.train_acc, color=color_acc, label='Accuracy')
        ax2.tick_params(axis='y', labelcolor=color_acc)
        plt.title('Training Loss and Accuracy')
        fig.tight_layout()
        plt.show()