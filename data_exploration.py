import os

import torch
from collections import Counter
import matplotlib.pyplot as plt

def count_labels(data_path):

    images, labels, _ = torch.load(data_path, weights_only=False)
    label_counts = Counter(labels.numpy())
    return label_counts

def plot_label_distribution(train_counts, val_counts, label_names):
    labels = list(train_counts.keys())
    train_values = [train_counts[label] for label in labels]
    val_values = [val_counts[label] for label in labels]

    x = range(len(labels))

    plt.figure(figsize=(10, 6))
    plt.bar(x, train_values, width=0.2, label='Train', align='center')
    plt.bar([p + 0.2 for p in x], val_values, width=0.2, label='Validation', align='center')

    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title('Label Distribution Across Train and Validation Sets')
    plt.xticks([p + 0.2 for p in x], [label_names[label] for label in labels])
    plt.legend()
    plt.grid(True)
    plt.show()

def load_data(data_path):
    images, labels, _ = torch.load(data_path, weights_only=False)
    return images, labels

def display_images(images, labels, label_names):
    plt.figure(figsize=(14, 8))
    for i, label_name in enumerate(label_names):
        idx = (labels == i).nonzero(as_tuple=True)[0][0].item()
        img = images[idx].squeeze().numpy()
        plt.subplot(2, 4, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(label_name)
        plt.axis('off')
    plt.show()

def main():

    train_path = 'processed_data/train_data.pt'
    val_path = 'processed_data/val_data.pt'

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("[Error] Processed train/val data not found.")
        return
    print("[Info] Loading training & validation data from .pt files...")
    label_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    images, labels = load_data(train_path)
    display_images(images, labels, label_names)
    print("[Info] Displayed sample images from the training set.")

    train_counts = count_labels(train_path)
    val_counts = count_labels(val_path)
    plot_label_distribution(train_counts, val_counts, label_names)
    print("[Info] Displayed label distribution across train and validation sets.")


if __name__ == "__main__":
    main()