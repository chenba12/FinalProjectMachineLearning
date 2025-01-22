import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

def load_data(data_path):
    images, labels, _ = torch.load(data_path, weights_only=False)
    return images, labels

def prepare_data(images, labels):
    images = images.view(images.size(0), -1).numpy()
    labels = labels.numpy()
    return images, labels

def train_decision_tree(train_images, train_labels):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(train_images, train_labels)
    return clf

def evaluate_model(clf, val_images, val_labels):
    val_preds = clf.predict(val_images)
    accuracy = accuracy_score(val_labels, val_preds)
    report = classification_report(val_labels, val_preds, digits=3)
    return accuracy, report, val_preds

def plot_confusion_matrix(val_labels, val_preds, output_path, class_names):
    cm = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_decision_tree(clf, feature_names, class_names, output_path):
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True, fontsize=8, rounded=True)
    plt.title("Decision Tree Visualization")
    plt.savefig(output_path)
    plt.close()

def main():
    train_path = 'processed_data/train_data.pt'
    val_path = 'processed_data/val_data.pt'

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("[Error] Processed train/val data not found.")
        return

    print("[Info] Loading training & validation data from .pt files...")
    train_images, train_labels = load_data(train_path)
    val_images, val_labels = load_data(val_path)

    train_images, train_labels = prepare_data(train_images, train_labels)
    val_images, val_labels = prepare_data(val_images, val_labels)

    print("[Info] Training decision tree model...")
    clf = train_decision_tree(train_images, train_labels)

    print("[Info] Evaluating model on validation set...")
    val_accuracy, val_report, val_preds = evaluate_model(clf, val_images, val_labels)

    print("\n======================================")
    print(" Validation Classification Report:")
    print("======================================")
    print(val_report)
    print(f"[Summary] Validation Accuracy: {val_accuracy*100:.2f}%")

    # Create folders
    os.makedirs("images_results", exist_ok=True)
    os.makedirs("results", exist_ok=True)


    # Plot Confusion Matrix
    class_names = [str(i) for i in np.unique(train_labels)]  # Assuming class labels are integers
    plot_confusion_matrix(val_labels, val_preds, "image_results/confusion_matrix.png", class_names)
    print("[Info] Confusion Matrix plot saved.")

    # Visualize Decision Tree
    plot_decision_tree(clf, feature_names=[f"Pixel {i}" for i in range(train_images.shape[1])],
                       class_names=class_names,
                       output_path="image_results/decision_tree.png")
    print("[Info] Decision Tree visualization saved.")

    # Save the model
    model_path = os.path.join("results", "decision_tree_model.pkl")
    joblib.dump(clf, model_path)
    print(f"[Info] Model saved to {model_path}")

if __name__ == "__main__":
    main()