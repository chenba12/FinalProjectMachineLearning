
# **Machine learning: Facial Expression Recognition Project**  
**By Beni Tibi, and Chen Ben Ami**  
**Ariel University**

---

## **Overview**

This repository showcases a **facial expression recognition** system using **deep learning**. The system compares **four** main approaches:

1. **Decision tree model**  
   - A simple baseline that always predicts the most frequent class.
2. **Random Forest**
    - An ensemble of decision trees for improved performance.
3. **Basic Fully Connected Network (MLP)**  
   - A simple neural network flattening the image and using fully connected layers.
4. **Advanced Convolutional Neural Network (CNN)**  
   - Multiple convolutional blocks (Conv2D, BatchNorm, ReLU, MaxPool, Dropout) for superior performance.

Each model is trained to recognize **7 emotion classes** (angry, disgust, fear, happy, neutral, sad, surprise) from grayscale images sized **48×48**. After training, we demonstrate a **live inference** script where the advanced CNN classifies expressions in **real time** via a webcam feed.

---

## **Purpose**

1 **Comparison**  
   - Evaluate accuracy, precision, recall for each approach.  
   - Show that the **CNN** typically outperforms simpler models on an image-based problem.

2 **Practical Flow**  
   - **Data Download** (from Kaggle)  
   - **Data Preparation** (resize, grayscale, normalization)  
   - **Training** multiple models  
   - **Evaluation** (metrics)  
   - **Live Inference** using the advanced model.

---

## **Dataset**

- **Source**: [Kaggle - Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)  
- Contains 7 categories in subfolders: e.g., `train/angry/`, `train/happy/`, etc.
- Images are manually split into `train/`, `validation/`, and `test/` subfolders.

---

## **Installation Instructions**

1. **Clone the repository** (or download):
   ```bash
   git clone https://github.com/YourUserName/facial_expression_recognition_project.git
   cd facial_expression_recognition_project
   ```

2. **(Optional) Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   ```

3. **Install required libraries**:
   ```bash
   pip install -r requirements.txt
   ```
   - This includes **PyTorch**, **NumPy**, **Matplotlib**, **OpenCV**, **scikit-learn**, **kaggle**, etc.

4. **Kaggle Setup** (if you need to download the dataset):
   - Place your `kaggle.json` credentials in `~/.kaggle/` (Linux/Mac) or `%HOMEPATH%\.kaggle\` (Windows).  
   - Ensure correct permissions (e.g., `chmod 600 ~/.kaggle/kaggle.json`).

5. **Download dataset** (if required):
   ```bash
   python download_dataset.py
   ```
   - This will fetch the Face Expression Recognition dataset and unzip it under `data/face-expression-recognition-dataset/`.

6. **Prepare the `.pt` data** (preprocessing):
   ```bash
   python dataset_preparation.py
   ```
   - Reads raw images from `train/`, `validation/`, `test/`, converts them to **(N,1,48,48)** shape, normalizes, and saves `.pt` files to `processed_data/`.

---

## **How to Run the Models**

1. **Decision Tree**  
   ```bash
   python decision_tree.py
   ```
    - Trains the baseline model, logs classification report, saves results in `decision_tree_results.txt`.

2. **Random Forest**  
   ```bash
   python random_forest.py
   ```
   - Trains the ensemble model, logs classification report, saves results in `random_forest_results.txt`.

3. **Basic NN (MLP)**  
   ```bash
   python basic_nn.py
   ```
   - Multi-layer perceptron, logs classification report, saves results in `basic_nn_results.txt`.

4. **Advanced CNN**  
   ```bash
   python advanced_network.py
   ```
   - Trains the deeper convolutional network, saves best weights to `advanced_model.pth`.

5. **Live Inference**  
   ```bash
   python live_inference.py
   ```
   - Must have `advanced_model.pth` from the advanced CNN.  
   - Opens webcam, detects face, classifies expression in real time. Press `q` to quit.

---

## **Contact / Credits**

- **Authors**:  
  - Chen Ben Ami
  - Beni Tibi  
- **Institution**: *Ariel University*  

**Kaggle Reference**:  
[Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)

---
## **Future Ideas**

1. **Data Augmentation** (random flips/rotations) to improve generalization.  
2. **Class Weights** or **oversampling** for underrepresented expressions (e.g., disgust).  
3. **Transfer Learning** from pretrained networks (like ResNet) for potentially higher accuracy.  
4. **Hyperparameter Tuning** (learning rate, dropout rates, layer dimensions).
---
