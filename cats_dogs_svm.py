# Cats vs Dogs Classification using SVM

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# 1. Image size
IMG_SIZE = 64

# 2. Dataset path
DATA_DIR = "data"


# 3. Load images and labels
def load_data():
    X = []
    y = []

    for label, folder in enumerate(["cats", "dogs"]):
        folder_path = os.path.join(DATA_DIR, folder)

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.flatten()   # Convert image to 1D array

            X.append(img)
            y.append(label)

    return np.array(X), np.array(y)


# 4. Load dataset
X, y = load_data()


# 5. Normalize pixel values
X = X / 255.0


# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 7. Train SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)


# 8. Predictions
y_pred = model.predict(X_test)


# 9. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
