import numpy as np
import cv2
import os
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# LOAD DATASET
# -------------------------------
data = []
labels = []
IMG_SIZE = 128
dataset_path = "dataset"

print("📁 Categories found:", os.listdir(dataset_path))

for category in os.listdir(dataset_path):
    path = os.path.join(dataset_path, category)

    # Robust labeling
    if category.lower() == "normal":
        label = 0
    else:
        label = 1

    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)
            img_array = cv2.imread(img_path)

            if img_array is None:
                continue

            img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append(img_array)
            labels.append(label)

        except:
            pass

data = np.array(data) / 255.0
labels = np.array(labels)

print("✅ Number of images loaded:", len(data))
print("📊 Label distribution:", np.bincount(labels))

# -------------------------------
# PREPARE DATA
# -------------------------------
data_flat = data.reshape(len(data), -1)

X_train, X_test, y_train, y_test = train_test_split(
    data_flat, labels, test_size=0.2, random_state=42
)

# -------------------------------
# ML MODELS
# -------------------------------

# Logistic Regression
lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# SVM
svm = make_pipeline(
    StandardScaler(),
    SVC(kernel='linear', probability=True)
)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# KNN
knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# -------------------------------
# PRINT METRICS
# -------------------------------
print("\n📊 Model Accuracies:")
print("Logistic Regression:", accuracy_score(y_test, y_pred_lr))
print("SVM:", accuracy_score(y_test, y_pred_svm))
print("KNN:", accuracy_score(y_test, y_pred_knn))
print("Random Forest:", accuracy_score(y_test, y_pred_rf))

print("\n📋 Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("\n📊 Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# -------------------------------
# SAVE MODELS
# -------------------------------
joblib.dump(lr, "lr_model.pkl")
joblib.dump(svm, "svm_model.pkl")
joblib.dump(knn, "knn_model.pkl")
joblib.dump(rf, "rf_model.pkl")

# -------------------------------
# SAVE METRICS
# -------------------------------
metrics = {
    "Logistic Regression": float(accuracy_score(y_test, y_pred_lr)),
    "SVM": float(accuracy_score(y_test, y_pred_svm)),
    "KNN": float(accuracy_score(y_test, y_pred_knn)),
    "Random Forest": float(accuracy_score(y_test, y_pred_rf))
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("\n✅ Training complete. Models + metrics saved.")