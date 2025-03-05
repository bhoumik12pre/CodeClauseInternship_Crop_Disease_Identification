import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Set dataset path
dataset_path = r"D:/Virtual_Internship/CodeClause/Crop/PlantVillage"
img_size = 224

def load_images_and_labels(dataset_path):
    images, labels = [], []
    class_names = os.listdir(dataset_path)  # Folder names as labels

    for class_index, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)

        if not os.path.isdir(class_path):
            continue  # Skip non-directory files

        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                continue  # Skip unreadable images

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype(np.float32) / 255.0  # Normalize

            images.append(img)
            labels.append(class_index)

    return np.array(images), np.array(labels), class_names

# Load dataset
X, y, class_labels = load_images_and_labels(dataset_path)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = to_categorical(y_train, num_classes=len(class_labels))
y_test = to_categorical(y_test, num_classes=len(class_labels))

np.save("model/X_train.npy", X_train)
np.save("model/X_test.npy", X_test)
np.save("model/y_train.npy", y_train)
np.save("model/y_test.npy", y_test)

print(f"Dataset loaded: {X_train.shape[0]} train, {X_test.shape[0]} test")
