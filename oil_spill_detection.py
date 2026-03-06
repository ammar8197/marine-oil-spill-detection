import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

data = []
labels = []

dataset_path = "dataset"

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    for img in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img)

        image = cv2.imread(img_path)
        image = cv2.resize(image, (64,64))
        image = image.flatten()

        data.append(image)
        labels.append(folder)

X = np.array(data)
y = np.array(labels)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestClassifier()

model.fit(X_train,y_train)

pred = model.predict(X_test)

print(classification_report(y_test,pred))
