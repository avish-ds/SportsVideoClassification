import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*iCCP: known incorrect sRGB profile.*")
import pickle
import cv2
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*iCCP: known incorrect sRGB profile.*")

# Import the model and load the weights
from keras.models import load_model
model = load_model(r"D:\Sports Video Classification\SportsVideoClassification\videoClassificationModel")

# Load the label binarizer
lb = pickle.loads(open(r"D:\Sports Video Classification\SportsVideoClassification\videoClassificationBinarizer.pickle", 'rb').read())

# Load the test data
data_path = r"D:\data"
sports_labels = set(['boxing','swimming','table_tennis','cricket','football','hockey'])
pathToImages = list(paths.list_images(data_path))
data = []
labels = []
for images in pathToImages:
    label = images.split(os.path.sep)[-2]
    if label not in sports_labels:
        continue
    image = cv2.imread(images)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image = cv2.resize(image, (244, 224))
    data.append(image)
    labels.append(label)
data = np.array(data)
labels = np.array(labels)
labels = lb.transform(labels)

# Make predictions
predictions = model.predict(data)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(labels, axis=1)

# Calculate performance metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1_score = f1_score(y_true, y_pred, average='weighted')

# Calculate ROC AUC score
n_classes = len(lb.classes_)
roc_auc_scores = []
for i in range(n_classes):
    roc_auc_scores.append(roc_auc_score(labels[:, i], predictions[:, i]))
mean_roc_auc = np.mean(roc_auc_scores)

# Display the performance metrics
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 Score: {:.2f}%".format(f1_score * 100))
print("ROC AUC: {:.2f}%".format(mean_roc_auc * 100))

# Generate and display the classification report
target_names = lb.classes_
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))

# Plot accuracy as a bar graph
categories = lb.classes_
accuracy_values = [accuracy * 100]
plt.bar(categories, accuracy_values)
plt.title('Accuracy')
plt.xlabel('Categories')
plt.ylabel('Accuracy (%)')
plt.show()
