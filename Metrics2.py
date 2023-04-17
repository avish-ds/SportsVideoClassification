import os
import pickle
import cv2
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score

# Layers of neural network
from keras.layers import Input
from keras.layers.pooling import AveragePooling2D
from keras.applications import resnet
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.models import Model
from keras.models import load_model
# Optimizer
from keras.optimizers import gradient_descent_v2

# Determine the base directory path based on the current computer's file system
if os.path.exists(r"D:\\"):
    base_dir = r"D:\Sports Video Classification\SportsVideoClassification"
else:
    base_dir = r"C:\Sports Video Classification\SportsVideoClassification"

# Construct the complete file paths using the base directory path
data_path = (r"D:\data")
output_model = os.path.join(base_dir, "videoClassificationModel")
output_binarizer = os.path.join(base_dir, "videoClassificationBinarizer.pickle")

# Load the trained model and label binarizer
model = load_model(output_model)
lb = pickle.loads(open(output_binarizer, "rb").read())

# Load the test data and labels
print("[INFO] Loading test data...")
test_data = []
test_labels = []
test_image_paths = sorted(list(paths.list_images(data_path)))
for image_path in test_image_paths:
    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = resnet.preprocess_input(image)
        test_data.append(image)
    else:
        print("Error in image")
        continue
    # Extract the class label from the image path and update the labels list
    label = image_path.split(os.path.sep)[-2]
    test_labels.append(label)

# Convert the test data and labels to NumPy arrays
test_data = np.array(test_data, dtype="float32")
test_labels = np.array(test_labels)

# Perform one-hot encoding on the test labels
test_labels = lb.transform(test_labels)
print("Test Labels:",test_labels)


unseen_labels = set(test_labels.argmax(axis=1)) - set(np.arange(len(lb.classes_)))
if len(unseen_labels) > 0:
    print(f"Warning: Unseen labels found in test set: {unseen_labels}")

# Make predictions on the test data
print("[INFO] Evaluating network...")
predictions = model.predict(test_data, batch_size=36)

# Convert the predictions to binary labels (0 or 1) based on threshold (e.g., 0.5)
binary_predictions = np.argmax(predictions, axis=1)

# Convert the binary predictions and true labels back to class labels
binary_labels = lb.inverse_transform(binary_predictions)
true_labels = lb.inverse_transform(np.argmax(test_labels, axis=1))

# Calculate and print the confusion matrix
cm = confusion_matrix(true_labels, binary_labels)
print("Confusion Matrix:")
print(cm)

# Calculate and print the accuracy
accuracy = accuracy_score(true_labels, binary_labels)
print("Accuracy: {:.2f}%".format(accuracy * 100))
