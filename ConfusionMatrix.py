import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
import os
import cv2
import pickle
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.layers.pooling import AveragePooling2D
from keras.applications import resnet
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.models import Model,load_model

if os.path.exists(r"D:\\Sports Video Classification"):
            base_dir = r"D:\Sports Video Classification\SportsVideoClassification"
elif os.path.exists(r"D:\\"):
            base_dir = r"D:\SportsVideoClassification"
else:
            base_dir=r"C:\SportsVideoClassification"

# Load the saved model and label binarizer
model=load_model(os.path.join(base_dir, "videoClassificationModel"))
lb=pickle.loads(open(os.path.join(base_dir, "videoClassificationBinarizer.pickle"),"rb").read())



if os.path.exists(r"D:\\"):
        data_path=r"D:\\data"
else:
        data_path=r"C:\\data"

sports_labels=set(['boxing','swimming','table_tennis','cricket','football','hockey'])
pathToImages=list(paths.list_images(data_path))
data=[]
labels=[]
for images in pathToImages:
    label=images.split(os.path.sep)[-2]
    if label not in sports_labels:
        continue
    image=cv2.imread(images)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
    image=cv2.resize(image,(244,224))
    data.append(image)
    labels.append(label)
data=np.array(data)
labels=np.array(labels)
lb=LabelBinarizer()
labels=lb.fit_transform(labels)

(x_train,x_test,y_train,y_test)=train_test_split(data,labels,test_size=0.25,stratify=labels,random_state=42)


# Make predictions on the test set
predictions = model.predict(x_test)

# Convert predictions and true labels from one-hot encoding to labels
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Generate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
accuracy = accuracy_score(true_labels, predicted_labels)

# Print the confusion matrix
print("Confusion Matrix:",cm)
print("Accuracy: {:.2f}%".format(accuracy * 100))
