import os
import pickle
import cv2
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

#layers of neural network
from keras.layers import Input
from keras.layers.pooling import AveragePooling2D
from keras.applications import resnet
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.models import Model

#optimizer
from keras.optimizers import gradient_descent_v2

data_path=r"D:\data"
output_model=r"D:\Sports Video Classification\SportsVideoClassification\videoClassificationModel"
output_binarizer=r"D:\Sports Video Classification\SportsVideoClassification\videoClassificationBinarizer"
epoch=15
sports_labels=set(['boxing','swimming','table_tennis','cricket','football','hockey'])
print("Images being loaded")
pathToImages=list(paths.list_images(data_path))
data=[]
labels=[]
for images in pathToImages:
    label=images.split(os.path.sep)[-2]
    if label not in sports_labels:
        continue
    image=cv2.imread(images)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
    image=cv2.resize(image,(244,244))
    data.append(image)
    labels.append(label)
data=np.array(data)
labels=np.array(labels)
lb=LabelBinarizer()
labels=lb.fit_transform(labels)
(x_train,x_test,y_train,y_test)=train_test_split(data,labels,test_size=0.25,stratify=labels,random_state=42)
trainingAugmentation=ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)
validationAugmentation=ImageDataGenerator()
mean=np.array([123.68,116.779,103.939],dtype="float32")
trainingAugmentation.mean=mean
validationAugmentation.mean=mean

#base model trained by keras
model=resnet.ResNet50
baseModel=model(weights="imagenet",include_top=False,input_tensor=Input(shape=(224,244,3)))
headModel=baseModel.output
headModel=AveragePooling2D(pool_size=(7,7))(headModel)
headModel=Flatten(name="flatten")(headModel)
headModel=Dense(512,activation="relu")(headModel)
headModel=Dropout(0.5)(headModel)
headModel=Dense(len(lb.classes_),activation="softmax")(headModel)
model=Model(inputs=baseModel.input,outputs=headModel)


for baseModelLayers in baseModel.layers:
    baseModelLayers.trainable=False
opt=gradient_descent_v2.SGD(lr=0.0001,momentum=0.9,decay=1e-4/epoch)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
History=model.fit_generator(
    trainingAugmentation.flow(x_train,y_train,batch_size=32),
    steps_per_epoch=len(x_train)//32,
    validation_data=validationAugmentation.flow(x_test,y_test),
    validation_steps=len(x_test)//32,
    epochs=epoch
)


model.save(output_model)
lbinarizer=open(r"D:\Sports Video Classification\SportsVideoClassification\videoClassificationBinarizer.pickle","wb")
lbinarizer.write(pickle.dumps(lb))
lbinarizer.close()




