from keras.models import load_model
from collections import deque
import numpy as np
import pickle
import cv2
# from ex2 import *


#loading the model first
def process_video(path):
    model=load_model(r"C:\SportsVideoClassification\videoClassificationModel")
    lb=pickle.loads(open(r"C:\SportsVideoClassification\videoClassificationBinarizer.pickle","rb").read())
    outputVideo1=r"C:\SportsVideoClassification\outputVideos\output04.avi"
    mean=np.array([123.68,116.779,103.939][::1],dtype="float32")
    Queue=deque(maxlen=128)
    capture_video=cv2.VideoCapture(path)
    writer=None
    (Width,Height)=(None,None)
    frame_no=1
    while True:
        (taken,frame)=capture_video.read()
        if not taken:
            break
        if frame is None:
            continue
        if frame_no%10==0:
            if Width is None or Height is None:
                (Width,Height)=frame.shape[:2]
            output=frame.copy()
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame=cv2.resize(frame,(244,224)).astype("float32")
            frame-=mean
            predictions=model.predict(np.expand_dims(frame,axis=0))[0]
            Queue.append(predictions)
            results=np.array(Queue).mean(axis=0)
            i=np.argmax(results)
            label=lb.classes_[i]
            text="Sport is:{}".format(label)
            print(text)
            # obj=SportsVideoClassification()
            # obj.item="Sport is Table Tennis"
            # print(text)
            cv2.putText(output,text,(45,60),cv2.FONT_HERSHEY_COMPLEX,1.25,(255,0,0),5)
    
            if writer is None:
                fourcc=cv2.VideoWriter_fourcc(*"MJPG")
                writer=cv2.VideoWriter("outputVideo1",fourcc,30,(Width,Height),True)
            writer.write(output)
            cv2.imshow("Working",output)
            key=cv2.waitKey(1)&0xFF
            if key==ord("q"):
                break
        frame_no+=1
    print("Finalizing")
    writer.release()
    capture_video.release()
process_video(r"C:\SportsVideoClassification\trainingVideos\video2.mp4")