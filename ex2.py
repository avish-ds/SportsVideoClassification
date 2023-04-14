import sys
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QSplitter, QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QFileDialog, QVBoxLayout, QHBoxLayout, QHeaderView, QTableWidgetItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
from PyQt5.QtGui import QImage
import numpy as np
# import TestOutput
from keras.models import load_model
import pickle
from collections import deque

class SportsVideoClassification(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(1300,590)
        self.item=1234321
       
        self.setStyleSheet('background-color: #f2d8ee;')
        # UI elements for selecting and playing a video
        self.input_path_textfield = QLineEdit(self)
        self.input_path_textfield.setReadOnly(True)
        self.input_path_textfield.setGeometry(QtCore.QRect(10, 450, 400, 23))
        self.input_path_button = QPushButton('Select Video', self)
        self.input_path_button.clicked.connect(self.select_video)
        self.input_path_button.setGeometry(QtCore.QRect(150, 490, 75, 23))

        self.line = QtWidgets.QFrame(self)
        self.line.setGeometry(QtCore.QRect(420, 0, 31, 571))
        self.line.setSizeIncrement(QtCore.QSize(0, 0))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")

        self.line2 = QtWidgets.QFrame(self)
        self.line2.setGeometry(QtCore.QRect(835, 0, 31, 571))
        self.line2.setSizeIncrement(QtCore.QSize(0, 0))
        self.line2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line2.setObjectName("line")

        font = self.input_path_button.font()
        font.setWeight(QFont.Bold)
        self.input_path_button.setFont(font)

        self.video_player = QLabel(self)
        self.video_player.setAlignment(Qt.AlignCenter)
        self.video_player.setGeometry(QtCore.QRect(10, 20, 420, 361))

        self.video_player2 = QLabel(self)
        self.video_player2.setAlignment(Qt.AlignCenter)
        self.video_player2.setGeometry(QtCore.QRect(870, 20, 420, 361))

        # UI elements for displaying and editing video classification
        self.labelTable = QTableWidget(self)
        self.labelTable.setColumnCount(2)
        self.labelTable.setRowCount(3)
        self.labelTable.setHorizontalHeaderLabels(['Sports label', 'Timings'])
        self.labelTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.labelTable.verticalHeader().setVisible(False)
        # self.labelTable.setEditTriggers(QTableWidget.NoEditTriggers)
        # self.input_path_button.setGeometry(QtCore.QRect(290, 490, 75, 23))
        self.labelTable.setGeometry(QtCore.QRect(440, 20, 401, 361))

        # Put the UI elements together in a vertical layout
        # top_section = QWidget(self)
        # top_section_layout = QVBoxLayout(top_section)
        # top_section_layout.addWidget(self.input_path_textfield)
        # top_section_layout.addWidget(self.input_path_button)
        # top_section_layout.addWidget(self.video_player)
        # top_section.setLayout(top_section_layout)
        # top_section.setFixedWidth(100)

        # bottom_section = QWidget(self)
        # bottom_section_layout = QVBoxLayout(bottom_section)
        # bottom_section_layout.addWidget(self.labelTable)
        # bottom_section.setLayout(bottom_section_layout)

        # splitter = QSplitter(Qt.Vertical, self)
        # splitter.addWidget(top_section)
        # splitter.addWidget(bottom_section)

        # layout = QHBoxLayout(self)
        # layout.addWidget(splitter)
        # self.setLayout(layout)
    def set_item(self):
        self.labelTable.setItem(1,1,QtWidgets.QTableWidgetItem(str(1)))
        print("yes")

    def select_video(self):
        # Use QFileDialog to prompt the user to select a video file
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(self,"Select Video File", "","Video Files (*.mp4 *.avi *.mkv *.mov *.flv *.wmv *.mpeg *.mpg *.3gp *.webm)", options=options)
        if file_path:
            self.input_path_textfield.setText(file_path)
            
            # ex3.set1()
            self.labelTable.setItem(1,1,QtWidgets.QTableWidgetItem(str(self.item)))
            self.process_video(file_path)
            self.play_video(file_path)
            self.labelTable.setItem(2,1,QtWidgets.QTableWidgetItem(str(self.item)))

    

    def play_video(self, file_path):
        # Use OpenCV to read the selected video file and play it in the video player
        cap = cv2.VideoCapture(file_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel =frame.shape
            bytes_per_line = 3 * width
            q_img = QPixmap.fromImage(QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888))
            self.video_player.setPixmap(q_img.scaled(self.video_player.width(), self.video_player.height(), Qt.KeepAspectRatio))
            # self.video_player.setPixmap(q_img.scaled(self.video_player.width(), self.video_player.height(), Qt.KeepAspectRatio))   
        cap.release()
        QApplication.processEvents()
    
    def display(self,taken,frame,width,height):
        
        print("Function running")
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        print("Frame recieved")
        bytes_per_line=3*width
        q_img = QPixmap.fromImage(QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888))
        self.video_player.setPixmap(q_img.scaled(self.video_player.width(), self.video_player.height(), Qt.KeepAspectRatio))
        QApplication.processEvents()



    def process_video(self,path):
        # ex2Object=SportsVideoClassification()
        model=load_model(r"D:\Sports Video Classification\SportsVideoClassification\videoClassificationModel")
        lb=pickle.loads(open(r"D:\Sports Video Classification\SportsVideoClassification\videoClassificationBinarizer.pickle","rb").read())
        outputVideo1=r"D:\SportsVideoClassification\SportsVideoClassification\outputVideos\output04.avi"
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
                print("Frame is none")
                continue
            if frame_no%500==0:
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
                # ex2Object.display(taken,frame,Width,Height)
            


                # q_img = QPixmap.fromImage(QImage(frame.data, Width, Height, 3*Width, QImage.Format_RGB888))
                # ex2Object.video_player2.setPixmap(q_img.scaled(ex2Object.video_player2.width(), ex2Object.video_player2.height(), Qt.KeepAspectRatio))



                # obj=SportsVideoClassification()
                # obj.item="Sport is Table Tennis"
                # print(text)
                cv2.putText(output,text,(45,60),cv2.FONT_HERSHEY_COMPLEX,1.25,(255,0,0),5)
    
                if writer is None:
                    fourcc=cv2.VideoWriter_fourcc(*"MJPG")
                    writer=cv2.VideoWriter("outputVideo4",-1,30,(244,224),True)
                    if writer is None:
                        print("Video not supported!")
                        return
                writer.write(output)
                # cv2.imshow("Working",output)
                bytes_per_line = 3 * Width
                output = cv2.resize(output, (self.video_player2.width(), self.video_player2.height()))
                q_img = QPixmap.fromImage(QImage(output.data, output.shape[1], output.shape[0], output.strides[0], QImage.Format_RGB888))
                self.video_player2.setPixmap(q_img.scaled(self.video_player.width(), self.video_player.height(), Qt.KeepAspectRatio))
                QApplication.processEvents()


                
                
                
                key=cv2.waitKey(1)&0xFF
                if key==ord("q"):
                    break
            frame_no+=1
        print("Finalizing")
        
        writer.release()
        capture_video.release()

    
  

    
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SportsVideoClassification()
    window.show()
    sys.exit(app.exec_())

